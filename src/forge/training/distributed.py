"""Distributed strategies, and the configs they need.

Three backends behind one interface so the comparison is like for like:
  fsdp      PyTorch FullyShardedDataParallel, full_shard, the default
  deepspeed ZeRO stage 3, benchmarked against FSDP on the SAME global batch
  ray       multi-node orchestration and parallel mining sweeps

The config GENERATORS below are pure functions and are tested. The parts that construct
live process groups are not, because they need hardware.

Every generator refuses to emit a config whose global batch differs from the benchmark's,
via forge.training.scaling. See that module for why: a strategy comparison run at two
different effective batch sizes is two different experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from forge.training.scaling import ScalingError, global_batch

STRATEGIES = ("none", "fsdp", "deepspeed", "ray")


@dataclass(frozen=True)
class DistConfig:
    strategy: str = "fsdp"
    world_size: int = 1
    per_device_batch: int = 32
    grad_accum: int = 2
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    sharding: str = "full_shard"
    zero_stage: int = 3
    cpu_offload: bool = False

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}, expected one of {STRATEGIES}")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError(f"unknown precision {self.precision!r}")

    @property
    def global_batch_size(self) -> int:
        return global_batch(self.per_device_batch, self.grad_accum, self.world_size)


def deepspeed_config(cfg: DistConfig, lr: float, warmup_steps: int = 500) -> dict[str, Any]:
    """Emit a DeepSpeed JSON config.

    `train_batch_size` here is DeepSpeed's GLOBAL batch and it validates
    train_batch_size == micro_batch * grad_accum * world_size internally. Deriving it
    rather than hardcoding it is what stops a benchmark drifting between strategies.
    """
    if cfg.zero_stage not in (0, 1, 2, 3):
        raise ValueError("zero_stage must be 0, 1, 2 or 3")
    conf: dict[str, Any] = {
        "train_batch_size": cfg.global_batch_size,
        "train_micro_batch_size_per_gpu": cfg.per_device_batch,
        "gradient_accumulation_steps": cfg.grad_accum,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": cfg.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {"warmup_min_lr": 0, "warmup_max_lr": lr, "warmup_num_steps": warmup_steps},
        },
        "activation_checkpointing": {"partition_activations": cfg.gradient_checkpointing},
        "steps_per_print": 100,
        "wall_clock_breakdown": True,   # so profiling has something to read
    }
    if cfg.precision == "bf16":
        conf["bf16"] = {"enabled": True}
    elif cfg.precision == "fp16":
        conf["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}
    if cfg.cpu_offload:
        if cfg.zero_stage != 3:
            raise ValueError("cpu_offload requires zero_stage 3")
        conf["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        conf["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}
    return conf


def fsdp_plan(cfg: DistConfig) -> dict[str, Any]:
    """The FSDP settings, as data, so the two strategies can be diffed rather than
    described. Sharding a transformer at the LAYER boundary rather than wrapping the
    whole model is what makes FSDP actually save memory."""
    return {
        "sharding_strategy": cfg.sharding,
        "mixed_precision": cfg.precision,
        "auto_wrap_policy": "transformer_layer",
        "activation_checkpointing": cfg.gradient_checkpointing,
        "cpu_offload": cfg.cpu_offload,
        "limit_all_gathers": True,
        "use_orig_params": True,      # required for torch.compile and for param groups
        "backward_prefetch": "backward_pre",
    }


def ray_scaling_config(cfg: DistConfig, use_gpu: bool = True) -> dict[str, Any]:
    return {
        "num_workers": cfg.world_size,
        "use_gpu": use_gpu,
        "resources_per_worker": {"GPU": 1 if use_gpu else 0, "CPU": 4},
        "placement_strategy": "PACK",   # keep workers close; collectives are latency bound
    }


def benchmark_matrix(
    target_global_batch: int,
    world_sizes: tuple[int, ...] = (1, 2, 4),
    per_device_batch: int = 32,
    strategies: tuple[str, ...] = ("fsdp", "deepspeed"),
) -> list[DistConfig]:
    """Build a strategy-by-world-size matrix that all shares one global batch size.

    This is the harness for the Phase 7 exit criterion. Constructing it by hand is how
    the invariant gets broken.
    """
    from forge.training.scaling import grad_accum_for_world_size

    out: list[DistConfig] = []
    for ws in world_sizes:
        accum = grad_accum_for_world_size(target_global_batch, per_device_batch, ws)
        for s in strategies:
            out.append(
                DistConfig(strategy=s, world_size=ws, per_device_batch=per_device_batch,
                           grad_accum=accum)
            )
    sizes = {c.global_batch_size for c in out}
    if len(sizes) != 1:
        raise ScalingError(f"benchmark matrix is not batch-invariant: {sorted(sizes)}")
    return out


def _transformer_layer_class(model):
    """The class FSDP should shard at, found rather than hardcoded.

    Sharding at the transformer LAYER boundary is what makes FSDP save memory; wrapping
    the whole model shards nothing useful. HuggingFace encoders keep their layers in a
    ModuleList, so the class is readable off the first element instead of being a
    per-backbone constant that goes stale when the backbone changes.
    """
    for attr in ("encoder.encoder.layer", "encoder.layer", "layers", "encoder.layers"):
        node = model
        for part in attr.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and len(node):
            return type(node[0])
    return None


def build(cfg: DistConfig, model, optimizer=None):  # pragma: no cover - needs >1 GPU
    """Wrap `model` in the strategy `cfg` names. Returns the wrapped model.

    Only fsdp is live. deepspeed and ray are config generators here: their configs are
    produced and tested by this module, but launching them needs their own runtimes and
    entry points, and a half-wired launcher that silently falls back to single-device is
    worse than one that says it is not implemented.
    """
    if cfg.strategy == "none":
        return model
    if cfg.strategy != "fsdp":
        raise NotImplementedError(
            f"strategy {cfg.strategy!r} has a config generator here but no launcher. "
            "Use deepspeed_config with the deepspeed entry point, or ray_scaling_config "
            "with Ray Train."
        )

    import torch
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    plan = fsdp_plan(cfg)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.precision]
    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[plan["sharding_strategy"]]

    layer_cls = _transformer_layer_class(model)
    if layer_cls is not None:
        import functools

        policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
    else:
        # Falling back is reported, not silent: a size-based policy shards at arbitrary
        # boundaries and will not match the plan this module emitted.
        import functools
        import warnings

        warnings.warn(
            "no transformer layer ModuleList found; falling back to a size-based wrap "
            "policy, which does NOT match fsdp_plan's transformer_layer setting",
            stacklevel=2,
        )
        policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)

    return FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        limit_all_gathers=plan["limit_all_gathers"],
        use_orig_params=plan["use_orig_params"],
        device_id=torch.cuda.current_device(),
    )
