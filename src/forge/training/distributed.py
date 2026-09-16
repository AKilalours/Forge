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


def deepspeed_config(cfg: DistConfig, lr: float, warmup_steps: int = 500,
                     total_num_steps: int | None = None) -> dict[str, Any]:
    """Emit a DeepSpeed JSON config.

    `train_batch_size` here is DeepSpeed's GLOBAL batch and it validates
    train_batch_size == micro_batch * grad_accum * world_size internally. Deriving it
    rather than hardcoding it is what stops a benchmark drifting between strategies.

    THE SCHEDULER IS OPTIONAL, AND THAT IS A BUG FIX. This function used to emit a
    WarmupDecayLR block unconditionally, carrying warmup_min_lr, warmup_max_lr and
    warmup_num_steps. DeepSpeed's WarmupDecayLR ALSO requires total_num_steps, so the
    config was invalid and deepspeed.initialize() raised:

        TypeError: WarmupDecayLR.__init__() missing 1 required positional argument:
                   'total_num_steps'

    Every test here asserted the shape of the returned dict. None handed it to DeepSpeed,
    because that needs the library and a GPU. So a config generator whose output the
    target rejects sat in the repo looking tested. That is the same failure as everything
    else this project has turned up: the thing was written carefully and never executed.

    total_num_steps=None now emits NO scheduler rather than an invalid one. A throughput
    benchmark does not want a schedule anyway: a decaying learning rate changes the
    numbers the optimizer produces and changes nothing about how long a step takes, so
    including one adds a variable without adding information. Real training passes the
    step count and gets the schedule.
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
        "activation_checkpointing": {"partition_activations": cfg.gradient_checkpointing},
        "steps_per_print": 100,
        "wall_clock_breakdown": True,   # so profiling has something to read
    }
    if total_num_steps is not None:
        if total_num_steps < warmup_steps:
            raise ValueError(
                f"total_num_steps ({total_num_steps}) is below warmup_steps "
                f"({warmup_steps}); the schedule would warm up past the end of training"
            )
        conf["scheduler"] = {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": warmup_steps,
                # REQUIRED by WarmupDecayLR. Omitting it is what made this config invalid.
                "total_num_steps": total_num_steps,
            },
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



class DistributedRun:
    """One training step's worth of behaviour, so the harness cannot get it wrong.

    WHY THIS EXISTS, AND IT IS NOT STYLE. FSDP and DeepSpeed divide the work of a
    gradient-accumulation step differently, and the difference is silent when you get it
    wrong:

      FSDP      you scale the loss by 1/accum yourself, call loss.backward() every
                micro-batch, and call optimizer.step() once at the boundary.
      DeepSpeed the engine owns accumulation. You call engine.backward(raw_loss) and
                engine.step() every micro-batch, and the engine scales by
                gradient_accumulation_steps internally and no-ops the optimizer until the
                boundary.

    Write the FSDP loop and point it at a DeepSpeed engine and you scale the loss twice,
    so the effective learning rate is off by a factor of accum. Write the DeepSpeed loop
    and point it at FSDP and you take accum optimizer steps per batch, so the global batch
    is actually per_device_batch and every number in the comparison is wrong. Neither
    raises. Both produce a plausible throughput figure, which is the worst outcome for a
    benchmark whose entire purpose is comparing two strategies fairly.

    forge.training.scaling exists to stop the global batch drifting between world sizes.
    This class is the same guarantee across STRATEGIES: the caller feeds micro-batch
    losses and never touches the optimizer, so there is one place the semantics live.
    """

    def __init__(self, model, strategy: str, grad_accum: int, optimizer=None, engine=None):
        self.model = model
        self.strategy = strategy
        self.grad_accum = grad_accum
        self.optimizer = optimizer
        self._engine = engine
        self._seen = 0

    @property
    def scales_loss_internally(self) -> bool:
        """True when the backend divides by grad_accum for you. Dividing again is the bug."""
        return self.strategy == "deepspeed"

    def micro_step(self, loss) -> bool:
        """Consume one micro-batch's RAW, unscaled loss. True when the optimizer stepped.

        Pass the loss as the model returned it. Scaling is this method's job precisely
        because getting it wrong is invisible.
        """
        if self.strategy == "deepspeed":
            self._engine.backward(loss)      # scales by grad_accum itself
            self._engine.step()              # no-ops until the accumulation boundary
            self._seen += 1
            return bool(self._engine.is_gradient_accumulation_boundary())

        (loss / self.grad_accum).backward()
        self._seen += 1
        if self._seen % self.grad_accum:
            return False
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return True


def build_run(cfg: DistConfig, model, optimizer=None, lr: float = 2e-5,
              total_num_steps: int | None = None) -> DistributedRun:
    """Construct the strategy in `cfg` and return the object that drives its steps.

    This is the interface a benchmark should use. `build()` below returns a bare model,
    which is honest for FSDP and impossible for DeepSpeed, where deepspeed.initialize()
    returns an engine that owns the optimizer, the scheduler and the accumulation
    boundary. A function that returned only the model would be discarding the half of
    DeepSpeed that matters and leaving the caller to reinvent it.
    """
    if cfg.strategy in ("none", "fsdp"):
        return DistributedRun(
            model=build(cfg, model), strategy=cfg.strategy,
            grad_accum=cfg.grad_accum, optimizer=optimizer,
        )
    if cfg.strategy != "deepspeed":
        raise NotImplementedError(
            f"strategy {cfg.strategy!r} has a config generator here but no launcher. "
            "ray_scaling_config feeds Ray Train, which runs this whole script as a worker "
            "rather than wrapping the model, so it does not belong behind this call."
        )

    import deepspeed  # noqa: PLC0415 - optional [dist] extra, absent on CPU machines

    conf = deepspeed_config(cfg, lr=lr, total_num_steps=total_num_steps)
    engine, ds_optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=conf,
    )
    return DistributedRun(
        model=engine, strategy="deepspeed", grad_accum=cfg.grad_accum,
        optimizer=ds_optimizer, engine=engine,
    )

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
