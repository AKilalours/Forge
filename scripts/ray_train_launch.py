"""Launch the same distributed training step under Ray Train instead of torchrun.

    python scripts/ray_train_launch.py --workers 2          # CPU, gloo
    python scripts/ray_train_launch.py --workers 2 --gpu    # untested here; see the caveat

WHAT RAY IS FOR HERE, AND WHAT IT IS NOT. Ray Train's job is to START and SUPERVISE a
distributed torch job: it places workers, sets up the process group, hands each worker its
rank, and tears the whole thing down when one dies. That is a different concern from FSDP
and DeepSpeed, which decide how a model is SHARDED once training is already running. So Ray
is not a third strategy to benchmark against those two; it is a replacement for `torchrun`.

WHICH IS WHY THIS SCRIPT MEASURES ALMOST NOTHING. It runs on CPU with the gloo backend, on
one machine, with a deliberately tiny model. The throughput number it produces is
meaningless and the artifact says so in its own fields. What it demonstrates is that the
integration WORKS: that ray_scaling_config's output is accepted by TorchTrainer, that each
worker gets a distinct rank and a shared process group, and that the global-batch invariant
this repository enforces everywhere survives being launched by a different launcher.

THE HONEST LIMIT, STATED ONCE. This has never run on GPU and never on more than one
machine, which is the only situation where Ray earns its place. A single-node CPU run is
evidence that the wiring is correct and evidence of nothing else. docs/jd_coverage.md says
the same thing in the row it belongs to.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

OUT = Path("reports/experiments/ray")
TARGET_GLOBAL_BATCH = 64
PER_DEVICE_BATCH = 16
SEQ_LEN = 128        # tiny on purpose: this is a wiring check, not a throughput measurement
HIDDEN = 128
LAYERS = 2


def train_loop_per_worker(config: dict) -> dict:
    """Runs once per Ray worker, and RETURNS its measurements.

    Imports live here, not at module scope, because Ray serialises this function to the
    workers and the driver should not need torch loaded to submit the job.

    WHY IT RETURNS RATHER THAN ONLY REPORTING. Ray Train V2 (this is 2.58) does not put
    worker metrics on Result.metrics; that attribute does not exist on Result at all, which
    is why the first version of this script read None for every field and still wrote a
    confident artifact. V2 surfaces rank 0's RETURN VALUE as result.return_value, and
    train.report streams metrics for checkpointing and progress rather than for handing a
    value back to the driver. So the measurements are returned, and reported as well
    because reporting is what makes them visible in Ray's own UI and logs.
    """
    import torch
    import torch.distributed as dist
    from ray import train

    ctx = train.get_context()
    rank, world_size = ctx.get_world_rank(), ctx.get_world_size()

    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Embedding(1000, HIDDEN),
        *[torch.nn.TransformerEncoderLayer(HIDDEN, 4, HIDDEN * 2, batch_first=True)
          for _ in range(LAYERS)],
        torch.nn.Linear(HIDDEN, 2),
    )
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    accum = config["grad_accum"]
    ids = torch.randint(0, 1000, (PER_DEVICE_BATCH, SEQ_LEN))
    labels = torch.randint(0, 2, (PER_DEVICE_BATCH,))

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            out = model(ids).mean(dim=1)
            (torch.nn.functional.cross_entropy(out, labels) / accum).backward()
        optimizer.step()

    for _ in range(2):          # warmup
        one_step()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(config["steps"]):
        one_step()
    dist.barrier()
    wall = time.perf_counter() - t0

    measured = {
        "rank": rank,
        "world_size": world_size,
        "grad_accum": accum,
        "global_batch_size": PER_DEVICE_BATCH * accum * world_size,
        "wall_seconds": round(wall, 3),
        "backend": dist.get_backend(),
    }
    train.report(measured)      # visible in Ray's logs and UI
    return measured             # what actually reaches the driver in Train V2


REQUIRED_WORKER_METRICS = ("backend", "global_batch_size", "wall_seconds", "world_size")


def validate_worker_metrics(metrics: dict, workers: int, target_global_batch: int) -> list[str]:
    """Return the reasons this run may NOT be written as evidence. Empty list means it may.

    WHY THIS IS A FUNCTION AND NOT AN IF-BLOCK. The first version of this script built its
    "what_this_shows" sentence as a hardcoded string and filled the supporting fields with
    metrics.get(...), which returns None on a miss. Ray Train V2 does not populate
    Result.metrics at all, so every field came back None and the script wrote an artifact
    asserting that Ray placed the workers and built the process group, with nothing behind
    it, and exited 0.

    A claim whose every supporting field is null is not a weak result, it is a fabricated
    one. Pulling the check out here means it can be tested without a Ray cluster, which is
    the difference between a rule and an intention.
    """
    missing = [k for k in REQUIRED_WORKER_METRICS if metrics.get(k) is None]
    if missing:
        return [
            "Ray Train finished but the workers reported nothing usable: "
            f"{missing} are missing. Without them there is no evidence the process group "
            "was built or the step ran, and an artifact claiming otherwise would be a "
            "fabrication."
        ]
    problems = []
    if metrics["world_size"] != workers:
        problems.append(
            f"asked Ray for {workers} workers, the run reports {metrics['world_size']}; "
            "the scaling config was not honoured"
        )
    if metrics["global_batch_size"] != target_global_batch:
        problems.append(
            f"global batch came back as {metrics['global_batch_size']}, not "
            f"{target_global_batch}; changing the launcher must not change what is trained"
        )
    return problems


def launch(workers: int, steps: int, use_gpu: bool) -> dict:
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    from forge.training.distributed import DistConfig, ray_scaling_config
    from forge.training.scaling import assert_global_batch_invariant, grad_accum_for_world_size

    accum = grad_accum_for_world_size(TARGET_GLOBAL_BATCH, PER_DEVICE_BATCH, workers)
    # The same invariant every other launcher in this repo holds. Ray changing WHO starts
    # the workers must not change WHAT they train on.
    assert_global_batch_invariant(
        [{"per_device_batch": PER_DEVICE_BATCH, "grad_accum": accum, "world_size": workers}],
        label=f"ray workers={workers}",
    )

    cfg = DistConfig(strategy="ray", world_size=workers,
                     per_device_batch=PER_DEVICE_BATCH, grad_accum=accum)
    plan = ray_scaling_config(cfg, use_gpu=use_gpu)

    # log_to_driver stays ON. The first version silenced it, which hid exactly the worker
    # output that would have explained why nothing came back.
    ray.init(ignore_reinit_error=True)
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"grad_accum": accum, "steps": steps},
        # Built from the generator this repo already had and already tested, rather than
        # hand-written here. That generator had never been handed to Ray until now.
        scaling_config=ScalingConfig(
            num_workers=plan["num_workers"],
            use_gpu=plan["use_gpu"],
            resources_per_worker={"CPU": 1} if not use_gpu else plan["resources_per_worker"],
            placement_strategy=plan["placement_strategy"],
        ),
        run_config=RunConfig(storage_path=str(Path(OUT).resolve() / "ray_results"),
                             name=f"forge_ray_ws{workers}"),
    )
    result = trainer.fit()
    ray.shutdown()

    # Train V2: rank 0's return value, not Result.metrics, which does not exist here.
    metrics = result.return_value or {}

    problems = validate_worker_metrics(metrics, workers, TARGET_GLOBAL_BATCH)
    if problems:
        raise SystemExit(
            "\n".join(problems)
            + f"\nresult.return_value = {result.return_value!r}"
            + f"\nresult fields        = {[a for a in dir(result) if not a.startswith('_')]}"
        )

    out = {
        "launcher": "ray.train.torch.TorchTrainer",
        "workers": workers,
        "use_gpu": use_gpu,
        "backend": metrics.get("backend"),
        "global_batch_size": metrics.get("global_batch_size"),
        "grad_accum": accum,
        "per_device_batch": PER_DEVICE_BATCH,
        "steps": steps,
        "wall_seconds": metrics.get("wall_seconds"),
        "host": f"{platform.processor() or platform.machine()} ({platform.system()})",
        "scaling_config": plan,
        # Written only after the asserts above pass, so each clause is backed by a field
        # in this same file rather than by the author's expectation.
        "what_this_shows": (
            f"Ray Train placed {metrics['world_size']} workers and built a "
            f"{metrics['backend']} process group; ray_scaling_config's output was accepted "
            f"by TorchTrainer; and the global batch came back as "
            f"{metrics['global_batch_size']}, the same value torchrun produces, so changing "
            "the launcher did not change what is trained."
        ),
        "what_this_does_not_show": (
            "Anything about throughput. This is CPU, gloo, one machine, and a two-layer toy "
            "model with seq_len 128. Ray earns its place on multiple machines, which this "
            "has never run on. Do not read wall_seconds as a performance result."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"ray_ws{workers}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "scaling_config"}, indent=1))
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    launch(a.workers, a.steps, a.gpu)
