"""Measure training throughput for one strategy at one world size. Launched by torchrun.

    torchrun --nproc_per_node=1 scripts/scaling_run.py --strategy fsdp
    torchrun --nproc_per_node=2 scripts/scaling_run.py --strategy fsdp
    torchrun --nproc_per_node=4 scripts/scaling_run.py --strategy fsdp
    torchrun --nproc_per_node=2 scripts/scaling_run.py --strategy deepspeed
    python scripts/scaling_run.py --summarise --strategy fsdp

Run records are named <strategy>_ws<N>.json so two strategies cannot overwrite each
other's measurements, which is a mistake you discover only when the numbers look odd.

THE INVARIANT THIS HARNESS EXISTS TO HOLD. Every world size runs the SAME global batch,
with gradient accumulation absorbing the difference, because a comparison run at two
effective batch sizes is two experiments and its speedup number means nothing. That rule
and the arithmetic behind it live in forge.training.scaling; this script calls them rather
than reimplementing them, so the invariant is enforced by the module that documents it.

Gradient checkpointing is OFF here, and that is a measured decision rather than a default:
reports/experiments/profile/train_step_comparison.json shows it costing 24% of step time
to save memory an 80GB card does not need.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

OUT = Path("reports/experiments/scaling")


def device_slug(device_name: str) -> str:
    """Filesystem-safe directory name for one GPU model.

    WHY RECORDS ARE FILED BY DEVICE. They used to be <strategy>_ws<N>.json in one flat
    directory, and those files persist across sessions. Re-running world sizes 1 and 2 on
    an L40S pod overwrote two A100 records and left the A100's ws4 in place, producing a
    curve that spanned two GPU models and reported 108.6% scaling efficiency. The same
    collision, seen from the other side, would have destroyed a published measurement by
    overwriting it.

    A guard that detects a mixed summary is worth having, and it is downstream of the real
    problem: the filename carried no device, so two machines were writing to one namespace.
    Filing by device makes the collision impossible rather than detectable.
    """
    return re.sub(r"[^a-z0-9]+", "-", device_name.lower()).strip("-")
TARGET_GLOBAL_BATCH = 64      # configs/training/baseline.yaml: batch_size 32 x grad_accum 2
PER_DEVICE_BATCH = 16         # divides cleanly at world sizes 1, 2 and 4
SEQ_LEN = 512                 # configs/models/forge_base.yaml
BACKBONE = "microsoft/deberta-v3-base"


def measure(steps: int, strategy: str = "fsdp") -> dict:
    import torch
    import torch.distributed as dist

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.distributed import DistConfig, build_run
    from forge.training.scaling import (
        ThroughputMeasurement,
        assert_global_batch_invariant,
        grad_accum_for_world_size,
    )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    accum = grad_accum_for_world_size(TARGET_GLOBAL_BATCH, PER_DEVICE_BATCH, world_size)
    cfg = DistConfig(
        strategy=strategy, world_size=world_size, per_device_batch=PER_DEVICE_BATCH,
        grad_accum=accum, precision="bf16", gradient_checkpointing=False,
    )
    # Reject a broken matrix here rather than after producing a speedup number.
    assert_global_batch_invariant(
        [{"per_device_batch": PER_DEVICE_BATCH, "grad_accum": accum, "world_size": world_size}],
        label=f"world_size={world_size}",
    )

    torch.manual_seed(42)
    model = build_model(ForgeConfig(backbone=BACKBONE, max_length=SEQ_LEN)).cuda()
    # DeepSpeed builds its own optimizer from the config it is handed, so constructing one
    # here and passing it in would create two and silently step the wrong one.
    optimizer = (
        None if strategy == "deepspeed"
        else torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    )
    run = build_run(cfg, model, optimizer=optimizer, lr=2e-5)
    model = run.model
    model.train()

    g = torch.Generator(device="cpu").manual_seed(42 + rank)
    vocab = 128100  # deberta-v3-base
    ids = torch.randint(0, vocab, (PER_DEVICE_BATCH, SEQ_LEN), generator=g).cuda()
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "doc_labels": torch.randint(0, 2, (PER_DEVICE_BATCH,), generator=g).cuda(),
        "token_labels": torch.randint(0, 3, (PER_DEVICE_BATCH, SEQ_LEN), generator=g).cuda(),
    }

    def one_step():
        """One GLOBAL batch. The loop is identical for both strategies on purpose.

        run.micro_step owns the difference between them: who scales the loss by
        grad_accum and who decides when the optimizer actually moves. Writing that
        difference out here is how a benchmark ends up comparing two different
        experiments while reporting a plausible number. See DistributedRun.
        """
        if strategy != "deepspeed":
            optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            if strategy == "deepspeed":
                # The engine handles bf16 itself; an autocast context on top of it casts
                # twice and muddies what the measurement is of.
                out = model(batch["input_ids"], batch["attention_mask"],
                            batch["doc_labels"], batch["token_labels"])
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(batch["input_ids"], batch["attention_mask"],
                                batch["doc_labels"], batch["token_labels"])
            run.micro_step(out["loss"])

    for _ in range(3):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()          # every rank starts the timed region together, or the wall
                            # clock of rank 0 includes other ranks' warmup

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()
    wall = time.perf_counter() - t0

    m = ThroughputMeasurement(
        world_size=world_size, steps=steps, wall_seconds=wall,
        global_batch_size=cfg.global_batch_size, tokens_per_example=SEQ_LEN,
    )
    result = {
        "world_size": world_size,
        "strategy": strategy,
        "per_device_batch": PER_DEVICE_BATCH,
        "grad_accum": accum,
        "global_batch_size": cfg.global_batch_size,
        "seq_len": SEQ_LEN,
        "precision": "bf16",
        "gradient_checkpointing": False,
        "steps": steps,
        "wall_seconds": round(wall, 3),
        "examples_per_second": round(m.examples_per_second(), 3),
        "tokens_per_second": round(m.tokens_per_second(), 1),
        "seconds_per_step": round(m.seconds_per_step(), 4),
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "device_name": torch.cuda.get_device_name(0),
    }
    if rank == 0:
        out_dir = OUT / device_slug(result["device_name"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{strategy}_ws{world_size}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result, indent=1), flush=True)
    dist.destroy_process_group()
    return result


def _resolve_device(explicit: str | None) -> str | None:
    """Pick which device directory to summarise.

    Explicit wins. With exactly one present, use it. With several, REFUSE and list them,
    rather than guessing and producing a curve for whichever sorted first.
    """
    if explicit:
        return explicit
    dirs = sorted(d.name for d in OUT.iterdir() if d.is_dir()) if OUT.exists() else []
    if len(dirs) == 1:
        return dirs[0]
    if not dirs:
        print(f"no device directories under {OUT}/. Run a measurement first.")
        return None
    print("several devices have records here, so --device is required:")
    for d in dirs:
        print(f"  --device {d}")
    return None


def summarise(strategy: str = "fsdp", device: str | None = None) -> int:
    from forge.training.scaling import ScalingError, ThroughputMeasurement, scaling_efficiency

    slug = _resolve_device(device)
    if slug is None:
        return 1
    out_dir = OUT / slug

    runs = {}
    for f in sorted(out_dir.glob(f"{strategy}_ws*.json")):
        d = json.loads(f.read_text())
        runs[d["world_size"]] = d
    if 1 not in runs:
        print(f"no {strategy}_ws1.json under {out_dir}/: the single-GPU baseline is "
              "what every efficiency is measured against")
        return 1

    def tm(d):
        return ThroughputMeasurement(
            world_size=d["world_size"], steps=d["steps"], wall_seconds=d["wall_seconds"],
            global_batch_size=d["global_batch_size"], tokens_per_example=d["seq_len"],
        )

    # THE DEVICE INVARIANT. summarise() globs <strategy>_ws*.json, and those files persist
    # across sessions. Re-running world sizes 1 and 2 on a new pod overwrites those two and
    # leaves ws4 from the PREVIOUS machine in place, so the table silently spans two
    # hardware platforms. That is exactly what happened: an A100 ws4 row landed in an L40S
    # curve and reported 108.6% scaling efficiency.
    #
    # Superlinear efficiency is the tell. It cannot happen, so a number above 100% always
    # means the comparison is broken rather than the hardware being remarkable. Refusing
    # here is better than printing it, because a reader who does not know that rule reads
    # 108.6% as a good result.
    devices = {runs[p].get("device_name", "unknown") for p in runs}
    if len(devices) != 1:
        by_device: dict[str, list[int]] = {}
        for p in sorted(runs):
            by_device.setdefault(runs[p].get("device_name", "unknown"), []).append(p)
        print("REFUSING TO SUMMARISE: these runs are from different hardware, so the "
              "speedup between them is not a speedup.")
        for dev, ws in by_device.items():
            print(f"  {dev}: world sizes {ws}")
        print(f"Delete the stale records in {out_dir}/ or re-run every world size on "
              "one machine.")
        return 1

    base = tm(runs[1])
    rows = []
    for ws in sorted(runs):
        d = runs[ws]
        try:
            eff = 1.0 if ws == 1 else scaling_efficiency(base, tm(d))
        except ScalingError as e:
            print(f"ws{ws}: {e}")
            continue
        if eff > 1.0 + 1e-9:
            print(f"REFUSING TO SUMMARISE: world size {ws} reports {eff:.1%} scaling "
                  "efficiency. Above 100% is superlinear, which does not happen; it means "
                  "the runs are not comparable. Check that every record has the same "
                  "device_name, global_batch_size and gradient_checkpointing.")
            return 1
        rows.append({
            "world_size": ws,
            "examples_per_second": d["examples_per_second"],
            "tokens_per_second": d["tokens_per_second"],
            "seconds_per_step": d["seconds_per_step"],
            "peak_memory_gib": d["peak_memory_gib"],
            "speedup": round(d["examples_per_second"] / runs[1]["examples_per_second"], 3),
            "scaling_efficiency": round(eff, 3),
        })

    out = {
        "strategy": strategy,
        "device_name": runs[1]["device_name"],
        "global_batch_size": runs[1]["global_batch_size"],
        "seq_len": runs[1]["seq_len"],
        "precision": runs[1]["precision"],
        "gradient_checkpointing": runs[1]["gradient_checkpointing"],
        "invariant": (
            "Every world size runs the same global batch; grad_accum absorbs the "
            "difference. Efficiency is speedup divided by world size, so 1.0 is linear "
            "and never happens."
        ),
        "rows": rows,
    }
    path = out_dir / ("scaling_summary.json" if strategy == "fsdp"
                      else f"scaling_summary_{strategy}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'ws':>3} {'ex/s':>9} {'tok/s':>11} {'s/step':>8} {'peak GiB':>9} "
          f"{'speedup':>8} {'efficiency':>11}")
    for r in rows:
        print(f"{r['world_size']:>3} {r['examples_per_second']:>9.2f} "
              f"{r['tokens_per_second']:>11.0f} {r['seconds_per_step']:>8.3f} "
              f"{r['peak_memory_gib']:>9.2f} {r['speedup']:>8.2f} "
              f"{r['scaling_efficiency']:>11.1%}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--strategy", default="fsdp", choices=("fsdp", "deepspeed"))
    ap.add_argument("--device", default=None,
                    help="device directory to summarise, e.g. nvidia-l40s. Required only "
                         "when records from several GPU models are present.")
    a = ap.parse_args()
    raise SystemExit(
        summarise(a.strategy, a.device) if a.summarise
        else (measure(a.steps, a.strategy) and 0)
    )
