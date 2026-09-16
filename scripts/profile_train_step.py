"""Profile one training step at the shapes configs/training/baseline.yaml actually uses.

WHAT THIS DOES AND DOES NOT MEASURE. It profiles the model step: forward, loss, backward,
optimizer. It does NOT profile data loading or tokenization, because it feeds synthetic
batches rather than the corpus. That is deliberate and it is the honest scope: the question
this run has to answer is which OPERATORS dominate a step, because kernels/cuda/README.md
refuses a fused kernel until a profile ranks the stage it fuses. A data pipeline profile is
a separate measurement and needs the corpus present.

It also answers a second question the config raises. baseline.yaml sets
gradient_checkpointing: true, which trades compute for memory by recomputing activations
during backward. That is the right default on a small card. On an 80GB A100 it may be
paying for memory nobody needs. So both settings are timed and both peak allocations are
recorded, and the answer is a number rather than an opinion.

    python scripts/profile_train_step.py
    python scripts/profile_train_step.py --batch-size 16 --steps 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

OUT = Path("reports/experiments/profile")
# From configs/training/baseline.yaml and configs/models/forge_base.yaml. Not invented.
DEFAULT_BATCH, DEFAULT_SEQ = 32, 512
DEFAULT_BACKBONE = "microsoft/deberta-v3-base"


def make_batch(batch_size: int, seq_len: int, vocab: int, device):
    import torch

    g = torch.Generator(device="cpu").manual_seed(42)
    ids = torch.randint(0, vocab, (batch_size, seq_len), generator=g)
    token_labels = torch.randint(0, 3, (batch_size, seq_len), generator=g)
    # Roughly a tenth ignored, so the ignore_index path in the token loss is exercised
    # rather than silently skipped by a batch that happens to have no padding.
    token_labels[torch.rand(token_labels.shape, generator=g) < 0.1] = -100
    return {
        "input_ids": ids.to(device),
        "attention_mask": torch.ones_like(ids).to(device),
        "doc_labels": torch.randint(0, 2, (batch_size,), generator=g).to(device),
        "token_labels": token_labels.to(device),
    }


def build(backbone: str, seq_len: int, checkpointing: bool, device):
    import torch

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.train import enable_gradient_checkpointing

    torch.manual_seed(42)
    model = build_model(ForgeConfig(backbone=backbone, max_length=seq_len)).to(device)
    enabled = enable_gradient_checkpointing(model.encoder) if checkpointing else False
    model.train()
    return model, enabled


def step_fn(model, batch, optimizer, autocast_dtype):
    import torch

    def run():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            out = model(batch["input_ids"], batch["attention_mask"],
                        batch["doc_labels"], batch["token_labels"])
        out["loss"].backward()
        optimizer.step()

    return run


def timed(run, steps: int, device) -> dict:
    """Median step time over `steps`, after a warmup, synchronising each step.

    Median rather than mean: one slow step from an allocator growth or a clock change
    drags a mean and says nothing about the steady state.
    """
    import torch

    for _ in range(3):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    per_step = []
    for _ in range(steps):
        t0 = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_step.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median_step_ms": round(statistics.median(per_step), 2),
        "p90_step_ms": round(sorted(per_step)[int(0.9 * (len(per_step) - 1))], 2),
        "steps_timed": steps,
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if device.type == "cuda" else None
        ),
    }


def main() -> int:
    import torch

    from forge.training.profiling import profile_step

    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ)
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    OUT.mkdir(parents=True, exist_ok=True)

    arms = {}
    for checkpointing in (True, False):
        name = "grad_ckpt_on" if checkpointing else "grad_ckpt_off"
        print(f"\n=== {name} ===", flush=True)

        model, enabled = build(args.backbone, args.seq_len, checkpointing, device)
        if checkpointing and not enabled:
            print("  gradient checkpointing was requested but not enabled; recording that")
        vocab = model.encoder.config.vocab_size
        batch = make_batch(args.batch_size, args.seq_len, vocab, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        run = step_fn(model, batch, optimizer, dtype)

        timing = timed(run, args.steps, device)
        print(f"  median {timing['median_step_ms']} ms | p90 {timing['p90_step_ms']} ms "
              f"| peak {timing['peak_memory_gib']} GiB", flush=True)

        summary = profile_step(run, out_dir=OUT, label=name)
        print(f"  top op: {summary['top'][0]['op']} at {summary['top'][0]['share']:.1%}")

        arms[name] = {"checkpointing_enabled": enabled, **timing,
                      "top": summary["top"][:10], "device_name": summary["device_name"]}
        del model, optimizer, batch, run
        if device.type == "cuda":
            torch.cuda.empty_cache()

    on, off = arms["grad_ckpt_on"], arms["grad_ckpt_off"]
    comparison = {
        "config_source": "configs/training/baseline.yaml + configs/models/forge_base.yaml",
        "backbone": args.backbone,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "precision": "bf16" if dtype is not None else "fp32",
        "device_name": on["device_name"],
        "scope": (
            "Model step only: forward, loss, backward, optimizer. Synthetic batches, so "
            "data loading and tokenization are NOT included."
        ),
        "arms": arms,
        "gradient_checkpointing_cost": {
            "step_time_overhead": round(on["median_step_ms"] / off["median_step_ms"] - 1, 4),
            "memory_saved_gib": (
                round(off["peak_memory_gib"] - on["peak_memory_gib"], 2)
                if on["peak_memory_gib"] is not None else None
            ),
        },
    }
    path = OUT / "train_step_comparison.json"
    path.write_text(json.dumps(comparison, indent=1) + "\n")

    c = comparison["gradient_checkpointing_cost"]
    print(f"\ngradient checkpointing costs {c['step_time_overhead']:+.1%} step time "
          f"and saves {c['memory_saved_gib']} GiB")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
