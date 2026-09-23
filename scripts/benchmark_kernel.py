"""Time the fused row gather against ATen, after proving it computes the same thing.

    python scripts/benchmark_kernel.py                     # bf16 and fp32, four sizes
    python scripts/benchmark_kernel.py --rows 4096 --repeats 50

TWO REFUSALS, BOTH LEARNED FROM THIS PROJECT'S OWN MISTAKES.

It refuses to run when the extension did not build. Without that check the fused path falls
back to `table[index]` and the benchmark compares ATen against ATen, producing a speedup of
1.0 with no indication that the kernel was never involved. The Spark sweep in this repo
published a number that varied a setting the runner never read; this is the same failure
wearing a different hat.

It refuses to publish a timing if the gradients disagree. A kernel that is faster and
slightly wrong is worse than a slow one: training still converges, the loss curve still
looks fine, and every published figure then describes a model trained with a wrong gradient
on the relative-position table.

WHAT IS TIMED. CUDA events around forward, backward, and the pair, with a warmup and a
synchronise before each measurement, because a wall-clock timer around an async launch
measures the launch and not the work. The median of `repeats`, not the mean, since the first
few iterations carry allocator and autotune noise.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "experiments" / "kernel"
TABLE_ROWS, DIM = 512, 768
DTYPES = {"bfloat16": "bfloat16", "float32": "float32"}


def _time(fn, repeats: int, warmup: int = 10) -> float:
    """Median milliseconds for `fn`, measured with CUDA events."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _case(n_rows: int, dtype, repeats: int) -> dict:
    import torch

    from forge.kernels.fused_gather import fused_gather, reference_gather

    generator = torch.Generator(device="cuda").manual_seed(0)
    base = torch.randn(TABLE_ROWS, DIM, device="cuda", dtype=dtype, generator=generator)
    # Heavy duplication, as the real op has: the relative-position buckets are shared
    # across the window, which is what makes the backward an accumulation rather than a
    # scatter of distinct rows.
    index = torch.randint(0, TABLE_ROWS // 8, (n_rows,), device="cuda", generator=generator)
    upstream = torch.randn(n_rows, DIM, device="cuda", dtype=dtype)

    fused_t = base.clone().requires_grad_(True)
    aten_t = base.clone().requires_grad_(True)

    # Correctness first, on the exact tensors about to be timed.
    fused_gather(fused_t, index).backward(upstream)
    reference_gather(aten_t, index).backward(upstream)
    tolerance = 3e-3 if dtype is torch.bfloat16 else 1e-5
    delta = (fused_t.grad - aten_t.grad).abs().max().item()
    if not torch.allclose(fused_t.grad, aten_t.grad, rtol=tolerance, atol=tolerance):
        raise SystemExit(
            f"refusing to time a kernel whose gradient disagrees with ATen by {delta:.3e} "
            f"at rows={n_rows} dtype={dtype}. Fix correctness before measuring speed."
        )

    def make(fn, tensor):
        def forward_only():
            fn(tensor, index)

        def forward_backward():
            tensor.grad = None
            fn(tensor, index).backward(upstream)

        return forward_only, forward_backward

    fused_fwd, fused_both = make(fused_gather, fused_t)
    aten_fwd, aten_both = make(reference_gather, aten_t)

    result = {
        "rows": n_rows,
        "dtype": str(dtype).replace("torch.", ""),
        "max_abs_grad_delta": delta,
        "aten_forward_ms": round(_time(aten_fwd, repeats), 4),
        "fused_forward_ms": round(_time(fused_fwd, repeats), 4),
        "aten_forward_backward_ms": round(_time(aten_both, repeats), 4),
        "fused_forward_backward_ms": round(_time(fused_both, repeats), 4),
    }
    result["forward_speedup"] = round(
        result["aten_forward_ms"] / result["fused_forward_ms"], 3)
    result["forward_backward_speedup"] = round(
        result["aten_forward_backward_ms"] / result["fused_forward_backward_ms"], 3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096, 16384, 65536])
    parser.add_argument("--dtypes", nargs="+", default=list(DTYPES), choices=list(DTYPES))
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args(argv)

    import torch

    from forge.kernels.fused_gather import available, backend, unavailable_reason

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device; this benchmark has nothing to measure")
    if not available():
        raise SystemExit(
            f"the extension did not build: {unavailable_reason()}. Refusing to run, "
            f"because the fallback would compare ATen against ATen and report 1.0x as "
            f"though the kernel had been measured."
        )

    report = {
        "backend": backend(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "table_rows": TABLE_ROWS,
        "dim": DIM,
        "repeats": args.repeats,
        "scope": "One gather of whole rows from the relative-position table, and its "
                 "scatter-add backward. NOT a full training step: the step-level claim is "
                 "this speedup applied to the 20.87% of device time the profile attributes "
                 "to aten::scatter_add_, and that share is in "
                 "reports/experiments/profile/train_step_comparison.json.",
        "cases": [],
    }
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for rows in args.rows:
            case = _case(rows, dtype, args.repeats)
            report["cases"].append(case)
            print(f"{dtype_name:9s} rows={rows:6d}  forward {case['forward_speedup']}x  "
                  f"forward+backward {case['forward_backward_speedup']}x", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fused_gather.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {path}")

    best = max(c["forward_backward_speedup"] for c in report["cases"])
    if best <= 1.0:
        print("\nThe kernel is not faster than ATen on any shape measured. That is a "
              "result, and it belongs in the repo as one: ATen's scatter_add_ is already "
              "well optimised, and a specialised kernel that does not beat it says the "
              "generality was not costing what the profile implied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
