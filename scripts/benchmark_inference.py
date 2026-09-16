"""Latency and throughput of the detector's serving path, measured rather than assumed.

    python scripts/benchmark_inference.py                 # CPU, which is what deploys
    python scripts/benchmark_inference.py --device cuda   # the GPU arm, on a pod

WHY THIS EXISTS. The training side of this project has a profile that named a bottleneck
and changed a config. The serving side had nothing: no latency number, no throughput
number, and a batch size of 32 hardcoded in scorer.py that nobody had ever measured. The
release gate refers to a "P95 latency budget" and forge.inference.batching tunes its wait
window "against the P95 latency budget in the release gate". No P95 had ever been measured,
so that budget was a number in prose with no artifact behind it.

SCOPE, stated because the number is meaningless without it. This times the MODEL FORWARD
over already-tokenised windows: the part that scales with batch size and the part a serving
decision can act on. Tokenisation, windowing and HTTP overhead are NOT included. A
request's true latency is this plus those, and this file does not pretend otherwise.

WHY CPU IS THE HEADLINE. FORGE serves on CPU in float32. That is not a limitation being
apologised for: the detector is a 184M-parameter encoder scoring 512-token windows, and on
CPU it is fast enough for the deployment it has. The GPU arm is measured for comparison, so
the choice is defensible with numbers instead of a claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

OUT = Path("reports/experiments/inference")
SEQ_LEN = 512               # configs/models/forge_base.yaml, and the serving window
BACKBONE = "microsoft/deberta-v3-base"
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
MIN_SAMPLES = 3             # a median and a nearest-rank p95 need at least this
BUDGET_S = 20.0             # per point, so the whole sweep stays bounded on CPU
# Document lengths in WINDOWS. scorer.py windows with a stride, so a long document costs
# several forwards; 1 window is a short comment, 16 is a long article.
DOC_WINDOWS = (1, 2, 4, 8, 16)


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation: with 10 samples, interpolating invents
    precision the sample size does not support."""
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * len(ordered) + 0.5) - 1))
    return ordered[idx]


def measure(device: str, repeats: int) -> dict:
    import torch

    from forge.modeling.encoder import ForgeConfig, build_model

    torch.manual_seed(42)
    model = build_model(ForgeConfig(backbone=BACKBONE, max_length=SEQ_LEN))
    model.eval()
    if device == "cuda":
        model = model.cuda()

    def forward(n_windows: int) -> float:
        """Seconds for one forward over n_windows. Synchronised on CUDA, or the timer
        measures kernel launch rather than execution."""
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, 128100, (n_windows, SEQ_LEN), generator=g)
        mask = torch.ones_like(ids)
        if device == "cuda":
            ids, mask = ids.cuda(), mask.cuda()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(ids, mask)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def timed(n_windows: int) -> dict:
        """Sample until either `repeats` samples or BUDGET_S of wall clock, whichever
        comes first, with a floor of MIN_SAMPLES.

        A fixed repeat count is wrong here because cost grows with batch size: seven
        samples at batch 1 is two seconds and seven at batch 32 is several minutes on CPU.
        The budget keeps the whole sweep bounded, and the sample count is recorded per row
        so nobody reads a p95 from three samples as if it came from thirty.
        """
        warm = forward(n_windows)               # warmup: first call pays lazy init
        samples: list[float] = []
        spent = 0.0
        while len(samples) < repeats and (len(samples) < MIN_SAMPLES or spent < BUDGET_S):
            dt = forward(n_windows)
            samples.append(dt * 1000)
            spent += dt
        median = statistics.median(samples)
        row = {
            "median_ms": round(median, 2),
            "p95_ms": round(_percentile(samples, 0.95), 2),
            "windows_per_second": round(n_windows / (median / 1000), 2),
            "ms_per_window": round(median / n_windows, 2),
            "samples": len(samples),
        }
        # Streamed, not buffered to the end. A silent five-minute run is indistinguishable
        # from a hang, and the first version of this script was exactly that.
        print(f"{n_windows:>6} {row['median_ms']:>10.1f} {row['p95_ms']:>9.1f} "
              f"{row['windows_per_second']:>9.2f} {row['ms_per_window']:>10.1f} "
              f"{row['samples']:>8}", flush=True)
        del warm
        return row

    print(f"{'batch':>6} {'median ms':>10} {'p95 ms':>9} {'win/s':>9} {'ms/window':>10} "
          f"{'samples':>8}", flush=True)
    batch_rows = [{"batch_windows": b, **timed(b)} for b in BATCH_SIZES]

    # A document of N windows is scored in one forward, so its latency IS the batch row for
    # N. The first version of this script re-measured them, doubling the runtime to
    # reproduce numbers it already had. Derived, not re-run.
    by_windows = {r["batch_windows"]: r for r in batch_rows}
    doc_rows = [
        {"document_windows": n, **{k: v for k, v in by_windows[n].items()
                                   if k != "batch_windows"}}
        for n in DOC_WINDOWS if n in by_windows
    ]

    best = max(batch_rows, key=lambda r: r["windows_per_second"])
    single = next(r for r in batch_rows if r["batch_windows"] == 1)
    result = {
        "device": device,
        "device_name": (
            torch.cuda.get_device_name(0) if device == "cuda"
            else f"{platform.processor() or platform.machine()} ({platform.system()})"
        ),
        "backbone": BACKBONE,
        "seq_len": SEQ_LEN,
        "precision": "float32",
        "torch_threads": torch.get_num_threads(),
        "scope": (
            "Model forward over pre-tokenised windows only. Tokenisation, windowing and "
            "HTTP overhead are NOT included, so a real request costs this plus those."
        ),
        "batch_sweep": batch_rows,
        "document_latency": doc_rows,
        "peak_throughput": {
            "batch_windows": best["batch_windows"],
            "windows_per_second": best["windows_per_second"],
            "speedup_over_batch_1": round(
                best["windows_per_second"] / single["windows_per_second"], 2
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{device}_latency.json"
    path.write_text(json.dumps(result, indent=1) + "\n")

    print(f"\ndevice: {result['device_name']}  threads: {result['torch_threads']}")
    print(f"peak throughput at batch {best['batch_windows']}: "
          f"{best['windows_per_second']:.2f} windows/s, "
          f"{result['peak_throughput']['speedup_over_batch_1']}x batch 1")
    print(f"wrote {path}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--repeats", type=int, default=7,
                    help="samples per point. 7 is enough for a median and a nearest-rank "
                         "p95 without pretending to more precision than that supports.")
    a = ap.parse_args()
    measure(a.device, a.repeats)
