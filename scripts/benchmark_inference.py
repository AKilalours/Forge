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
ROOT_DIR = Path(__file__).resolve().parents[1]
SERVING_BATCH = 8           # forge.inference.scorer.SERVING_BATCH_WINDOWS, the measured peak
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


def measure(device: str, repeats: int, threads: int | None = None,
            only_batch: int | None = None) -> dict:
    import torch

    from forge.modeling.encoder import ForgeConfig, build_model

    # THREADS ARE THE LEVER ON THIS PATH, not batch size. The first version of this script
    # left torch at its default and recorded torch_threads=4 on a 10-core machine. The
    # Spark sweep later confirmed what that caveat suspected: giving the same work all ten
    # cores raised throughput 51%. Every latency figure measured at the default was
    # understated, so the thread count is now chosen explicitly and recorded, rather than
    # inherited from whatever torch decided.
    if threads is not None:
        torch.set_num_threads(max(1, threads))

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
    # only_batch exists for the thread sweep. Scanning every batch size at every thread
    # count is ~40 minutes of mostly redundant work: the batch sweep already established
    # that throughput is flat in batch size, so re-establishing it at five thread counts
    # measures the same flat line five times. The thread question needs one batch size.
    sizes = (only_batch,) if only_batch else BATCH_SIZES
    batch_rows = [{"batch_windows": b, **timed(b)} for b in sizes]

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
    # batch 1 is only present in a full sweep. --only-batch measures one point, so the
    # speedup-over-batch-1 figure has no denominator. None is correct; reaching for a row
    # that is not there raised StopIteration, and substituting "the smallest row we happen
    # to have" would quietly redefine the ratio.
    single = next((r for r in batch_rows if r["batch_windows"] == 1), None)
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
            "speedup_over_batch_1": (
                round(best["windows_per_second"] / single["windows_per_second"], 2)
                if single else None
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"{device}_latency.json" if not only_batch
                  else f"{device}_latency_b{only_batch}_t{torch.get_num_threads()}.json")
    path.write_text(json.dumps(result, indent=1) + "\n")

    print(f"\ndevice: {result['device_name']}  threads: {result['torch_threads']}")
    line = (f"peak throughput at batch {best['batch_windows']}: "
            f"{best['windows_per_second']:.2f} windows/s")
    if single:
        line += f", {result['peak_throughput']['speedup_over_batch_1']}x batch 1"
    print(line)
    print(f"wrote {path}")
    return result


def thread_sweep(device: str, repeats: int) -> dict:
    """Throughput against intra-op thread count, at the batch size the server uses.

    The batch sweep answered "does batching help" with no. This answers "what does help",
    and it is the question the earlier caveat left open rather than closed. Run in one
    process per thread count, because torch.set_num_threads inside a live process does not
    reliably resize a pool that has already been used.
    """
    import os
    import subprocess
    import sys

    counts = sorted({1, 2, 4, 8, os.cpu_count() or 1})
    counts = [c for c in counts if c <= (os.cpu_count() or 1)]
    rows = []
    print(f"{'threads':>8} {'median ms':>10} {'win/s':>9} {'speedup':>9}", flush=True)
    for n in counts:
        proc = subprocess.run(
            [sys.executable, __file__, "--device", device, "--threads", str(n),
             "--repeats", str(repeats), "--only-batch", str(SERVING_BATCH)],
            capture_output=True, text=True, cwd=ROOT_DIR,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"thread={n} run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        data = json.loads((OUT / f"{device}_latency_b{SERVING_BATCH}_t{n}.json").read_text())
        peak = data["peak_throughput"]
        rows.append({
            "threads": data["torch_threads"],
            "requested_threads": n,
            "peak_batch": peak["batch_windows"],
            "windows_per_second": peak["windows_per_second"],
        })
        base = rows[0]["windows_per_second"]
        print(f"{n:>8} {'':>10} {rows[-1]['windows_per_second']:>9.2f} "
              f"{rows[-1]['windows_per_second'] / base:>9.2f}", flush=True)

    base = rows[0]["windows_per_second"]
    for r in rows:
        r["speedup_over_one_thread"] = round(r["windows_per_second"] / base, 3)
        r["efficiency"] = round(r["windows_per_second"] / base / r["requested_threads"], 3)

    out = {
        "device": device,
        "host_cpu_count": os.cpu_count(),
        "fixed": "peak throughput across the batch sweep, at each thread count",
        "note": (
            "One process per thread count. torch.set_num_threads does not reliably resize "
            "a thread pool that has already run work, so measuring several counts inside "
            "one process silently reports the first one several times."
        ),
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{device}_threads.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads. Default leaves torch's own choice, which on a "
                         "10-core Mac is 4 and understates every figure by about a third.")
    ap.add_argument("--only-batch", type=int, default=None,
                    help="measure a single batch size. Used by --thread-sweep so it does "
                         "not re-measure a flat curve once per thread count.")
    ap.add_argument("--thread-sweep", action="store_true",
                    help="scan thread count at a fixed batch size. On CPU this is the "
                         "dimension that actually moves throughput.")
    ap.add_argument("--repeats", type=int, default=7,
                    help="samples per point. 7 is enough for a median and a nearest-rank "
                         "p95 without pretending to more precision than that supports.")
    a = ap.parse_args()
    if a.thread_sweep:
        thread_sweep(a.device, a.repeats)
    else:
        measure(a.device, a.repeats, a.threads, a.only_batch)
