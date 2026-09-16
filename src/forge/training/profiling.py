"""Phase 7. Profiling.

Rule for this file: only measured numbers get reported. PyTorch Profiler and Nsight
Systems produce traces; the traces go in reports/experiments/ next to the before and
after configuration. "GPU utilization improved" without a trace is not a result.

What gets measured: data loading, tokenization, forward, backward, attention,
collective communication, checkpoint write.

WHY THE SUMMARY IS COMMITTED AND THE TRACE IS NOT. A chrome trace is tens of megabytes
and unreadable without a viewer, so it stays under reports/ but out of git. What gets
committed is the ranked table: operator, device time, share of the step. That is the
artifact a CUDA decision has to cite, because kernels/cuda/README.md refuses to accept a
fused kernel until a profile ranks the stage it fuses. Without a committed ranking, the
kernel rests on a trace that only its author ever saw.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Wait before recording, then warm up, then record. The first steps of any training loop
# are allocator growth, autotuning and lazy CUDA context creation, and profiling them
# reports a startup cost as if it were a per-step cost.
WAIT, WARMUP, ACTIVE = 1, 2, 3


def profile_step(
    fn: Callable[..., Any],
    *args: Any,
    steps: int | None = None,
    out_dir: str | Path = "reports/experiments/profile",
    label: str = "train_step",
    **kwargs: Any,
) -> dict:
    """Run `fn(*args, **kwargs)` under the PyTorch profiler and rank what it spent time on.

    Returns the summary dict and writes two files:
      <out_dir>/<label>_trace.json   the chrome trace, for a viewer
      <out_dir>/<label>_summary.json the ranked table, small enough to commit

    Works on CPU as well as CUDA, so the shape of the output can be tested without a GPU.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cuda = torch.cuda.is_available()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    total = steps if steps is not None else WAIT + WARMUP + ACTIVE

    with profile(
        activities=activities,
        schedule=schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # stacks make the trace several times larger for little gain here
    ) as prof:
        for _ in range(total):
            fn(*args, **kwargs)
            if cuda:
                # Without this the profiler attributes asynchronous kernel time to
                # whichever python call happened to be running when it landed.
                torch.cuda.synchronize()
            prof.step()

    prof.export_chrome_trace(str(out / f"{label}_trace.json"))

    sort_key = "self_cuda_time_total" if cuda else "self_cpu_time_total"
    rows = []
    for e in prof.key_averages():
        device_us = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        cpu_us = float(e.self_cpu_time_total)
        rows.append(
            {
                "op": e.key,
                "calls": int(e.count),
                "self_device_us": round(device_us, 1),
                "self_cpu_us": round(cpu_us, 1),
            }
        )
    rows.sort(key=lambda r: r["self_device_us"] if cuda else r["self_cpu_us"], reverse=True)

    measured = sum(r["self_device_us"] if cuda else r["self_cpu_us"] for r in rows) or 1.0
    for r in rows:
        r["share"] = round((r["self_device_us"] if cuda else r["self_cpu_us"]) / measured, 4)

    summary = {
        "label": label,
        "device": "cuda" if cuda else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
        "sorted_by": sort_key,
        "active_steps": ACTIVE,
        "total_self_time_us": round(measured, 1),
        # Twenty is enough to see the shape and to argue about the top of it. The full
        # ranking is in the trace for anyone who wants it.
        "top": rows[:20],
        "note": (
            "Shares are of measured self time across the profiled steps, not of wall "
            "clock. A stage that is 3 percent here does not become worth a fused kernel "
            "because the number looks small in isolation: see kernels/cuda/README.md, "
            "which requires this ranking before any kernel is written."
        ),
    }
    (out / f"{label}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
