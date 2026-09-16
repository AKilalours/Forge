"""A scaling table may only be built from runs that are actually comparable.

WHY THIS FILE EXISTS. scripts/scaling_run.py writes one record per world size, named
<strategy>_ws<N>.json, and summarise() globs them. Those files persist across sessions.

On a second pod, world sizes 1 and 2 were re-run and overwrote their records. World size 4
was not re-run, so the record from the PREVIOUS machine stayed. The summary then combined
two L40S measurements with one A100 measurement and reported:

      4    254.54    130325    0.251    10.98    4.34    108.6%

108.6% scaling efficiency. Superlinear speedup does not happen: four GPUs cannot do more
than four times the work of one. The number is impossible, which is the only reason the
mistake was visible at all. Had the stale record come from slightly SLOWER hardware, the
table would have shown a plausible efficiency and nobody would have questioned it.

So there are two guards, and the second one matters more than the first. Checking
device_name catches the known cause. Refusing any efficiency above 100% catches the whole
class, including causes nobody has thought of yet: a changed global batch, a different
gradient_checkpointing setting, a record from a different precision. The invariant is
about the arithmetic being possible, not about the one bug that was found.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "scaling_run.py"


def _record(ws: int, ex_per_s: float, device: str = "NVIDIA L40S") -> dict:
    """A run record with the fields summarise() reads. wall_seconds is derived so that
    examples_per_second comes out as asked, because summarise recomputes it."""
    steps, global_batch = 12, 64
    return {
        "world_size": ws, "strategy": "fsdp", "per_device_batch": 16,
        "grad_accum": 64 // (16 * ws), "global_batch_size": global_batch, "seq_len": 512,
        "precision": "bf16", "gradient_checkpointing": False, "steps": steps,
        "wall_seconds": round(steps * global_batch / ex_per_s, 4),
        "examples_per_second": ex_per_s,
        "tokens_per_second": ex_per_s * 512,
        "seconds_per_step": round(global_batch / ex_per_s, 4),
        "peak_memory_gib": 12.0, "device_name": device,
    }


def _run_summarise(tmp_path: Path, records: list[dict],
                   device_dir: str = "nvidia-l40s") -> subprocess.CompletedProcess:
    """Records now live under reports/experiments/scaling/<device>/, because the flat
    layout let two machines write to one namespace. Tests that force records from
    different devices into ONE directory still exercise the device guard, which is the
    point: the directory layout makes the collision unlikely, the guard makes it loud."""
    out = tmp_path / "reports" / "experiments" / "scaling" / device_dir
    out.mkdir(parents=True)
    for rec in records:
        (out / f"fsdp_ws{rec['world_size']}.json").write_text(json.dumps(rec))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--summarise", "--strategy", "fsdp",
         "--device", device_dir],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )


def test_a_clean_single_device_curve_summarises(tmp_path: Path) -> None:
    proc = _run_summarise(tmp_path, [
        _record(1, 58.63), _record(2, 106.31),
    ])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "90" in proc.stdout, "expected an efficiency near 90% in the table"


def test_records_from_two_different_devices_are_refused(tmp_path: Path) -> None:
    """THE REGRESSION, exactly as it happened: L40S ws1 and ws2, A100 ws4 left behind."""
    proc = _run_summarise(tmp_path, [
        _record(1, 58.63, "NVIDIA L40S"),
        _record(2, 106.31, "NVIDIA L40S"),
        _record(4, 254.54, "NVIDIA A100-SXM4-80GB"),
    ])
    assert proc.returncode == 1
    assert "different hardware" in proc.stdout
    assert "NVIDIA L40S" in proc.stdout and "A100" in proc.stdout, (
        "the message must name which records came from where, or the reader has to guess"
    )


def test_superlinear_efficiency_is_refused_even_on_one_device(tmp_path: Path) -> None:
    """The guard that catches causes nobody has thought of.

    Same device everywhere, so the device check passes, but two GPUs report more than
    twice the throughput of one. That is impossible, so something else differs between the
    records. Refusing on the arithmetic catches the whole class rather than one cause.
    """
    proc = _run_summarise(tmp_path, [
        _record(1, 58.63), _record(2, 130.0),
    ])
    assert proc.returncode == 1
    assert "superlinear" in proc.stdout.lower()


def test_the_missing_baseline_is_still_refused(tmp_path: Path) -> None:
    """Every efficiency is measured against world size 1; without it there is no table."""
    proc = _run_summarise(tmp_path, [_record(2, 106.31), _record(4, 200.0)])
    assert proc.returncode == 1
    assert "ws1" in proc.stdout


@pytest.mark.parametrize("ws", [1, 2])
def test_a_single_world_size_is_not_a_scaling_curve(tmp_path: Path, ws: int) -> None:
    """One point is a measurement, not a curve. ws1 alone should succeed trivially at
    100%; ws2 alone has no baseline and must fail."""
    proc = _run_summarise(tmp_path, [_record(ws, 58.63)])
    assert proc.returncode == (0 if ws == 1 else 1)


def test_two_device_directories_require_an_explicit_choice(tmp_path: Path) -> None:
    """With records from several machines present, summarise must ASK rather than pick.

    Guessing would produce a valid-looking curve for whichever directory sorted first,
    which is the quiet version of the bug this layout exists to prevent.
    """
    root = tmp_path / "reports" / "experiments" / "scaling"
    for dev, ex in (("nvidia-l40s", 58.63), ("nvidia-a100-sxm4-80gb", 65.03)):
        d = root / dev
        d.mkdir(parents=True)
        (d / "fsdp_ws1.json").write_text(json.dumps(_record(1, ex, dev)))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--summarise", "--strategy", "fsdp"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 1
    assert "--device" in proc.stdout
    assert "nvidia-l40s" in proc.stdout and "nvidia-a100-sxm4-80gb" in proc.stdout


def test_one_device_directory_needs_no_flag(tmp_path: Path) -> None:
    """The common case stays a single command."""
    root = tmp_path / "reports" / "experiments" / "scaling" / "nvidia-l40s"
    root.mkdir(parents=True)
    for ws, ex in ((1, 58.63), (2, 106.31)):
        (root / f"fsdp_ws{ws}.json").write_text(json.dumps(_record(ws, ex)))
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--summarise", "--strategy", "fsdp"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_summary_lands_inside_its_device_directory(tmp_path: Path) -> None:
    """Two devices must not fight over one scaling_summary.json either."""
    _run_summarise(tmp_path, [_record(1, 58.63), _record(2, 106.31)])
    summary = (tmp_path / "reports" / "experiments" / "scaling" / "nvidia-l40s"
               / "scaling_summary.json")
    assert summary.exists(), "the summary must be written beside the records it came from"
