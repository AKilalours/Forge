"""The serving batch size must be the one the measurement supports.

WHY THIS FILE EXISTS. scorer.py hardcoded `batch_size=32` in its DataLoader and
BatchPolicy defaulted `max_batch=32`. Neither number had ever been measured. Both were the
value that looks right for a GPU, on a serving path that runs on CPU in float32.

When the sweep was finally run (scripts/benchmark_inference.py, committed at
reports/experiments/inference/cpu_latency.json), batch 32 turned out to be the WORST point
in it: 2.03 windows/s against 2.50 at batch 8, while making one request wait 15.8 seconds
instead of 3.2. The code had been paying five times the latency for negative throughput.

The wider result matters more than the constant. Throughput on CPU is flat within about
13% from batch 1 to batch 16, so batching is not the lever on this path at all. It is a
GPU optimisation that was carried over to hardware where it does not apply. The comment in
batching.py about tuning the wait window "against the P95 latency budget in the release
gate" referred to a P95 that had never been measured.

This test stops the constant and the measurement drifting apart again. It reads the
committed artifact, finds the batch size that actually maximised throughput, and asserts
the code uses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "reports" / "experiments" / "inference" / "cpu_latency.json"


@pytest.fixture(scope="module")
def measurement() -> dict:
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT.relative_to(ROOT)} not committed; run "
                    "scripts/benchmark_inference.py")
    return json.loads(ARTIFACT.read_text())


def test_the_serving_batch_size_is_the_measured_optimum(measurement: dict) -> None:
    from forge.inference.scorer import SERVING_BATCH_WINDOWS

    best = max(measurement["batch_sweep"], key=lambda r: r["windows_per_second"])
    assert SERVING_BATCH_WINDOWS == best["batch_windows"], (
        f"scorer.py serves at batch {SERVING_BATCH_WINDOWS}, but the committed sweep peaks "
        f"at {best['batch_windows']} ({best['windows_per_second']} windows/s). Either "
        "re-run scripts/benchmark_inference.py and commit the new artifact, or change the "
        "constant. They must not disagree."
    )


def test_the_dynamic_batcher_does_not_exceed_the_measured_optimum() -> None:
    """max_batch past the peak adds latency to every request and buys no throughput."""
    from forge.inference.batching import BatchPolicy
    from forge.inference.scorer import SERVING_BATCH_WINDOWS

    assert BatchPolicy().max_batch <= SERVING_BATCH_WINDOWS, (
        f"BatchPolicy waits for up to {BatchPolicy().max_batch} windows, beyond the "
        f"measured peak of {SERVING_BATCH_WINDOWS}. Every request past the peak pays "
        "queueing latency for throughput that is not there."
    )


def test_batch_thirty_two_is_recorded_as_worse_than_the_optimum(measurement: dict) -> None:
    """Pin the specific regression, so the reason for the change survives in the suite.

    If a future sweep on different hardware shows 32 winning, this fails and the constant
    should be revisited. That is the correct outcome: the number follows the measurement,
    not the other way round.
    """
    rows = {r["batch_windows"]: r for r in measurement["batch_sweep"]}
    if 32 not in rows:
        pytest.skip("this sweep did not include batch 32")
    best = max(measurement["batch_sweep"], key=lambda r: r["windows_per_second"])
    assert rows[32]["windows_per_second"] <= best["windows_per_second"], (
        "batch 32 is now the fastest point; the serving constant should be revisited"
    )


def test_the_artifact_records_how_many_samples_each_point_took(measurement: dict) -> None:
    """A p95 from three samples and a p95 from thirty are different claims.

    The sweep spends a fixed time budget per point, so large batches get fewer samples.
    Recording the count is what stops a reader treating them as equally precise.
    """
    for row in measurement["batch_sweep"]:
        assert row.get("samples", 0) >= 1, f"no sample count recorded for {row}"
