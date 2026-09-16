"""The profiler wrapper's contract, exercised on CPU.

A profiling helper that only works on the GPU is a helper nobody can test, and this
project has already been bitten once by a module that could not run where CI runs.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")


def test_it_ranks_what_the_step_spent_time_on(tmp_path):
    import torch

    from forge.training.profiling import profile_step

    def step():
        a = torch.randn(256, 256)
        (a @ a).sum().item()

    summary = profile_step(step, out_dir=tmp_path, label="unit")

    assert summary["device"] in {"cpu", "cuda"}
    assert summary["top"], "a step that did work must rank at least one operator"
    assert summary["total_self_time_us"] > 0


def test_both_artifacts_land_and_the_summary_is_small_enough_to_commit(tmp_path):
    import torch

    from forge.training.profiling import profile_step

    profile_step(lambda: torch.randn(64, 64).sum().item(), out_dir=tmp_path, label="unit")

    trace = tmp_path / "unit_trace.json"
    summary = tmp_path / "unit_summary.json"
    assert trace.exists() and summary.exists()
    # The reason the trace is gitignored and the summary is not.
    assert summary.stat().st_size < 64_000


def _share_sum_tolerance(rows: list[dict], measured_us: float) -> float:
    """How far above 1.0 the shares may legitimately sum, derived rather than guessed.

    THE BUG THIS REPLACES. The original assertion was `sum(shares) <= 1.0001`, a tolerance
    chosen for floating-point noise. That is the wrong error source. profiling.py rounds
    each share to 4 decimals, so every row carries up to 5e-5 of ROUNDING error, and the
    sum of N rows can exceed 1.0 by up to N * 5e-5. At ten rows that is 5e-4, five times
    the tolerance I allowed.

    It passed on 3.12 and failed on 3.10 at 1.0002, which made it look like a
    version-specific bug. It is not. The two interpreters produce slightly different
    operator sets and timings, so the roundings land differently, and the assertion was
    close enough to the boundary that it was a coin flip. It would have failed eventually
    on either. A tolerance that depends on which way ten roundings happen to fall is a
    flaky test wearing the costume of a strict one.

    Second, smaller source: self_device_us is itself rounded to 0.1us before the division,
    contributing 0.05/measured per row. Negligible on a GPU step measured in seconds,
    material on a unit test whose total is microseconds, which is exactly where this test
    runs.
    """
    n = len(rows)
    return n * 5e-5 + (n * 0.05 / measured_us if measured_us else 0.0)


def test_shares_are_fractions_of_measured_time(tmp_path):
    import torch

    from forge.training.profiling import profile_step

    s = profile_step(lambda: torch.randn(128, 128).sum().item(), out_dir=tmp_path, label="unit")
    written = json.loads((tmp_path / "unit_summary.json").read_text())
    rows = written["top"]
    assert rows == s["top"]
    assert all(0.0 <= r["share"] <= 1.0 for r in rows)

    total = sum(r["share"] for r in rows)
    tolerance = _share_sum_tolerance(rows, written["total_self_time_us"])
    assert total <= 1.0 + tolerance, (
        f"{len(rows)} shares sum to {total}, above 1.0 by more than the "
        f"{tolerance:.2e} that rounding each to 4 decimals can explain. That is a real "
        "accounting error, not rounding: an operator is being counted twice, or the "
        "denominator is not the total self time."
    )


def test_the_tolerance_scales_with_row_count_and_is_not_a_magic_number():
    """Pin the derivation, so nobody widens the constant until the test stops failing.

    Raising a tolerance until a test passes converts a broken invariant into a green one.
    The tolerance has to follow from the rounding precision and the number of rows, and
    this states the arithmetic so a future edit has to argue with it.
    """
    rows = [{"share": 0.1}] * 10
    # BOTH terms, which my first version of this assertion forgot: the 4-decimal share
    # rounding AND the 0.1us rounding of self_device_us before the division. Writing only
    # the first is how a derivation quietly becomes a magic number again.
    expected = 10 * 5e-5 + 10 * 0.05 / 1e6
    assert _share_sum_tolerance(rows, measured_us=1e6) == pytest.approx(expected, rel=1e-9)
    # More rows, more rounding, more slack. Fewer rows, less.
    assert _share_sum_tolerance([{"share": 1.0}], 1e6) < _share_sum_tolerance(rows, 1e6)
    # A tiny measured total makes the microsecond rounding matter, and the tolerance grows.
    assert _share_sum_tolerance(rows, measured_us=10.0) > _share_sum_tolerance(rows, 1e6)
    # It must never be zero, or the first rounding in either direction fails the suite.
    assert _share_sum_tolerance([{"share": 1.0}], 1e9) > 0
