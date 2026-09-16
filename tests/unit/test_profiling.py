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


def test_shares_are_fractions_of_measured_time(tmp_path):
    import torch

    from forge.training.profiling import profile_step

    s = profile_step(lambda: torch.randn(128, 128).sum().item(), out_dir=tmp_path, label="unit")
    rows = json.loads((tmp_path / "unit_summary.json").read_text())["top"]
    assert rows == s["top"]
    assert all(0.0 <= r["share"] <= 1.0 for r in rows)
    assert sum(r["share"] for r in rows) <= 1.0001
