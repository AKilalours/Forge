"""`forge mine` must actually mine.

The bug this file exists for: the command loaded the config, raised PhaseNotImplemented if
`--model` was absent, and RETURNED if it was present. Satisfying the documented precondition
produced exit code 0, no output and no work. `run_round` was never called from anywhere but
the tests.

The first test fails against that version: it asserts the round ran and wrote a file.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from forge.cli import app
from forge.hard_negative.mining import ReserveDoc

runner = CliRunner()

CONFIG = "configs/training/hard_negative_minimal.yaml"
REGISTERS = {
    "academic": "Citation-heavy academic prose with a formal register, item {i}, on methodology.",
    "legal": "SECTION {i}. PURSUANT TO THE FOREGOING PROVISIONS THE COMMITTEE SHALL DETERMINE.",
    "casual": "hey so basically you gotta just try it and see what happens, number {i}, no big deal",
}


class FakeScorer:
    """Confidently wrong on two registers of three, like test_mining_round's."""

    model_version = "forge-test@deadbeef"
    threshold = 0.5

    def __init__(self, *_args, **_kwargs) -> None:
        self.calls = 0

    def score(self, texts):
        self.calls += 1
        return [0.96 if ("Citation-heavy" in t or "PURSUANT" in t) else 0.02 for t in texts]


def _reserve(n: int = 80):
    return [
        ReserveDoc(f"{reg}_{i}", f"grp_{reg}_{i}", tpl.format(i=i), "fineweb", "web", reg)
        for reg, tpl in REGISTERS.items()
        for i in range(n)
    ]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Swap the two pieces that need weights and parquet, leave the round itself real."""
    import forge.hard_negative.reserve as R

    monkeypatch.setattr(R, "CheckpointScorer", FakeScorer)
    monkeypatch.setattr(R, "load_reserve", lambda root, limit=None: _reserve())
    return tmp_path


def test_mine_runs_the_round_and_writes_it(wired):
    """THE REGRESSION TEST. The old command exited 0 having written nothing."""
    out = wired / "reports"
    result = runner.invoke(
        app, ["mine", "--config", CONFIG, "--out", str(out), "--round", "r1"]
    )
    assert result.exit_code == 0, result.output

    written = out / "r1.json"
    assert written.exists(), "the round was not written; the command did no work"
    payload = json.loads(written.read_text())
    assert payload["mining"]["scanned"] == 240
    assert payload["mining"]["above_confidence_gate"] > 0
    assert payload["n_train_refs"] > 0


def test_mine_reports_what_it_did(wired):
    result = runner.invoke(app, ["mine", "--config", CONFIG, "--out", str(wired / "r")])
    assert "scanning 240 reserve documents" in result.output
    assert "selected" in result.output
    assert "written" in result.output


def test_mine_only_mines_the_registers_the_model_failed_on(wired):
    out = wired / "reports"
    runner.invoke(app, ["mine", "--config", CONFIG, "--out", str(out), "--round", "r2"])
    payload = json.loads((out / "r2.json").read_text())
    assert payload["n_train_refs"] + payload["n_holdout_refs"] <= 160, (
        "the casual register was never a failure and must not be mined"
    )


def test_a_missing_detector_refuses_with_a_reason(monkeypatch, tmp_path):
    """No weights means no mining. It must say so, not mine with a default threshold."""
    import forge.hard_negative.reserve as R
    from forge.inference.scorer import ArmUnavailable

    def _no_weights(*_a, **_k):
        raise ArmUnavailable("no checkpoint at outputs/forge_min_mirror/best.pt")

    monkeypatch.setattr(R, "CheckpointScorer", _no_weights)
    result = runner.invoke(app, ["mine", "--config", CONFIG, "--out", str(tmp_path)])
    assert result.exit_code != 0
    assert "trained detector" in str(result.exception) or "trained detector" in result.output


def test_the_mining_gate_must_not_be_looser_than_the_operating_threshold(wired):
    """Guards the invariant scan() enforces: mining is at least as strict as production."""
    result = runner.invoke(
        app,
        ["mine", "--config", CONFIG, "--out", str(wired / "r"), "--min-confidence", "0.1"],
    )
    assert result.exit_code != 0, "a gate below the operating threshold must be refused"


def test_the_mining_gate_has_no_stale_default() -> None:
    """Regression: --min-confidence defaulted to 0.90 and every arm deploys above 0.996.

    The gate may not be looser than the deployed threshold, or it selects documents
    production never called AI, so that default could not succeed for any trained arm.
    It failed inside scan(), after loading a 735 MB checkpoint and the whole reserve
    pool: roughly a minute of work to be told the arguments were wrong.

    Defaulting to the arm's own threshold also means a retrained arm cannot leave the
    default behind, which a literal would.
    """
    import inspect

    from forge.cli import mine

    default = inspect.signature(mine).parameters["min_confidence"].default
    assert getattr(default, "default", default) is None


def test_the_flywheel_dag_does_not_pin_a_gate_either() -> None:
    # The DAG carried the same 0.90, so every scheduled round would have failed on
    # arrival. It now omits the flag and inherits the arm's threshold.
    from pathlib import Path

    dag = (Path(__file__).resolve().parents[2]
           / "orchestration" / "dags" / "forge_flywheel.py").read_text()
    assert "--min-confidence 0.90" not in dag
