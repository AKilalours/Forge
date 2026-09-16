"""Experiment tracking must leave evidence, including when it did not happen.

WHY THIS FILE EXISTS. train.py enabled W&B behind a silent three-way condition: the
backend had to be wandb, the run must not be a smoke run, and WANDB_API_KEY had to be set.
If any failed, tracking did not happen and nothing said so. A run that was never tracked
produced a run record indistinguishable from one that was, so months later "is there a
W&B link for this experiment?" had no answer in the repo.

That is the same shape as every other finding in this project: a mechanism that does
nothing, quietly, and looks identical to one that worked.

The fix is that the run record carries a `tracking` block stating whether tracking was on,
and if not, which of the three conditions stopped it. When it IS on, the block carries the
run id and the run URL, so the README links a committed artifact rather than a URL typed
in by hand. A link in prose is a claim; a link in a run record is evidence.

These tests are pure structure checks on the block's contract, so they need no GPU, no
network and no W&B account.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "experiments"
RUN_RECORDS = ("indist_baseline.json", "indist_mirror.json")


def _records() -> list[tuple[str, dict]]:
    out = []
    for name in RUN_RECORDS:
        path = REPORTS / name
        if path.exists():
            out.append((name, json.loads(path.read_text())))
    return out


def test_a_tracking_block_states_enabled_and_a_reason_when_it_is_not() -> None:
    """The contract, checked on a synthetic block so it holds before any new run exists."""
    for block in (
        {"backend": "wandb", "enabled": False, "reason": "WANDB_API_KEY is not set"},
        {"backend": None, "enabled": False, "reason": "tracking.backend is not wandb"},
        {"backend": "wandb", "enabled": True, "project": "forge", "run_name": "x",
         "run_id": "abc123", "run_url": "https://wandb.ai/u/forge/runs/abc123"},
    ):
        assert "enabled" in block
        if block["enabled"]:
            assert block.get("run_url", "").startswith("https://"), (
                "an enabled tracking block must carry a resolvable run URL, or the README "
                "has nothing to link that anyone can verify"
            )
        else:
            assert block.get("reason"), (
                "tracking that did not happen must say which condition stopped it; silence "
                "is what made this unanswerable in the first place"
            )


@pytest.mark.parametrize("name", RUN_RECORDS)
def test_committed_run_records_do_not_claim_tracking_they_cannot_evidence(name: str) -> None:
    """A record may predate this change and have no tracking block at all. It may not claim
    tracking was enabled without a URL to show for it."""
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{name} not committed")
    block = json.loads(path.read_text()).get("tracking")
    if block is None:
        pytest.skip(f"{name} predates the tracking block; nothing is claimed either way")
    if block.get("enabled"):
        assert str(block.get("run_url", "")).startswith("https://"), (
            f"{name} says tracking was enabled but carries no run URL"
        )
    else:
        assert block.get("reason"), f"{name} says tracking was off but not why"


def test_the_readme_only_links_a_wandb_run_that_a_record_carries() -> None:
    """Stops the README linking a run URL that no committed artifact backs.

    This is the check that makes the MLOps row honest. Pasting a W&B link into a README is
    a claim anybody can make; the link having to appear in a run record means the run is
    tied to a code_commit, a dataset_version and a set of validation numbers.
    """
    md = (ROOT / "README.md").read_text()
    import re

    linked = set(re.findall(r"https://wandb\.ai/[^\s)\"'<>]+", md))
    if not linked:
        pytest.skip("the README does not link a W&B run yet")
    recorded = {
        str(rec.get("tracking", {}).get("run_url"))
        for _, rec in _records()
        if rec.get("tracking", {}).get("run_url")
    }
    unbacked = {u for u in linked if u.rstrip("/") not in {r.rstrip("/") for r in recorded}}
    assert not unbacked, (
        f"README links W&B runs that no committed run record carries: {sorted(unbacked)}. "
        "Commit the run record that produced them, or remove the link."
    )
