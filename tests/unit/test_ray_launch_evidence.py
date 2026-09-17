"""A run may only be written as evidence if it reported evidence.

WHY THIS FILE EXISTS. scripts/ray_train_launch.py wrote an artifact that said:

    "Ray Train placed the workers, built the process group, gave each worker a distinct
     rank, and ran the same gradient-accumulation step this repository uses elsewhere."

with backend, global_batch_size, wall_seconds and world_size all null, and exited 0.

The cause was Ray Train V2: Result.metrics does not exist in Ray 2.58, so every
metrics.get(...) returned None. But the API change is not the interesting part. The
interesting part is that the CLAIM and the EVIDENCE were produced independently. The
sentence was a hardcoded string; the fields came from a dict that happened to be empty.
Nothing checked that the second supported the first, so the script asserted its conclusion
and shipped.

Eighteen other findings in this project were mechanisms that never ran. This one ran,
produced nothing, and described itself as having succeeded. That is worse, and it is the
reason the check lives in a function that can be tested without a Ray cluster: a rule you
cannot test is an intention.

The artifact test at the bottom is the other half. It reads whatever is committed under
reports/experiments/ray/ and refuses null evidence, so a future fabricated run cannot be
committed even if someone bypasses the launcher.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from ray_train_launch import (  # noqa: E402
    REQUIRED_WORKER_METRICS,
    validate_worker_metrics,
)

GOOD = {"backend": "gloo", "global_batch_size": 64, "wall_seconds": 0.702, "world_size": 2}


def test_a_complete_report_is_accepted() -> None:
    assert validate_worker_metrics(GOOD, workers=2, target_global_batch=64) == []


def test_an_empty_report_is_refused() -> None:
    """THE REGRESSION. This exact input produced a confident artifact."""
    problems = validate_worker_metrics({}, workers=2, target_global_batch=64)
    assert problems, "an empty report must never be written as evidence"
    assert "fabrication" in problems[0], (
        "the message must say what is wrong with writing it, not just that a key is absent"
    )


@pytest.mark.parametrize("missing", REQUIRED_WORKER_METRICS)
def test_any_single_missing_field_is_refused(missing: str) -> None:
    """Partial evidence is not evidence. A run that reports a backend but no world size
    cannot support a sentence about how many workers were placed."""
    metrics = {k: v for k, v in GOOD.items() if k != missing}
    assert validate_worker_metrics(metrics, workers=2, target_global_batch=64)


def test_a_worker_count_that_does_not_match_the_request_is_refused() -> None:
    """Ray silently giving fewer workers than asked for would turn a distributed run into
    a single-process run wearing a distributed label."""
    problems = validate_worker_metrics({**GOOD, "world_size": 1}, workers=2,
                                       target_global_batch=64)
    assert any("scaling config was not honoured" in p for p in problems)


def test_a_changed_global_batch_is_refused() -> None:
    """The invariant this repository holds in four places now: forge.training.scaling
    across world sizes, DistributedRun across strategies, the Spark runner across partition
    counts, and here across LAUNCHERS. Changing who starts the workers must not change what
    they train on."""
    problems = validate_worker_metrics({**GOOD, "global_batch_size": 32}, workers=2,
                                       target_global_batch=64)
    assert any("must not change what is trained" in p for p in problems)


def test_every_committed_ray_artifact_carries_real_evidence() -> None:
    """The other half: a fabricated artifact cannot be committed even by hand.

    Reads what is actually in the repository rather than what the launcher would produce,
    because the failure being prevented is a file on disk making a claim.
    """
    ray_dir = ROOT / "reports" / "experiments" / "ray"
    if not ray_dir.exists():
        pytest.skip("no ray artifacts committed")
    artifacts = sorted(ray_dir.glob("ray_ws*.json"))
    if not artifacts:
        pytest.skip("no ray run records committed")

    for path in artifacts:
        record = json.loads(path.read_text())
        for field in ("backend", "global_batch_size", "wall_seconds", "workers"):
            assert record.get(field) is not None, (
                f"{path.name} claims a successful Ray run but {field} is null. The "
                "artifact's own prose is not evidence; these fields are."
            )
        assert record["global_batch_size"] == 64, (
            f"{path.name} reports a global batch of {record['global_batch_size']}, which "
            "does not match what every other launcher in this repo produces"
        )
        assert "what_this_does_not_show" in record, (
            f"{path.name} has no statement of its own limits. A single-node CPU Ray run "
            "that does not say so reads as a distributed training result"
        )
