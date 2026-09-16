"""Phase 8 monitoring, feedback and registry promotion."""

import numpy as np
import pytest

from forge.evaluation.release_gate import evaluate
from forge.monitoring import drift
from forge.monitoring.feedback import (
    FeedbackItem,
    FeedbackQueue,
    FeedbackState,
    IllegalTransition,
    UnverifiedLabel,
    now,
)
from forge.registry.model_registry import (
    PromotionRefused,
    Registry,
    Stage,
    validate_entry,
)

RNG = np.random.default_rng(0)


def _scores(mean, n=1000, sd=0.1):
    return np.clip(RNG.normal(mean, sd, n), 0, 1).tolist()


# ------------------------------------------------------------------ drift

def test_a_stable_distribution_reports_stable():
    assert drift.check("scores", _scores(0.3), _scores(0.3)).severity == "stable"


def test_a_shifted_distribution_reports_significant():
    r = drift.check("scores", _scores(0.30), _scores(0.60))
    assert r.severity == "significant" and r.psi > drift.PSI_SIGNIFICANT


def test_drift_on_a_handful_of_requests_is_unmeasurable_not_zero():
    """A PSI computed on twenty requests is noise, and reporting it as a number pages
    someone at 3am for nothing."""
    r = drift.check("scores", _scores(0.3), _scores(0.9, n=20))
    assert r.severity == "unmeasurable" and not r.measurable
    assert "at least" in r.note


def test_psi_is_symmetric_and_kl_is_not():
    a, b = _scores(0.3), _scores(0.6)
    assert drift.psi(a, b) == pytest.approx(drift.psi(b, a), rel=1e-6)
    assert drift.kl_divergence(a, b) != pytest.approx(drift.kl_divergence(b, a), rel=1e-3)


def test_thresholds_are_labelled_as_a_convention_not_a_bound():
    d = drift.check("scores", _scores(0.3), _scores(0.3)).as_dict()
    assert "rule of thumb" in d["thresholds"]["note"]


def test_diagnosis_separates_a_bad_deploy_from_a_traffic_shift():
    """Without pairing score drift with input drift, every alert is ambiguous and the
    first hour of an incident goes to working out which question is being asked."""
    scores_moved = drift.check("scores", _scores(0.3), _scores(0.7))
    inputs_stable = drift.check("inputs", _scores(0.5), _scores(0.5))
    assert "deploy" in drift.diagnose(scores_moved, inputs_stable)

    inputs_moved = drift.check("inputs", _scores(0.3), _scores(0.7))
    assert "distribution shift" in drift.diagnose(scores_moved, inputs_moved)

    scores_stable = drift.check("scores", _scores(0.5), _scores(0.5))
    assert "stuck" in drift.diagnose(scores_stable, inputs_moved)


# ------------------------------------------------------------------ feedback

def _item(iid="f1"):
    return FeedbackItem(iid, "hash1", "forge-0.5", "ai", user_claim="human", submitted_at=now())


def test_raw_feedback_never_reaches_the_atlas():
    """A document's author is exactly the person most motivated to dispute a CORRECT 'AI'
    verdict. A detector that learns from unverified disagreement learns to agree with
    whoever complains loudest."""
    q = FeedbackQueue()
    q.submit(_item())
    assert q.to_atlas() == []


def test_verification_requires_a_verifier_supplied_label():
    q = FeedbackQueue()
    q.submit(_item())
    item = q.get("f1")
    item.transition(FeedbackState.UNDER_REVIEW)
    with pytest.raises(UnverifiedLabel):
        item.transition(FeedbackState.VERIFIED)                       # no verifier
    with pytest.raises(UnverifiedLabel):
        item.transition(FeedbackState.VERIFIED, verifier="alice")     # no label
    item.transition(FeedbackState.VERIFIED, verifier="alice", verified_label="human")
    assert q.to_atlas() == [item]


def test_the_users_claim_is_not_the_label():
    q = FeedbackQueue()
    q.submit(_item())
    item = q.get("f1")
    item.transition(FeedbackState.UNDER_REVIEW)
    item.transition(FeedbackState.VERIFIED, verifier="alice", verified_label="ai")
    assert item.user_claim == "human" and item.verified_label == "ai"


def test_illegal_transitions_are_refused():
    item = _item()
    with pytest.raises(IllegalTransition):
        item.transition(FeedbackState.VERIFIED, verifier="a", verified_label="human")
    item.transition(FeedbackState.UNDER_REVIEW)
    item.transition(FeedbackState.REJECTED)
    with pytest.raises(IllegalTransition):
        item.transition(FeedbackState.UNDER_REVIEW)


def test_rejection_rate_is_tracked_because_it_is_itself_a_signal():
    q = FeedbackQueue()
    for i in range(4):
        q.submit(_item(f"f{i}"))
    for i in range(4):
        it = q.get(f"f{i}")
        it.transition(FeedbackState.UNDER_REVIEW)
        it.transition(FeedbackState.REJECTED if i < 3 else FeedbackState.VERIFIED,
                      **({} if i < 3 else {"verifier": "a", "verified_label": "human"}))
    assert q.stats()["rejection_rate"] == 0.75


# ------------------------------------------------------------------ registry

GATE = {"max_fpr": 0.001, "max_fnr": 0.10, "min_ood_auroc": 0.90, "max_ece": 0.05,
        "max_p95_latency_ms": 100}
GOOD = {"fpr": 0.0005, "fnr": 0.05, "ood_auroc": 0.95, "ece": 0.02, "p95_latency_ms": 40}
BAD = {"fpr": 0.02, "fnr": 0.05, "ood_auroc": 0.95, "ece": 0.02, "p95_latency_ms": 40}


def _entry(v="v0.5"):
    return {
        "version": v, "weights_uri": f"s3://forge/{v}", "model_config": {"backbone": "deberta"},
        "dataset_version": "v0.2", "code_commit": "abc123", "metrics": GOOD,
        "environment": {"torch": "2.3"},
    }


def test_weights_alone_are_not_a_model_version():
    assert "dataset_version" in validate_entry({"version": "v1", "weights_uri": "s3://x"})
    with pytest.raises(ValueError, match="cannot be reproduced"):
        Registry().register({"version": "v1", "weights_uri": "s3://x"})


def test_promotion_without_a_gate_result_is_refused():
    """'Not evaluated' is not 'no failures found'. Conflating them is how a model that
    was never tested reaches production."""
    r = Registry()
    r.register(_entry())
    with pytest.raises(PromotionRefused, match="no gate result"):
        r.promote("v0.5", Stage.CANARY)


def test_a_failing_gate_blocks_promotion():
    r = Registry()
    r.register(_entry())
    r.record_gate("v0.5", evaluate(BAD, GATE))
    with pytest.raises(PromotionRefused, match="failed the release gate"):
        r.promote("v0.5", Stage.CANARY)


def test_production_promotion_goes_through_canary():
    r = Registry()
    r.register(_entry())
    r.record_gate("v0.5", evaluate(GOOD, GATE))
    with pytest.raises(PromotionRefused, match="through canary"):
        r.promote("v0.5", Stage.PRODUCTION)
    r.promote("v0.5", Stage.CANARY)
    assert r.promote("v0.5", Stage.PRODUCTION).stage is Stage.PRODUCTION


def test_promoting_a_new_model_archives_the_old_one():
    r = Registry()
    for v in ("v0.5", "v0.6"):
        r.register(_entry(v))
        r.record_gate(v, evaluate(GOOD, GATE))
        r.promote(v, Stage.CANARY)
        r.promote(v, Stage.PRODUCTION)
    assert r.get("v0.5").stage is Stage.ARCHIVED
    assert r.production().version == "v0.6"


def test_rollback_skips_canary_on_purpose():
    """The model being rolled back to already served production traffic, so canary has
    nothing left to discover, and the value of a rollback is that it is fast."""
    r = Registry()
    for v in ("v0.5", "v0.6"):
        r.register(_entry(v))
        r.record_gate(v, evaluate(GOOD, GATE))
        r.promote(v, Stage.CANARY)
        r.promote(v, Stage.PRODUCTION)
    r.rollback("v0.5")
    assert r.production().version == "v0.5"
    assert r.get("v0.6").stage is Stage.ARCHIVED


def test_registry_round_trips_to_disk(tmp_path):
    import json

    r = Registry(tmp_path / "registry.json")
    r.register(_entry())
    p = r.save()
    d = json.loads(p.read_text())
    assert d["v0.5"]["dataset_version"] == "v0.2"
    assert d["v0.5"]["code_commit"] == "abc123"
