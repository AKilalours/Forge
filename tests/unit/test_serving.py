"""Phase 8 serving: decision policy, abstention, batching."""

import pytest

from forge.inference.batching import (
    BatchPolicy,
    QueuedRequest,
    QueueOverflow,
    fits_latency_budget,
    plan_batches,
    should_flush,
)
from forge.inference.decision import (
    DecisionPolicy,
    Verdict,
    band_from_validation,
    decide,
)


def _policy(**kw):
    base = dict(threshold=0.5, fpr_budget=0.001, abstain_low=0.4, abstain_high=0.6,
                model_version="forge-0.5", calibrated=True)
    base.update(kw)
    return DecisionPolicy(**base)


def test_the_uncertain_band_produces_a_third_verdict():
    """A detector at a 0.1 percent FPR budget is tuned so being wrong about a human is
    expensive. Forcing a binary call throws that away exactly where the model is least
    sure, and 0.52 gets the same interface as 0.999."""
    p = _policy()
    assert decide(0.02, p).verdict is Verdict.HUMAN
    assert decide(0.45, p).verdict is Verdict.UNCERTAIN
    assert decide(0.55, p).verdict is Verdict.UNCERTAIN
    assert decide(0.99, p).verdict is Verdict.AI


def test_abstention_is_recorded_with_a_reason():
    d = decide(0.5, _policy())
    assert d.abstained and "uncertain band" in d.reason


def test_confidence_is_distance_from_the_threshold_not_the_raw_score():
    """A score of 0.02 against a threshold of 0.5 is a CONFIDENT human call. Reporting
    confidence 0.02 there would be actively misleading."""
    p = _policy(abstain_low=None, abstain_high=None)
    assert decide(0.02, p).confidence > 0.9
    assert decide(0.98, p).confidence > 0.9
    assert decide(0.50, p).confidence < 0.05


def test_every_decision_carries_its_operating_point():
    """A score of 0.96 is uninterpretable without knowing what the model was tuned to
    protect."""
    d = decide(0.9, _policy()).as_dict()
    assert d["model_version"] == "forge-0.5"
    assert d["threshold"] == 0.5
    assert d["fpr_budget"] == 0.001
    assert d["uncertainty"]["calibrated"] is True


def test_a_band_that_does_not_straddle_the_threshold_is_rejected():
    with pytest.raises(ValueError, match="outside the uncertain band"):
        _policy(threshold=0.9, abstain_low=0.4, abstain_high=0.6)


def test_half_specified_band_is_rejected():
    with pytest.raises(ValueError):
        DecisionPolicy(threshold=0.5, fpr_budget=0.001, abstain_low=0.4)


def test_policy_without_a_band_never_abstains():
    p = _policy(abstain_low=None, abstain_high=None)
    assert not p.abstains
    assert decide(0.5, p).verdict is Verdict.AI
    assert not decide(0.5, p).abstained


def test_band_is_derived_from_validation_not_intuition():
    """Choosing the band by eye gives either an abstention rate nobody accepts or a band
    that never fires."""
    import numpy as np

    rng = np.random.default_rng(0)
    labels = [0] * 3000 + [1] * 3000
    scores = np.concatenate([rng.uniform(0, 0.5, 3000), rng.uniform(0.5, 1.0, 3000)]).tolist()
    lo, hi = band_from_validation(labels, scores, fpr_budget=0.01, abstain_fraction=0.04)
    assert lo <= hi
    covered = sum(1 for s in scores if lo <= s < hi) / len(scores)
    assert 0.01 < covered < 0.12, covered


def test_zero_abstain_fraction_gives_a_degenerate_band():
    labels = [0] * 100 + [1] * 100
    scores = [i / 200 for i in range(200)]
    lo, hi = band_from_validation(labels, scores, 0.05, abstain_fraction=0.0)
    assert lo == hi


def test_scores_outside_zero_one_are_rejected():
    with pytest.raises(ValueError):
        decide(1.5, _policy())


# ------------------------------------------------------------------ batching

def _q(n, windows=1, t=0.0):
    return [QueuedRequest(f"r{i}", windows, t) for i in range(n)]


def test_batching_counts_windows_not_requests():
    """Thirty-two single-window requests and thirty-two hundred-window requests are the
    same request count and wildly different work; batching by request count blows the
    latency budget whenever long documents arrive together."""
    batches = plan_batches(_q(8, windows=8), BatchPolicy(max_batch=32), now_ms=0)
    assert all(b.n_windows <= 32 for b in batches)
    assert len(batches) == 2


def test_arrival_order_is_preserved():
    """Sorting by length to reduce padding starves long documents under load, which shows
    up as a P99 cliff on exactly the requests users care most about."""
    queue = [QueuedRequest("small", 1, 0.0), QueuedRequest("huge", 30, 1.0),
             QueuedRequest("small2", 1, 2.0)]
    ids = [i for b in plan_batches(queue, BatchPolicy(max_batch=32), 0) for i in b.request_ids]
    assert ids == ["small", "huge", "small2"]


def test_an_oversized_request_gets_its_own_batch_rather_than_being_dropped():
    batches = plan_batches([QueuedRequest("giant", 100, 0.0)], BatchPolicy(max_batch=32), 0)
    assert len(batches) == 1 and batches[0].n_windows == 100


def test_flush_on_a_full_batch():
    assert should_flush(_q(32), BatchPolicy(max_batch=32, max_wait_ms=10), now_ms=0)


def test_flush_on_age_bounds_tail_latency_at_low_load():
    """Without the age check, one request sits in the queue until enough traffic arrives
    to fill a batch, so latency is WORST when load is lightest."""
    policy = BatchPolicy(max_batch=32, max_wait_ms=10)
    assert not should_flush(_q(1, t=0.0), policy, now_ms=5)
    assert should_flush(_q(1, t=0.0), policy, now_ms=11)


def test_empty_queue_does_not_flush():
    assert not should_flush([], BatchPolicy(), now_ms=100)


def test_queue_overflow_is_an_error_not_unbounded_growth():
    with pytest.raises(QueueOverflow):
        plan_batches(_q(50), BatchPolicy(max_queue=10), 0)


def test_latency_budget_check():
    p = BatchPolicy(max_wait_ms=10)
    assert fits_latency_budget(p, model_p95_ms=40, budget_p95_ms=100)
    assert not fits_latency_budget(p, model_p95_ms=95, budget_p95_ms=100)


def test_invalid_batch_policy_is_rejected():
    with pytest.raises(ValueError):
        BatchPolicy(max_batch=0)
