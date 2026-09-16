"""Evaluation lab. These tests target the ways a lab reports a good number that is not real."""

import json

import numpy as np

from forge.evaluation.lab import (
    adversarial_table,
    choose_threshold,
    min_humans_for_fpr,
    run_lab,
    score_regime,
)


def _preds(n_human, n_ai, human_mean=0.05, ai_mean=0.9, ai_sd=0.10, seed=0):
    """Synthetic score distributions. ai_sd matters: with a tight AI distribution every
    regime scores an FNR of 0 at any sane threshold and the test asserts nothing."""
    rng = np.random.default_rng(seed)
    h = np.clip(rng.normal(human_mean, 0.05, n_human), 0, 1)
    a = np.clip(rng.normal(ai_mean, ai_sd, n_ai), 0, 1)
    y = np.concatenate([np.zeros(n_human, int), np.ones(n_ai, int)])
    return y.tolist(), np.concatenate([h, a]).tolist()


def test_threshold_is_fitted_on_validation_and_reused_everywhere():
    """Re-picking the threshold per regime reports the best case each time and hides that
    one deployed threshold cannot satisfy all regimes at once."""
    val = _preds(5000, 5000, seed=1)
    thr = choose_threshold(*val, 0.001)
    r1 = score_regime("R1_iid", *_preds(5000, 5000, seed=2), thr, 0.001)
    # a genuinely harder regime: AI scores overlap the human distribution
    r3 = score_regime(
        "R3_unseen_generator", *_preds(5000, 5000, ai_mean=0.35, ai_sd=0.30, seed=3), thr, 0.001
    )
    assert r1.fnr < 0.05, f"in-distribution FNR should be low, got {r1.fnr}"
    assert r3.fnr > r1.fnr, f"harder regime must degrade under the same threshold: {r3.fnr} vs {r1.fnr}"


def test_an_fpr_of_zero_on_fifty_humans_is_reported_as_unmeasurable():
    """0/50 is not evidence of a 0.1 percent FPR. Reporting it as 0.0 is the single
    easiest way to publish a false headline number."""
    r = score_regime("tiny", *_preds(50, 500), 0.5, 0.001)
    assert r.fpr_measurable is False
    assert "need" in r.note


def test_a_large_human_set_is_measurable():
    r = score_regime("big", *_preds(20000, 2000), 0.5, 0.001)
    assert r.fpr_measurable is True and r.note == ""


def test_min_humans_scales_inversely_with_the_budget():
    assert min_humans_for_fpr(0.001) > min_humans_for_fpr(0.01)


def test_contamination_is_reported_per_benchmark():
    val = _preds(3000, 3000)
    rep = run_lab(
        {"R1_iid": _preds(1000, 1000)}, val, 0.001, "forge-0.1", "v0.1", "abc123",
        train_hashes={"a", "b", "c"},
        eval_hashes={"raid": {"c", "d"}, "mage": {"x", "y"}},
    )
    assert rep.contamination["raid"]["clean"] is False
    assert rep.contamination["raid"]["overlap"] == 1
    assert rep.contamination["mage"]["clean"] is True


def test_headline_pulls_from_named_regimes_not_the_easiest_one():
    """The docs/evaluation.md row must come from R1, R3 and R5 by name, so it cannot be
    accidentally filled from whichever regime scored best."""
    val = _preds(3000, 3000)
    rep = run_lab(
        {
            "R1_iid": _preds(2000, 2000, seed=4),
            "R3_unseen_generator": _preds(2000, 2000, ai_mean=0.40, ai_sd=0.30, seed=5),
            "R5_adversarial": _preds(2000, 2000, ai_mean=0.30, ai_sd=0.30, seed=6),
            "easy_regime": _preds(2000, 2000, ai_mean=0.99, seed=7),
        },
        val, 0.001, "forge-0.1", "v0.1", "abc123",
    )
    head = rep.headline()
    r3 = next(r for r in rep.regimes if r.name == "R3_unseen_generator")
    assert head["ood_auroc"] == round(r3.auroc, 6)


def test_headline_is_empty_when_a_required_regime_is_missing():
    val = _preds(3000, 3000)
    rep = run_lab({"R1_iid": _preds(1000, 1000)}, val, 0.001, "forge-0.1", "v0.1", "abc")
    assert rep.headline()["ood_auroc"] is None


def test_adversarial_delta_shows_what_an_attack_buys():
    t = adversarial_table(0.05, {"paraphrase": 0.30, "whitespace": 0.06})
    assert t["paraphrase"]["delta_fnr"] > t["whitespace"]["delta_fnr"]
    assert abs(t["paraphrase"]["delta_fnr"] - 0.25) < 1e-9


def test_report_writes_a_reproducible_artifact(tmp_path):
    val = _preds(3000, 3000)
    rep = run_lab({"R1_iid": _preds(1000, 1000)}, val, 0.001, "forge-0.1", "v0.1", "abc123")
    p = rep.write(tmp_path)
    d = json.loads(p.read_text())
    for k in ("model_version", "dataset_version", "code_commit", "threshold", "fpr_budget"):
        assert d[k] is not None, f"{k} missing; a result without provenance is an anecdote"
