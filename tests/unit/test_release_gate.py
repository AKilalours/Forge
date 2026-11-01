from forge.evaluation.release_gate import evaluate

GATE = {
    "max_fpr": 0.001,
    "max_fnr": 0.10,
    "min_ood_auroc": 0.90,
    "max_ece": 0.05,
    "max_p95_latency_ms": 100,
    "no_regression_on": ["R1_iid"],
}


def test_accuracy_alone_does_not_ship_a_model():
    # Great AI detection, but it accuses 1 percent of humans. Must fail.
    metrics = {"fpr": 0.01, "fnr": 0.01, "ood_auroc": 0.99, "ece": 0.01, "p95_latency_ms": 20}
    r = evaluate(metrics, GATE)
    assert not r.passed
    assert any("fpr" in f for f in r.failures)


def test_clean_candidate_passes():
    metrics = {"fpr": 0.0005, "fnr": 0.05, "ood_auroc": 0.95, "ece": 0.02,
               "p95_latency_ms": 40, "R1_iid": 0.95}
    assert evaluate(metrics, GATE, baseline={"R1_iid": 0.94}).passed


def test_missing_metrics_and_baseline_fail_closed():
    result = evaluate({}, GATE)
    assert not result.passed
    assert any("missing required metric" in failure for failure in result.failures)
    assert any("missing baseline metric" in failure for failure in result.failures)


def test_regression_blocks_release():
    metrics = {"fpr": 0.0005, "fnr": 0.05, "ood_auroc": 0.95, "ece": 0.02, "p95_latency_ms": 40, "R1_iid": 0.90}
    r = evaluate(metrics, GATE, baseline={"R1_iid": 0.94})
    assert not r.passed
