"""Phase 8. The release gate.

A model does not ship because validation accuracy went up. It ships when it passes
every threshold below and regresses on nothing. This function is pure so the policy
itself is testable without a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GateResult:
    passed: bool
    failures: list[str]


def evaluate(metrics: dict, gate: dict, baseline: dict | None = None) -> GateResult:
    failures: list[str] = []

    def _number(value: object, label: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(f"{label} must be numeric")
            return None
        result = float(value)
        if not math.isfinite(result):
            failures.append(f"{label} must be finite")
            return None
        return result

    def _upper(key: str, gate_key: str) -> None:
        if gate_key not in gate:
            return
        if key not in metrics:
            failures.append(f"missing required metric: {key}")
            return
        value, threshold = _number(metrics[key], key), _number(gate[gate_key], gate_key)
        if value is not None and threshold is not None and value > threshold:
            failures.append(f"{key}={value:.5g} exceeds {gate_key}={threshold:.5g}")

    _upper("fpr", "max_fpr")
    _upper("fnr", "max_fnr")
    _upper("ece", "max_ece")
    _upper("p95_latency_ms", "max_p95_latency_ms")

    if "min_ood_auroc" in gate and "ood_auroc" not in metrics:
        failures.append("missing required metric: ood_auroc")
    elif "min_ood_auroc" in gate:
        value = _number(metrics["ood_auroc"], "ood_auroc")
        threshold = _number(gate["min_ood_auroc"], "min_ood_auroc")
        if value is not None and threshold is not None and value < threshold:
            failures.append(
                f"ood_auroc={value:.5g} below min_ood_auroc={threshold:.5g}"
            )

    for regime in gate.get("no_regression_on", []):
        if baseline is None or regime not in baseline:
            failures.append(f"missing baseline metric for required regime: {regime}")
        elif regime not in metrics:
            failures.append(f"missing candidate metric for required regime: {regime}")
        else:
            candidate = _number(metrics[regime], f"candidate {regime}")
            previous = _number(baseline[regime], f"baseline {regime}")
            if candidate is not None and previous is not None and candidate < previous:
                failures.append(f"regression on {regime}: {candidate:.5g} < {previous:.5g}")

    return GateResult(passed=not failures, failures=failures)
