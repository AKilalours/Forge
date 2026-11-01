"""Drift detection on the production score distribution.

A rising AI-probability distribution in production usually means one of two things: the
world changed (a new model everyone started using), or the detector did (a bad deploy).
Both go to the Failure Atlas. Neither goes straight to training.

The distinction that matters for interpreting an alert:

  score drift  the distribution of what the model OUTPUTS moved. Could be the model,
               could be the traffic.
  input drift  the distribution of what the model RECEIVES moved. If inputs are stable
               and scores moved, the model changed, and that is a deploy problem.

Reporting only score drift makes every incident ambiguous. Both are computed here.

PSI thresholds follow the usual convention (0.1 minor, 0.25 significant), which is a rule
of thumb from credit scoring, not a law. It is stated as a convention in the output so
nobody mistakes it for a derived number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PSI_MINOR = 0.10
PSI_SIGNIFICANT = 0.25
_EPS = 1e-6


def _hist(x, bins: int, lo: float, hi: float) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(x, float), bins=np.linspace(lo, hi, bins + 1))
    total = int(counts.sum())
    p = counts / total if total else np.zeros_like(counts, dtype=float)
    return np.clip(p, _EPS, None)


def psi(expected, actual, bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> float:
    """Population Stability Index. Symmetric, unlike KL."""
    e, a = _hist(expected, bins, lo, hi), _hist(actual, bins, lo, hi)
    return float(np.sum((a - e) * np.log(a / e)))


def kl_divergence(expected, actual, bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> float:
    """KL(actual || expected). Asymmetric: it punishes mass appearing where the reference
    had almost none, which is the direction you care about for a new generator showing up."""
    e, a = _hist(expected, bins, lo, hi), _hist(actual, bins, lo, hi)
    return float(np.sum(a * np.log(a / e)))


@dataclass
class DriftReport:
    name: str
    psi: float
    kl: float
    reference_mean: float
    current_mean: float
    n_reference: int
    n_current: int
    measurable: bool
    note: str = ""

    @property
    def severity(self) -> str:
        if not self.measurable:
            return "unmeasurable"
        if self.psi >= PSI_SIGNIFICANT:
            return "significant"
        if self.psi >= PSI_MINOR:
            return "minor"
        return "stable"

    def as_dict(self) -> dict:
        return {
            "name": self.name, "psi": round(self.psi, 6), "kl": round(self.kl, 6),
            "reference_mean": round(self.reference_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "n_reference": self.n_reference, "n_current": self.n_current,
            "severity": self.severity, "measurable": self.measurable,
            "thresholds": {"minor": PSI_MINOR, "significant": PSI_SIGNIFICANT,
                           "note": "conventional rule of thumb, not a derived bound"},
            "note": self.note,
        }


MIN_SAMPLES_FOR_DRIFT = 200


def check(name: str, reference, current, bins: int = 10, lo: float = 0.0, hi: float = 1.0) -> DriftReport:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    measurable = len(cur) >= MIN_SAMPLES_FOR_DRIFT and len(ref) >= MIN_SAMPLES_FOR_DRIFT
    return DriftReport(
        name=name,
        psi=psi(ref, cur, bins, lo, hi) if measurable else float("nan"),
        kl=kl_divergence(ref, cur, bins, lo, hi) if measurable else float("nan"),
        reference_mean=float(ref.mean()) if len(ref) else float("nan"),
        current_mean=float(cur.mean()) if len(cur) else float("nan"),
        n_reference=len(ref), n_current=len(cur), measurable=measurable,
        note="" if measurable else (
            f"need at least {MIN_SAMPLES_FOR_DRIFT} samples on both sides; a PSI computed "
            "on a handful of requests is noise and will page someone at 3am for nothing"
        ),
    )


def diagnose(score_drift: DriftReport, input_drift: DriftReport) -> str:
    """Which of the two explanations an alert supports.

    Without this pairing, every drift alert is ambiguous and the first hour of an incident
    goes to working out which question is even being asked.
    """
    if not (score_drift.measurable and input_drift.measurable):
        return "insufficient data to diagnose"
    scores_moved = score_drift.psi >= PSI_MINOR
    inputs_moved = input_drift.psi >= PSI_MINOR
    if scores_moved and not inputs_moved:
        return (
            "scores moved while inputs held steady: suspect the model or the deploy, "
            "not the traffic. Check the deployed model version first."
        )
    if scores_moved and inputs_moved:
        return (
            "both moved: the traffic changed and the model responded. Likely a genuine "
            "distribution shift; route samples to the Failure Atlas."
        )
    if inputs_moved and not scores_moved:
        return (
            "inputs moved but scores did not: either the model is insensitive to this "
            "change, or it is stuck. Check the prediction distribution is not degenerate."
        )
    return "stable"
