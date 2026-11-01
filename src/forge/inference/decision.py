"""Turning a score into a decision, including the decision not to decide.

--------------------------------------------------------------------------------
Abstention
--------------------------------------------------------------------------------
A detector operating at a 0.1 percent FPR budget is tuned so that being wrong about a
human is very expensive. Forcing every document into a binary call throws that away in
the region where the model is least sure. A document scoring 0.52 against a threshold of
0.51 is reported as "AI" with the same interface as one scoring 0.999, and a reader has
no way to tell them apart.

So the decision has three outcomes, not two:

    score < abstain_low        -> human
    abstain_low <= score < hi  -> UNCERTAIN, no claim made
    score >= abstain_high      -> ai

The uncertain band is derived from the calibrated score, which is why calibration is a
release-gate criterion rather than a nicety. It is a product decision as much as a
technical one: an AI-detection result is frequently used in ways that affect a person,
and "I do not know" is a legitimate and much safer answer than a coin flip dressed as a
verdict.

--------------------------------------------------------------------------------
Provenance on every response
--------------------------------------------------------------------------------
Every decision carries the model version, the operating threshold and the FPR budget the
threshold was calibrated against. A score without those is uninterpretable: 0.96 means
nothing unless you know what the model was tuned to protect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    HUMAN = "human"
    AI = "ai"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class DecisionPolicy:
    threshold: float                 # from the evaluation lab, fitted to the FPR budget
    fpr_budget: float
    abstain_low: float | None = None
    abstain_high: float | None = None
    model_version: str = "unknown"
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        lo, hi = self.abstain_low, self.abstain_high
        if (lo is None) != (hi is None):
            raise ValueError("abstain_low and abstain_high must be set together")
        if lo is not None and hi is not None:
            if not 0.0 <= lo <= hi <= 1.0:
                raise ValueError("require 0 <= abstain_low <= abstain_high <= 1")
            if not lo <= self.threshold <= hi:
                raise ValueError(
                    f"threshold {self.threshold} sits outside the uncertain band "
                    f"[{lo}, {hi}]. The band exists to straddle the decision point; a "
                    "threshold outside it means one of the two verdicts is unreachable."
                )

    @property
    def abstains(self) -> bool:
        return self.abstain_low is not None


@dataclass
class Decision:
    verdict: Verdict
    ai_probability: float
    confidence: float
    model_version: str
    threshold: float
    fpr_budget: float
    calibrated: bool
    abstained: bool
    reason: str = ""
    segments: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "prediction": self.verdict.value,
            "ai_probability": round(self.ai_probability, 6),
            "confidence": round(self.confidence, 6),
            "model_version": self.model_version,
            "threshold": round(self.threshold, 6),
            "fpr_budget": self.fpr_budget,
            "uncertainty": {
                "calibrated": self.calibrated,
                "abstained": self.abstained,
                "reason": self.reason,
            },
            "segments": self.segments,
        }


def decide(score: float, policy: DecisionPolicy, segments: list[dict] | None = None) -> Decision:
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score {score} is not a probability")

    low, high = policy.abstain_low, policy.abstain_high
    if low is not None and high is not None and low <= score < high:
        verdict, abstained = Verdict.UNCERTAIN, True
        reason = (
            f"score {score:.3f} falls in the uncertain band "
            f"[{low:.3f}, {high:.3f}); no claim is made"
        )
    else:
        verdict = Verdict.AI if score >= policy.threshold else Verdict.HUMAN
        abstained, reason = False, ""

    return Decision(
        verdict=verdict,
        ai_probability=score,
        # Distance from the decision point, not the raw score. A score of 0.02 against a
        # threshold of 0.5 is a confident HUMAN call; reporting confidence 0.02 there
        # would be actively misleading.
        confidence=abs(score - policy.threshold) / max(policy.threshold, 1 - policy.threshold),
        model_version=policy.model_version,
        threshold=policy.threshold,
        fpr_budget=policy.fpr_budget,
        calibrated=policy.calibrated,
        abstained=abstained,
        reason=reason,
        segments=segments or [],
    )


def band_from_validation(
    val_labels, val_scores, fpr_budget: float, abstain_fraction: float = 0.02
) -> tuple[float, float]:
    """Derive the uncertain band from validation scores, not from intuition.

    The band is placed so it covers `abstain_fraction` of validation documents around the
    operating threshold. Choosing it by eye produces either an abstention rate nobody
    will accept in production, or a band so narrow it never fires.
    """
    import numpy as np

    from forge.evaluation.metrics import threshold_at_fpr

    if not 0.0 <= abstain_fraction < 1.0:
        raise ValueError("abstain_fraction must be in [0, 1)")
    scores = np.sort(np.asarray(val_scores, float))
    thr = threshold_at_fpr(val_labels, val_scores, fpr_budget)
    if abstain_fraction == 0.0:
        return thr, thr

    pos = int(np.searchsorted(scores, thr))
    half = max(int(len(scores) * abstain_fraction / 2), 1)
    lo = float(scores[max(pos - half, 0)])
    hi = float(scores[min(pos + half, len(scores) - 1)])
    return min(lo, thr), max(hi, thr)
