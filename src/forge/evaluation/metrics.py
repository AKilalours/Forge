"""Detection metrics.

Accuracy is deliberately not the headline. In this problem the two errors are not
symmetric: calling a human's essay AI-generated has a real cost to a real person,
while missing one AI document does not. So FPR is the primary metric and everything
else is supporting evidence.

Everything here is numpy-only so metrics can be computed without torch installed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "false_positive_rate",
    "false_negative_rate",
    "threshold_at_fpr",
    "auroc",
    "expected_calibration_error",
    "robustness_delta",
    "data_efficiency",
]


def _as_arrays(y_true, y_score):  # type: ignore[no-untyped-def]
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    return y_true, y_score


def false_positive_rate(y_true, y_score, threshold: float = 0.5) -> float:
    """FP / (FP + TN). Positive class = AI. A false positive is an accused human."""
    y_true, y_score = _as_arrays(y_true, y_score)
    pred = y_score >= threshold
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    return fp / (fp + tn) if (fp + tn) else float("nan")


def false_negative_rate(y_true, y_score, threshold: float = 0.5) -> float:
    y_true, y_score = _as_arrays(y_true, y_score)
    pred = y_score >= threshold
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    return fn / (fn + tp) if (fn + tp) else float("nan")


def threshold_at_fpr(y_true, y_score, target_fpr: float) -> float:
    """The production threshold is chosen by the FPR budget, not by argmax accuracy.

    Returns the lowest threshold whose FPR on human documents is <= target_fpr.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    human_scores = np.sort(y_score[y_true == 0])
    if human_scores.size == 0:
        raise ValueError("no human examples to calibrate a threshold against")
    k = int(np.ceil((1.0 - target_fpr) * human_scores.size)) - 1
    k = min(max(k, 0), human_scores.size - 1)
    return float(np.nextafter(human_scores[k], np.inf))


def auroc(y_true, y_score) -> float:
    """Rank-based AUROC via the Mann-Whitney U identity, with tie handling."""
    y_true, y_score = _as_arrays(y_true, y_score)
    pos, neg = y_score[y_true == 1], y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=float)
    # average ranks within tie groups
    sorted_scores = y_score[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y_true == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def expected_calibration_error(y_true, y_score, n_bins: int = 15) -> float:
    """ECE. A detector that says 0.97 should be wrong about 3 percent of the time.

    Uncalibrated confidence is worse than no confidence, because downstream users
    (and the release gate) will treat the number as a probability.
    """
    y_true, y_score = _as_arrays(y_true, y_score)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = y_score.size
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (y_score > lo) & (y_score <= hi) if lo > 0 else (y_score >= lo) & (y_score <= hi)
        if not mask.any():
            continue
        acc = float(np.mean((y_score[mask] >= 0.5).astype(int) == y_true[mask]))
        conf = float(np.mean(np.maximum(y_score[mask], 1 - y_score[mask])))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def robustness_delta(fnr_attack: float, fnr_clean: float) -> float:
    """How much an attack buys the evader. Positive means the attack worked."""
    return fnr_attack - fnr_clean


def data_efficiency(ood_improvement: float, added_examples: int) -> float:
    """The research metric: OOD gain per additional synthetic example.

    This is the number that would make failure-driven selection interesting even if
    absolute accuracy matched random augmentation.
    """
    if added_examples <= 0:
        raise ValueError("added_examples must be positive")
    return ood_improvement / added_examples
