"""Token-level metrics. Plan section 22, gap G1.

Three metrics that answer three different questions, and conflating them hides real
failures:

  **token F1**    per-token classification quality. High even for a model that finds the
                  right regions with badly placed edges, because most tokens sit in the
                  middle of a region.

  **segment F1**  did the model find the right REGIONS? A predicted segment counts as a
                  match when its IoU with a true segment clears a threshold. This is what
                  a user reading a highlighted document actually perceives.

  **boundary F1** did the model put the edges in the right PLACE? Boundaries are scored
                  with a tolerance window, because being one token late on a seam is not
                  the same failure as missing the seam entirely.

A model can score 0.95 token F1 while missing half the segments: if AI regions are 5
percent of tokens, predicting "all human" scores 0.95 on the majority class alone. That
is exactly why segment and boundary F1 exist, and why token F1 must never be quoted on
its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forge.modeling.alignment import IGNORE_INDEX

HUMAN_ID = 0


@dataclass
class TokenScores:
    token_f1_macro: float
    token_f1_per_class: dict[int, float]
    segment_f1: float
    segment_precision: float
    segment_recall: float
    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    n_true_segments: int
    n_pred_segments: int
    support: dict[int, int]

    def as_dict(self) -> dict:
        return {
            "token_f1_macro": round(self.token_f1_macro, 6),
            "token_f1_per_class": {k: round(v, 6) for k, v in self.token_f1_per_class.items()},
            "segment_f1": round(self.segment_f1, 6),
            "segment_precision": round(self.segment_precision, 6),
            "segment_recall": round(self.segment_recall, 6),
            "boundary_f1": round(self.boundary_f1, 6),
            "boundary_precision": round(self.boundary_precision, 6),
            "boundary_recall": round(self.boundary_recall, 6),
            "n_true_segments": self.n_true_segments,
            "n_pred_segments": self.n_pred_segments,
            "support": self.support,
        }


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p, r = tp / (tp + fp), tp / (tp + fn)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def token_f1(y_true, y_pred, ignore_index: int = IGNORE_INDEX) -> tuple[float, dict[int, float], dict[int, int]]:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    mask = yt != ignore_index
    yt, yp = yt[mask], yp[mask]
    classes = sorted(set(yt.tolist()) | set(yp.tolist()))
    per, support = {}, {}
    for c in classes:
        tp = int(((yp == c) & (yt == c)).sum())
        fp = int(((yp == c) & (yt != c)).sum())
        fn = int(((yp != c) & (yt == c)).sum())
        per[int(c)] = _f1(tp, fp, fn)
        support[int(c)] = int((yt == c).sum())
    macro = float(np.mean(list(per.values()))) if per else 0.0
    return macro, per, support


def extract_segments(labels, ignore_index: int = IGNORE_INDEX, positive_only: bool = True):
    """Contiguous runs of one label. Ignored positions break a run rather than joining it.

    Joining across ignored positions would merge two genuinely separate AI regions that
    happen to be split by a special token, inflating segment recall.
    """
    labs = list(labels)
    segs, start, cur = [], None, None
    for i, lab in enumerate(labs + [None]):
        if lab == ignore_index or lab is None or lab != cur:
            if cur is not None and start is not None and (not positive_only or cur != HUMAN_ID):
                segs.append((start, i, int(cur)))
            start, cur = (None, None) if (lab == ignore_index or lab is None) else (i, lab)
        elif start is None:
            start, cur = i, lab
    return segs


def _iou(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def segment_f1(y_true, y_pred, iou_threshold: float = 0.5,
               ignore_index: int = IGNORE_INDEX) -> tuple[float, float, float, int, int]:
    """Greedy one-to-one matching by IoU, highest first.

    One-to-one matters: without it, one huge predicted segment could match every true
    segment and score perfect recall while being useless to a reader.
    """
    true_segs = extract_segments(y_true, ignore_index)
    pred_segs = extract_segments(y_pred, ignore_index)
    pairs = sorted(
        ((_iou(t, p), ti, pi) for ti, t in enumerate(true_segs) for pi, p in enumerate(pred_segs)
         if t[2] == p[2] and _iou(t, p) >= iou_threshold),
        key=lambda x: -x[0],
    )
    used_t, used_p, tp = set(), set(), 0
    for _, ti, pi in pairs:
        if ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        tp += 1
    fp, fn = len(pred_segs) - tp, len(true_segs) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return _f1(tp, fp, fn), prec, rec, len(true_segs), len(pred_segs)


def extract_boundaries(labels, ignore_index: int = IGNORE_INDEX) -> list[int]:
    """Positions where the label changes, skipping ignored positions."""
    out, prev = [], None
    for i, lab in enumerate(labels):
        if lab == ignore_index:
            continue
        if prev is not None and lab != prev:
            out.append(i)
        prev = lab
    return out


def boundary_f1(y_true, y_pred, tolerance: int = 2,
                ignore_index: int = IGNORE_INDEX) -> tuple[float, float, float]:
    """Boundaries within `tolerance` tokens count as matched, one to one.

    Tolerance exists because being one token late on a seam is a different failure from
    missing the seam, and a zero-tolerance metric cannot tell them apart.
    """
    t = extract_boundaries(y_true, ignore_index)
    p = extract_boundaries(y_pred, ignore_index)
    used, tp = set(), 0
    for pb in p:
        for i, tb in enumerate(t):
            if i not in used and abs(pb - tb) <= tolerance:
                used.add(i)
                tp += 1
                break
    fp, fn = len(p) - tp, len(t) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return _f1(tp, fp, fn), prec, rec


def score_tokens(y_true, y_pred, iou_threshold: float = 0.5, tolerance: int = 2,
                 ignore_index: int = IGNORE_INDEX) -> TokenScores:
    macro, per, support = token_f1(y_true, y_pred, ignore_index)
    sf1, sp, sr, nt, np_ = segment_f1(y_true, y_pred, iou_threshold, ignore_index)
    bf1, bp, br = boundary_f1(y_true, y_pred, tolerance, ignore_index)
    return TokenScores(macro, per, sf1, sp, sr, bf1, bp, br, nt, np_, support)
