"""Phase 3. Long-document inference by overlapping windows.

Truncating at 512 tokens means a document that is human for 400 tokens and AI for the
next 4,000 is scored entirely on its human opening. Windows of 512 with stride 384
give a 128-token overlap so no boundary falls in a blind spot.

Whether prediction smoothing across windows actually improves boundary F1 is an
experiment, not an assumption. See data_spec_v1 section 10, item 3.
"""

from __future__ import annotations


def windows(n_tokens: int, size: int = 512, stride: int = 384) -> list[tuple[int, int]]:
    """Pure function, testable without a model."""
    if size <= 0 or stride <= 0 or stride > size:
        raise ValueError("require 0 < stride <= size")
    if n_tokens <= size:
        return [(0, n_tokens)]
    out: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + size, n_tokens)
        out.append((start, end))
        if end == n_tokens:
            break
        start += stride
    return out


def aggregate(window_scores: list[float], weights: list[float] | None = None) -> float:
    if not window_scores:
        raise ValueError("no windows to aggregate")
    if weights is None:
        return sum(window_scores) / len(window_scores)
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return sum(s * w for s, w in zip(window_scores, weights, strict=True)) / total
