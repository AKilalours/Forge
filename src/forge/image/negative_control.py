"""The test that decides whether any image result is real.

THE PROBLEM. A human/AI image classifier can reach near-perfect accuracy by learning the
difference between two ENCODING PIPELINES rather than between a photograph and a
generation. Every headline metric looks excellent while this happens, so no amount of
staring at AUROC will reveal it.

THE TEST. Take human images only. Split them into two arbitrary halves and label one half
0 and the other 1. There is no real signal: the labels are a coin flip by construction. If
a model trained on that reaches meaningfully better than chance, it is reading something
about the images that correlates with how they were selected or encoded, and every number
produced by the real experiment is void.

This is cheap, it runs before the expensive work, and it is the only thing that separates
"my detector works" from "my pipeline leaks".

The harness takes the trainer as a callable so it can be exercised without a GPU. What is
tested here is the LOGIC of the control: that labels are arbitrary but reproducible, that
the pass/fail threshold accounts for sampling noise, and that a leaking pipeline is
actually caught.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

CONTROL_SALT = "negative-control-v1"


def arbitrary_label(image_id: str, salt: str = CONTROL_SALT) -> int:
    """A label with no relationship to the image, stable across runs and machines.

    Hash-derived rather than random so a failed control can be reproduced exactly, and so
    two runs of the control are comparable rather than two different coin flips.
    """
    digest = hashlib.sha256(f"{salt}:{image_id}".encode()).hexdigest()
    return int(digest[:8], 16) & 1


def arbitrary_labels(image_ids: Iterable[str], salt: str = CONTROL_SALT) -> dict[str, int]:
    return {i: arbitrary_label(i, salt) for i in image_ids}


def chance_margin(n: int, sigmas: float = 4.0) -> float:
    """How far above 0.5 an AUROC can drift on `n` samples before it means something.

    A finite sample will not score exactly 0.5 even on pure noise. The standard error of
    AUROC under the null is bounded above by 1/sqrt(12 * n_pos * n_neg / (n_pos+n_neg))
    for balanced classes; the simpler and slightly conservative 1/(2*sqrt(n)) is used
    here, scaled by `sigmas`.

    Being conservative is the right bias: a control that fails spuriously costs an
    afternoon, and a control that passes spuriously costs the credibility of every number
    in the paper.
    """
    if n <= 0:
        raise ValueError("need at least one sample to compute a margin")
    return sigmas / (2.0 * math.sqrt(n))


@dataclass(frozen=True)
class ControlResult:
    auroc: float
    n: int
    margin: float

    @property
    def passed(self) -> bool:
        return self.auroc <= 0.5 + self.margin

    def explain(self) -> str:
        if self.passed:
            return (
                f"negative control PASSED: AUROC {self.auroc:.4f} on {self.n} human images, "
                f"within {self.margin:.4f} of chance. No source signature detected."
            )
        return (
            f"negative control FAILED: AUROC {self.auroc:.4f} on {self.n} human images, "
            f"which is {self.auroc - 0.5:.4f} above chance and beyond the {self.margin:.4f} "
            "margin. The model is separating human images from human images, so it is "
            "reading the pipeline rather than the content. Every number from the real "
            "experiment is void until this passes. Look first at normalisation: are both "
            "classes going through forge.image.normalize with the same policy?"
        )


class NegativeControlFailed(RuntimeError):
    pass


def run_negative_control(
    image_ids: Sequence[str],
    train_and_score: Callable[[dict[str, int]], float],
    salt: str = CONTROL_SALT,
    sigmas: float = 4.0,
) -> ControlResult:
    """Label human images arbitrarily, train, and report whether chance was beaten.

    `train_and_score` receives the arbitrary labels and returns a held-out AUROC. It is
    injected rather than imported so this logic is testable without a GPU, and so the same
    control can wrap either detector.
    """
    if len(image_ids) < 2:
        raise ValueError("a negative control needs at least two images")
    labels = arbitrary_labels(image_ids, salt=salt)
    if len(set(labels.values())) < 2:
        raise ValueError(
            "arbitrary labelling produced a single class; the sample is too small or the "
            "salt is degenerate"
        )
    auroc = float(train_and_score(labels))
    return ControlResult(auroc=auroc, n=len(image_ids), margin=chance_margin(len(image_ids), sigmas))


def assert_negative_control(result: ControlResult) -> None:
    """Refuse to proceed on a leaking pipeline. Call this before reporting anything."""
    if not result.passed:
        raise NegativeControlFailed(result.explain())
