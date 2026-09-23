"""The part of the ONNX path that decides whether an export may be deployed.

The export itself needs torch and onnxruntime and is exercised by running the script. What
is tested here is `compare`, which is what turns two lists of probabilities into a verdict
about the export, and which has no dependencies at all. It is also where the dangerous
mistake lives: an export that agrees to four decimal places on average can still move a
document across the deployed threshold, and on the mirror arm that threshold is 0.998252,
so a 1e-3 shift is the difference between "AI" and "human".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from export_onnx import compare  # noqa: E402

THRESHOLD = 0.998252


def test_a_tiny_delta_across_the_threshold_is_a_flip():
    # 4e-6 apart, and on opposite sides. Average agreement says this export is perfect.
    result = compare([0.998250], [0.998254], THRESHOLD)

    assert result["max_abs_delta"] < 1e-5
    assert result["verdict_flips"] == 1


def test_a_large_delta_that_stays_on_one_side_is_not_a_flip():
    result = compare([0.10], [0.30], THRESHOLD)

    assert result["max_abs_delta"] == pytest.approx(0.2)
    assert result["verdict_flips"] == 0


def test_zero_flips_is_reported_beside_how_many_could_have_flipped():
    """Zero flips in a sample with nothing near the threshold is not evidence.

    Without `near_threshold_windows` a parity report on eight confidently-human documents
    would read as proof the export is safe, when the sample could not have shown otherwise.
    """
    result = compare([0.01, 0.02], [0.01, 0.02], THRESHOLD)

    assert result["verdict_flips"] == 0
    assert result["near_threshold_windows"] == 0


def test_mismatched_window_counts_are_refused():
    # Zipping them would silently compare window i against window i and drop the rest,
    # reporting an agreement over a sample that is not the same sample.
    with pytest.raises(SystemExit, match="different window counts"):
        compare([0.1, 0.2], [0.1], THRESHOLD)


def test_an_empty_comparison_does_not_claim_perfect_agreement():
    result = compare([], [], THRESHOLD)

    assert result["windows"] == 0
    assert result["max_abs_delta"] == 0.0
