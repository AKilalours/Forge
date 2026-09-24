"""What the deployed image page must say, and must not imply.

The failure this pins is not a crash. A ChatGPT-generated image scored 2.1% and the page
rendered a green tick and "NO AI DETECTED", which a reader takes as "this image is fine".
The detector's own probe measured 0.55 recall at its operating point, fitted in sample on
29 images, so a negative from it is closer to a coin flip than to evidence of absence.
The score was right. The presentation was not.
"""

from __future__ import annotations

from forge.ui.render import detector_caveat, image_result

MEASURED = {
    "available": True,
    "model_id": "umm-maybe/AI-image-detector",
    "recall_at_threshold": 0.55,
    "recall_sample": {"ai": 20, "human": 9},
    "recall_in_sample": True,
}


def _payload(detector: dict, verdict: str = "human") -> dict:
    return {
        "assessment": {"available": True, "verdict": verdict, "confidence": 0.0209,
                       "reason": "", "detail": ""},
        "evidence": {"streams": [], "conflict": "low"},
        "detector": detector,
        "filename": "x.jpg",
        "size_bytes": 1024,
        "authenticity": [],
        "robustness": [{"attack": "original", "delta": 0.0, "ai_probability": 0.021,
                        "changed": False}],
        # Attribution and maps are left empty: their own renderers are covered
        # elsewhere, and what this file is about is which sections appear.
        "attribution_map": {},
        "maps": [],
    }


def test_a_weak_negative_carries_the_measured_recall():
    caveat = detector_caveat(MEASURED)

    assert "55%" in caveat and "29 images" in caveat
    assert "45%" in caveat  # what it misses, stated rather than left to the reader


def test_the_caveat_comes_from_the_payload_not_from_a_literal():
    # Re-measuring the detector must change the page. A hardcoded sentence would not.
    other = dict(MEASURED, recall_at_threshold=0.9, recall_sample={"ai": 100, "human": 100})

    assert "90%" in detector_caveat(other) and "200 images" in detector_caveat(other)


def test_a_perfect_or_unmeasured_detector_gets_no_caveat():
    assert detector_caveat(dict(MEASURED, recall_at_threshold=1.0)) == ""
    assert detector_caveat({"available": False}) == ""


def test_a_weak_negative_is_not_rendered_as_a_clean_result():
    """The green tick was the actual defect. The word stays; the styling does not."""
    weak = image_result(_payload(MEASURED), compact=True)
    strong = image_result(_payload(dict(MEASURED, recall_at_threshold=1.0)), compact=True)

    assert "NO AI DETECTED" in weak          # the wording was already correct
    assert 'class="assess big warn"' in weak  # but not dressed as a confident pass
    assert 'class="assess big ok"' in strong


def test_an_ai_verdict_is_unaffected_by_the_caveat():
    # Recall limits what a NEGATIVE is worth. A positive from the same detector is not
    # weakened by it, and softening that too would be its own inaccuracy.
    out = image_result(_payload(MEASURED, verdict="ai"), compact=True)

    assert 'class="assess big hot"' in out


def test_compact_drops_the_diagnostics_and_full_keeps_them():
    compact = image_result(_payload(MEASURED), compact=True)
    full = image_result(_payload(MEASURED), compact=False)

    for section in ("Robustness", "occlusion", "Pixel statistics"):
        assert section.lower() not in compact.lower()
    assert "robustness" in full.lower()
