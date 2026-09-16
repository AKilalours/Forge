"""What the interface is allowed to say.

Two rules, both about claims the product cannot support:

  A detector cannot establish that a person made something. It reports whether the input
  resembles the AI distribution it was trained on. "HUMAN" is a claim about the world;
  "NO AI DETECTED" is a claim about the model, and only the second one is true.

  Version identifiers belong in the payload, not on the face of the product. "FORGE-Image
  v0.4" beside a verdict reads as a released product to anyone who does not know the
  project, and the string changes every run.
"""

from __future__ import annotations

import pathlib
import re

PAGE = pathlib.Path(__file__).resolve().parents[2] / "api" / "forge_app.py"


def page() -> str:
    return PAGE.read_text()


def test_the_verdict_word_is_no_ai_detected_not_human():
    body = page()
    assert "NO AI DETECTED" in body
    assert "AI DETECTED" in body
    assert "UNCERTAIN" in body
    assert re.search(r"WORD\s*=\s*\{[^}]*human:\s*'NO AI DETECTED'", body), (
        "the verdict map must translate the model's 'human' label before display"
    )


def test_no_version_identifier_is_rendered_beside_a_verdict_or_a_score():
    """Provenance lives in the technical footer, not in the headline panels."""
    body = page()
    banner = body[body.index("function banner("):body.index("function renderImage(")]
    for forbidden in ("model_version", "report_version", "v0.", "version"):
        assert forbidden not in banner, f"the banner renders {forbidden!r}"


def test_provenance_is_still_reachable_rather_than_deleted():
    """Demoted, not removed. A detector with no identifiable build cannot be audited."""
    body = page()
    assert "model_version" in body, "provenance was dropped entirely"
    assert "for reproducibility rather than" in body


def test_the_header_states_system_state_without_a_build_number():
    body = page()
    header = body[body.index("let s={}, img={};"):body.index("const drop=$('#drop')")]
    assert "image detector " in header
    # Only the rendered string matters; the endpoint path may legitimately mention arms.
    shown = header[header.index("txt.textContent"):header.index("dot.className")]
    assert "v0." not in shown and "version" not in shown
    assert "arms" not in shown, "the arm count is a developer readout, not product state"


def test_the_primary_evidence_stream_is_the_detector_and_is_named_not_guessed():
    """The hierarchy depends on one key. A silent fallback would invert it.

    An earlier version looked up 'visual' and fell back to streams[0]. That happened to be
    the right stream, so a wrong key rendered a correct panel: the class of bug that only
    shows up when the order changes. If the detector stream is absent the panel says so
    rather than promoting camera metadata to PRIMARY evidence.
    """
    body = page()
    assert "s.key==='visual_model'" in body
    assert "ev.streams[0]" not in body, "a positional fallback for the primary stream is back"


def test_the_key_the_page_looks_up_is_the_key_the_evidence_engine_emits():
    """Pins the contract across the two files, so a rename fails here, not in the browser."""
    import io

    import numpy as np
    import pytest

    pytest.importorskip("PIL")
    from PIL import Image

    from forge.image.report import build_report

    buffer = io.BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (160, 120, 3), dtype="uint8")).save(
        buffer, format="JPEG", quality=90
    )
    streams = build_report(buffer.getvalue(), with_stability=False).as_dict()["evidence"]["streams"]
    assert any(s["key"] == "visual_model" for s in streams), (
        f"the page looks up 'visual_model'; the engine emits {[s['key'] for s in streams]}"
    )


def test_development_language_is_off_the_production_surface():
    """The interface states what the system does, not the argument behind the decision."""
    from forge.image.report import Assessment

    reason = Assessment().reason
    for phrase in ("architecturally complete", "has not been trained", "fabricated"):
        assert phrase not in reason.lower(), f"{phrase!r} is still in the headline"
    assert "Visual detector unavailable" in reason
    assert Assessment().detail, "the longer explanation was deleted rather than moved"


def test_the_working_detector_is_the_tab_that_opens_first():
    """Text is trained and calibrated; the image detector does not exist.

    Opening on Image made the first thing anyone saw "No verdict", which reads as "the
    product does not work" while a working detector sat one click away.
    """
    body = page()
    start = body.index('<div class="tabs">')
    tabs = body[start:body.index("</div>\n  <div class=\"status\"", start)]
    assert tabs.index('data-tab="text"') < tabs.index('data-tab="image"')
    assert '<div class="tab on" data-tab="text">' in body
    assert '<section id="pane-image" class="hide">' in body


def test_the_diagnostics_are_collapsed_and_the_verdict_is_not():
    """Product question first, diagnostics second. Collapsed, never deleted."""
    body = page()
    for section in ("Pixel statistics", "Details"):
        assert f"<summary>{section}" in body, f"{section} is not a collapsed section"
    assert 'details class="card wide fold"' in body
    # The verdict, the evidence and the robustness panel are open; nothing else is.
    render = body[body.index("function renderImage(d){"):body.index("$('#go').onclick")]
    head = render[:render.index("<details")]
    for panel in ("banner({", "Evidence", "Robustness", "What the verdict rests on"):
        assert panel in head, f"{panel} is buried in a fold"


def test_the_panels_that_only_said_not_available_are_gone():
    """An AI-attribution panel reading "not available" three times is not information.

    It needs a trained localisation head. Until then the panel is absent, not empty: a row
    of "not available" pills tells a reader nothing except that something is missing.
    """
    body = page()
    assert "AI attribution" not in body
    assert "Attributed area" not in body
    assert "Mixed content" not in body


def test_no_verdict_is_shown_when_the_polarity_has_not_been_measured():
    """An inverted verdict accuses the person holding a real photograph.

    Worse than no verdict, and not recoverable in front of a reader who knows the answer.
    """
    import io

    import pytest as _pytest

    from forge.image.report import build_report

    _pytest.importorskip("PIL")
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 120, 90)).save(buffer, format="JPEG")
    report = build_report(buffer.getvalue(), with_detector=False)
    assert report.assessment.available is False
    assert report.assessment.verdict is None


def test_the_footer_carries_the_author():
    body = page()
    assert "Built by Akila Lourdes Miriyala Francis" in body
    assert "Report schema" not in body


def test_no_shipped_string_says_the_visual_detector_does_not_exist():
    """THE REGRESSION. A measured detector shipped while the payload still denied it.

    `Assessment.detail` and `Attribution.reason` are dataclass defaults, so nothing in the
    happy path exercised them, and every construction site overrides `detail`. That is
    exactly why they rotted: a detector was built, measured and wired to the page, and the
    JSON these defaults feed still told anyone reading it that no visual detector exists.
    A reviewer reading the API response rather than the page would have been told the
    project's main image claim was vapour.

    The wording is allowed to say a verdict is unavailable. It is not allowed to say the
    detector was never built.
    """
    from forge.image.report import Assessment, Attribution

    forbidden = ("does not exist yet", "which does not exist", "untrained; no attribution")
    for field in (Assessment().reason, Assessment().detail, Attribution().reason):
        low = field.lower()
        for phrase in forbidden:
            assert phrase not in low, f"{phrase!r} is still shipped in {field!r}"


def test_localisation_and_attribution_are_not_described_as_the_same_thing():
    """They were, and the confusion is one rename away from shipping a false caveat.

    `Attribution` in report.py is a per-region AI/human segmentation that needs a head this
    project never trained. `forge.image.attribution` is occlusion importance over the
    detector that IS loaded, and it travels under `attribution_map`. Saying "no attribution
    map can be produced" next to a rendered attribution map is the failure this guards.
    """
    from forge.image.report import Attribution

    reason = Attribution().reason.lower()
    assert "segmentation" in reason or "localisation" in reason
    assert "occlusion attribution" in reason, (
        "the reason must point at the map that DOES exist, or it reads as a flat denial"
    )


def test_the_page_has_exactly_two_tabs():
    """Text and Image, the two live detectors. The Results tab was removed.

    Measured numbers live in docs/evaluation.md and docs/writeup.md next to the method that
    produced them. A third tab restating them is a second copy to keep in step, and this
    project has already shipped several wrong answers that were exactly that: one place
    describing what another place does.
    """
    body = page()
    # Count data-tab, not class="tab": the wrapper is <div class="tabs"> and matches too.
    assert body.count("data-tab=") == 2, "the tab strip is no longer Text and Image only"
    assert 'data-tab="text"' in body and 'data-tab="image"' in body
    for gone in ('data-tab="results"', "pane-results", "outResults", "loadResults"):
        assert gone not in body, f"{gone} is still in the page"


def test_no_operating_point_sentence_sits_under_the_verdict():
    """It read "Not AI below 0.319, AI at or above 0.319", which contradicts itself.

    When the human ceiling and the confident-AI floor round to the same three decimals, the
    uncertain band is narrower than the rounding and the sentence says a number is both
    below and at-or-above the same value. The figures stay in the payload's `band`, in the
    Details section and in /health.
    """
    from forge.image.report import Assessment, _assessment

    class _Detection:
        ai_probability = 0.07
        threshold_ai = 0.3125
        confident_ai = 0.3187
        polarity_verified = True
        verdict = "human"

    assessment = _assessment([], _Detection())
    assert isinstance(assessment, Assessment)
    assert assessment.verdict == "human"
    assert assessment.reason == "", "a numeric band sentence is back under the headline"
    assert assessment.band, "the operating point must still travel in the payload"
