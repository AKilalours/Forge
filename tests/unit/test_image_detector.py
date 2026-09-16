"""Label polarity, and the refusals that keep a wrong verdict off the page.

The bug this file mostly exists for: assuming output index 1 is the AI class. A published
classifier may order its classes either way. Guessing wrong inverts every verdict on the
page while everything looks normal, which is exactly what happened to the MAGE benchmark in
the text track, where a wrong polarity would have turned an AUROC of 0.05 into a reported
0.95.
"""

from __future__ import annotations

import pytest

from forge.image.detector import (
    Detection,
    LabelPolarityUnknown,
    resolve_ai_index,
)


def test_polarity_is_read_from_the_names_either_way_round():
    assert resolve_ai_index({0: "human", 1: "artificial"}) == 1
    assert resolve_ai_index({0: "artificial", 1: "human"}) == 0
    assert resolve_ai_index({"0": "Real", "1": "Fake"}) == 1
    assert resolve_ai_index({0: "AI", 1: "Not AI"}) == 0


def test_not_ai_is_read_before_ai():
    """Substring order matters: 'not ai' contains 'ai'. Longest match wins."""
    assert resolve_ai_index({0: "ai_generated", 1: "not_ai"}) == 0
    assert resolve_ai_index({0: "non-ai", 1: "ai generated"}) == 1


def test_unrecognised_labels_are_refused_rather_than_defaulted():
    """THE ONE THAT MATTERS. No silent fall back to index 1."""
    with pytest.raises(LabelPolarityUnknown) as error:
        resolve_ai_index({0: "class_0", 1: "class_1"})
    assert "class_0" in str(error.value), "the message must name the labels it saw"


def test_two_ai_classes_are_refused_because_the_positive_class_is_ambiguous():
    with pytest.raises(LabelPolarityUnknown):
        resolve_ai_index({0: "midjourney", 1: "stable diffusion"})


def test_all_human_labels_are_refused_because_nothing_means_ai():
    with pytest.raises(LabelPolarityUnknown):
        resolve_ai_index({0: "real", 1: "authentic"})


def test_the_verdict_has_three_states_and_the_middle_one_declines():
    """A detector at a low false-positive budget must be allowed to say "I do not know".

    With no measurement both boundaries default to ABSTAIN_HIGH, which collapses the middle
    band to nothing: an unmeasured detector should not be inventing a zone of uncertainty it
    never estimated.
    """
    def verdict(p, low=0.30, high=0.44):
        return Detection(ai_probability=p, model_id="m",
                         threshold_ai=low, confident_ai=high).verdict

    assert verdict(0.01) == "human"
    assert verdict(0.99) == "ai"
    assert verdict(0.38) == "uncertain"
    assert verdict(0.30) == "uncertain", "the band is inclusive at the low edge"
    assert verdict(0.44) == "ai", "the band is exclusive at the high edge"
    assert verdict(0.30 - 1e-9) == "human"

    default = Detection(ai_probability=0.5, model_id="m")
    assert default.verdict == "human", (
        "without a measurement the two boundaries coincide, so there is no invented band"
    )


def test_a_detection_never_claims_to_be_calibrated():
    """Calibration needs a validation split from a corpus that does not exist yet."""
    assert Detection(ai_probability=0.9, model_id="m").calibrated is False


def test_detector_state_never_raises_when_nothing_can_load(monkeypatch):
    """The page reads this to render system state. It must not be able to take the page down."""
    import forge.image.detector as D

    # Every source of a model id has to be closed, not just the candidate list. The
    # measured model recorded under reports/experiments deliberately OUTRANKS the candidate
    # order (that fix stopped the probe measuring one model while the server served
    # another), so emptying CANDIDATES alone left a working detector and this test asserted
    # the failure path while exercising the success path.
    monkeypatch.setattr(D, "CANDIDATES", ())
    monkeypatch.setattr(D, "measured_model_id", lambda: None)
    monkeypatch.setenv("FORGE_IMAGE_DETECTOR", "")
    D.load_detector.cache_clear()
    state = D.detector_state()
    assert state["available"] is False
    assert state["reason"]
    D.load_detector.cache_clear()


def test_a_named_model_is_not_silently_replaced_by_a_fallback(monkeypatch):
    """If the report names a detector, that detector produced the score."""
    import forge.image.detector as D

    tried = []

    def _fail(model_id):
        tried.append(model_id)
        raise RuntimeError("nope")

    monkeypatch.setattr(D, "_load_one", _fail)
    monkeypatch.setenv("FORGE_IMAGE_DETECTOR", "someone/specific-model")
    D.load_detector.cache_clear()
    with pytest.raises(RuntimeError):
        D.load_detector()
    assert tried == ["someone/specific-model"], f"fell back to {tried}"
    D.load_detector.cache_clear()


def test_a_measurement_outranks_the_model_s_own_label_names(tmp_path, monkeypatch):
    """THE REGRESSION. id2label is documentation, and documentation can be wrong.

    Trusting the names produced "AI DETECTED 99.9%" on a Canon photograph and
    "NO AI DETECTED 1.5%" on a ChatGPT image carrying an IPTC trainedAlgorithmicMedia
    marker. A clean inversion, in the direction that accuses a real photographer.
    """
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({
        "model_id": "some/detector", "verified_ai_index": 0,
        "name_based_ai_index": 1, "inverted_relative_to_labels": True,
    }))
    assert D.verified_polarity("some/detector")["verified_ai_index"] == 0


def test_a_measurement_for_a_different_model_does_not_transfer(tmp_path, monkeypatch):
    """Polarity is a property of one model. Reusing another's is how a fix becomes a bug."""
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"model_id": "other/model", "verified_ai_index": 0}))
    assert D.verified_polarity("some/detector") is None


def test_no_record_means_unverified_rather_than_assumed_correct(tmp_path, monkeypatch):
    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    assert D.verified_polarity("some/detector") is None


def test_an_unreadable_record_is_treated_as_absent(tmp_path, monkeypatch):
    """A corrupt file must not be read as a confirmation."""
    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text("{not json")
    assert D.verified_polarity("some/detector") is None


def test_an_unverified_detector_says_so_in_the_evidence_panel():
    """The direction of the number is a claim until it is measured. Say which."""
    from forge.image.evidence import build_evidence

    streams = build_evidence([], detector_available=True, probability=0.99,
                             model_id="m", polarity_verified=False).streams
    visual = next(s for s in streams if s.key == "visual_model")
    assert "POLARITY UNVERIFIED" in visual.note

    verified = build_evidence([], detector_available=True, probability=0.99,
                              model_id="m", polarity_verified=True).streams
    assert "POLARITY UNVERIFIED" not in next(
        s for s in verified if s.key == "visual_model").note


def test_the_visual_stream_never_claims_calibration_it_does_not_have():
    """It said "calibrated on validation data" for a detector with no validation split."""
    from forge.image.evidence import build_evidence

    streams = build_evidence([], detector_available=True, probability=0.7,
                             model_id="m", polarity_verified=True).streams
    note = next(s for s in streams if s.key == "visual_model").note
    assert "calibrated on validation data" not in note
    assert "uncalibrated" in note


def test_the_verdict_uses_a_fitted_threshold_when_one_was_measured():
    """The default band is a placeholder. A measured operating point replaces it."""
    fitted = Detection(ai_probability=0.55, model_id="m", threshold_ai=0.32, confident_ai=0.44)
    assert fitted.verdict == "ai", "0.55 is above a fitted AI boundary of 0.44"

    default = Detection(ai_probability=0.20, model_id="m")
    assert default.verdict == "human", "below the documented default boundary"


def test_the_middle_band_declines_rather_than_guessing():
    d = Detection(ai_probability=0.38, model_id="m", threshold_ai=0.32, confident_ai=0.44)
    assert d.verdict == "uncertain"


def test_the_ai_boundary_is_never_below_the_not_ai_boundary():
    """Inverting them would make one of the three verdicts unreachable."""
    d = Detection(ai_probability=0.0, model_id="m", threshold_ai=0.32, confident_ai=0.44)
    assert d.threshold_ai <= d.confident_ai


def test_the_photograph_that_set_the_threshold_reads_as_not_ai():
    """THE REGRESSION. The probe threshold was max(human) + 1e-6, so the image that defined
    it landed exactly on the boundary and a real photograph was reported "uncertain"."""
    highest_human = 0.3125
    margin = max(0.005, 0.02 * highest_human)
    d = Detection(
        ai_probability=highest_human, model_id="m",
        threshold_ai=highest_human + margin, confident_ai=0.4366,
    )
    assert d.verdict == "human", (
        "the photograph that set the operating point must fall under it, not on it"
    )


def test_a_generated_image_at_the_group_median_reads_as_ai():
    d = Detection(ai_probability=0.4366, model_id="m", threshold_ai=0.3187, confident_ai=0.4366)
    assert d.verdict == "ai"


def test_the_measured_detector_is_the_one_that_loads(tmp_path, monkeypatch):
    """THE REGRESSION. A probe measured one model while the server served another.

    The record named umm-maybe; the loader walked its candidate list and got Organika,
    which had already FAILED the probe. The record did not match, was correctly ignored,
    and the page reported "polarity not verified" with a finished measurement on disk.
    """
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_IMAGE_DETECTOR", raising=False)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"model_id": "measured/model", "verified_ai_index": 1}))
    assert D.measured_model_id() == "measured/model"

    loaded = []
    monkeypatch.setattr(D, "_load_one", lambda m: loaded.append(m) or "detector")
    D.load_detector.cache_clear()
    D.load_detector()
    assert loaded == ["measured/model"], f"loaded {loaded} instead of the measured model"
    D.load_detector.cache_clear()


def test_an_explicit_model_still_outranks_the_measured_one(tmp_path, monkeypatch):
    """Someone naming a model is probing it. Overriding that would defeat the probe."""
    import json

    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    record = tmp_path / D.POLARITY_RECORD
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"model_id": "measured/model", "verified_ai_index": 1}))
    monkeypatch.setenv("FORGE_IMAGE_DETECTOR", "explicit/model")

    loaded = []
    monkeypatch.setattr(D, "_load_one", lambda m: loaded.append(m) or "detector")
    D.load_detector.cache_clear()
    D.load_detector()
    assert loaded == ["explicit/model"]
    D.load_detector.cache_clear()


def test_without_a_record_the_candidate_order_still_applies(tmp_path, monkeypatch):
    import forge.image.detector as D

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FORGE_IMAGE_DETECTOR", raising=False)
    assert D.measured_model_id() is None


def _png_with(text_chunks: dict) -> bytes:
    """A PNG carrying tEXt metadata, the way a generator writes its marker."""
    import io

    import pytest as _pytest

    _pytest.importorskip("PIL")
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    for key, value in text_chunks.items():
        info.add_text(key, value)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (90, 120, 60)).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def test_a_declared_image_is_called_ai_even_when_the_detector_disagrees():
    """THE REGRESSION. A ChatGPT image declared itself generated and the page said
    NO AI DETECTED at 1.5%, because a baseline detector was allowed to overrule a label
    the generator wrote about its own output."""
    from forge.image.report import build_report

    data = _png_with({"parameters": "a photo of a cat", "sampler": "Euler a"})
    report = build_report(data, with_detector=False)
    assert report.assessment.available is True
    assert report.assessment.verdict == "ai"
    assert "declared" in report.assessment.reason.lower()


def test_a_declaration_does_not_need_a_detector_at_all():
    """It is a label, not an inference. It stands alone."""
    from forge.image.report import build_report

    report = build_report(_png_with({"parameters": "x", "cfg scale": "7"}), with_detector=False)
    assert report.assessment.verdict == "ai"
    assert report.detector.get("available") is not True


def test_absence_of_a_declaration_never_produces_a_human_verdict():
    """THE ASYMMETRY. Most generators write no marker and every editor strips metadata.

    Reading absence as evidence of human authorship is the same false-positive engine this
    project exists to avoid, pointed the other way.
    """
    from forge.image.report import build_report

    report = build_report(_png_with({"Comment": "holiday photo"}), with_detector=False)
    assert report.assessment.available is False, "absence of a marker decided a verdict"
    assert report.assessment.verdict is None


def test_the_filename_plays_no_part_in_the_verdict():
    """Renaming a file cannot change the answer. Pixels and bytes only."""
    from forge.image.report import build_report

    data = _png_with({"parameters": "x", "sampler": "Euler"})
    honest = build_report(data, filename="ChatGPT Image.png", with_detector=False)
    renamed = build_report(data, filename="IMG_0042.JPG", with_detector=False)
    assert honest.assessment.verdict == renamed.assessment.verdict == "ai"

    plain = _png_with({"Comment": "x"})
    a = build_report(plain, filename="ai_generated_fake.png", with_detector=False)
    b = build_report(plain, filename="my_holiday.png", with_detector=False)
    assert a.assessment.verdict == b.assessment.verdict is None


def test_the_polarity_record_says_what_it_was_measured_on():
    """An operating point nobody can trace is not a measurement, it is a number.

    The first probe set lived in /tmp and was lost to a reboot, and the committed record
    could not say which images produced its threshold: only that there were 20 and 9. This
    project never redistributes image pixels, so the record carries file NAMES, sizes and
    directories, which are metadata and traceable, rather than the images themselves.

    Skips when no record is committed yet, since the record is a measurement artefact and
    not every checkout has one.
    """
    import json
    import pathlib

    path = pathlib.Path("reports/experiments/image_detector_polarity.json")
    if not path.exists():
        pytest.skip("no polarity record committed")

    record = json.loads(path.read_text(encoding="utf-8"))
    measured = record.get("measured_on")
    if measured is None:
        pytest.skip("record predates provenance; re-run scripts/image_detector_probe.py")

    for field in ("ai_dir", "human_dir", "ai_files", "human_files", "measured_at"):
        assert measured.get(field), f"{field} missing from the record's provenance"
    assert len(measured["ai_files"]) == record["n"]["ai"]
    assert len(measured["human_files"]) == record["n"]["human"], (
        "the recorded file list must be the set the numbers came from, not the directory"
    )
