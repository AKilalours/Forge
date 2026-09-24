"""Which detector the app serves, and why the weakest one was winning.

A ChatGPT image scored 2.1% and the page reported no AI. The score was honest and the
detector was not the problem on its own: umm-maybe/AI-image-detector measures 0.55 recall
at its operating point and is a 2023 model. The problem was that it was PINNED.

`load_detector` prefers a measured model over the candidate order, which is right. But
there was one record file for the whole repository, so "the measured model" meant
"whichever model was probed last", and Organika/sdxl-detector sat first in CANDIDATES and
was never loaded. Measuring a worse detector made the app serve a worse detector.
"""

from __future__ import annotations

import json

import pytest

from forge.image import detector as det


def _record(tmp_path, model_id, recall, n_ai=20, n_human=9):
    path = tmp_path / det.record_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model_id": model_id, "verified_ai_index": 0,
        "ai_recall_at_threshold": recall, "n": {"ai": n_ai, "human": n_human},
    }))
    return path


@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_best_measured_detector_wins_not_the_last_one_probed(at):
    _record(at, "umm-maybe/AI-image-detector", 0.55)
    _record(at, "Organika/sdxl-detector", 0.92)

    assert det.measured_model_id() == "Organika/sdxl-detector"


def test_a_larger_sample_breaks_a_tie(at):
    # Equal recall on 29 images and on 400 is not equal evidence.
    _record(at, "a/small", 0.90, n_ai=20, n_human=9)
    _record(at, "b/large", 0.90, n_ai=200, n_human=200)

    assert det.measured_model_id() == "b/large"


def test_one_record_per_model_so_probing_a_second_does_not_erase_the_first(at):
    _record(at, "umm-maybe/AI-image-detector", 0.55)
    _record(at, "Organika/sdxl-detector", 0.92)

    assert {r["model_id"] for r in det.polarity_records()} == {
        "umm-maybe/AI-image-detector", "Organika/sdxl-detector"}


def test_a_measurement_never_transfers_between_models(at):
    _record(at, "Organika/sdxl-detector", 0.92)

    assert det.verified_polarity("Organika/sdxl-detector") is not None
    assert det.verified_polarity("umm-maybe/AI-image-detector") is None


def test_no_measurements_means_no_pinned_model(at):
    # With nothing measured the candidate order decides, which is the documented fallback.
    assert det.measured_model_id() is None
