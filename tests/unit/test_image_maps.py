"""The forensic maps must respond to what their captions claim they respond to.

A heatmap is trusted on sight. Testing that it renders is not enough: it has to be
demonstrably brighter or darker in the region a reader is told to look at. These tests build
frames with a known planted region and assert the map notices it.
"""

from __future__ import annotations

import io

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from forge.image.maps import (  # noqa: E402
    _to_png_uri,
    build_maps,
    detail_grid,
    noise_grid,
)

PATCH = (slice(80, 176), slice(80, 176))     # the planted region, in pixels


def _frame(flatten_patch: bool, size: int = 256, seed: int = 0) -> bytes:
    """Uniform noise, optionally with one flat square pasted in."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(128, 30, (size, size, 3)).clip(0, 255).astype("uint8")
    if flatten_patch:
        arr[PATCH] = 128
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")   # lossless: isolate the planted effect
    return buffer.getvalue()


def _patch_cells(grid):
    """The grid cells covering the planted patch, and everything else."""
    rows, cols = grid.shape
    r0, r1 = int(rows * 80 / 256), int(np.ceil(rows * 176 / 256))
    c0, c1 = int(cols * 80 / 256), int(np.ceil(cols * 176 / 256))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[r0:r1, c0:c1] = True
    # Shrink by one cell so boundary cells, which straddle both regions, are excluded.
    inner = np.zeros_like(mask)
    inner[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = True
    return grid[inner], grid[~mask]


def test_the_noise_map_darkens_where_the_noise_floor_was_removed():
    """THE CLAIM ON THE PANEL: flat dark regions mean the noise floor is gone there."""
    grid = noise_grid(_frame(flatten_patch=True))
    planted, rest = _patch_cells(grid)
    assert planted.size and rest.size
    assert planted.max() < rest.min(), (
        f"planted flat region max {planted.max():.2f} should sit below the rest's "
        f"min {rest.min():.2f}"
    )


def test_the_noise_map_is_flat_when_nothing_was_planted():
    """The accepting side. Without a patch the map must not invent a hot region."""
    grid = noise_grid(_frame(flatten_patch=False))
    assert float(grid.std() / grid.mean()) < 0.25


def test_the_detail_map_darkens_in_a_locally_smooth_region():
    grid = detail_grid(_frame(flatten_patch=True))
    planted, rest = _patch_cells(grid)
    assert planted.max() < rest.min()


def test_normalization_is_within_the_image_so_a_flat_frame_is_not_all_hot():
    """A constant plane has no strongest region, and must not render as a peak everywhere."""
    uri = _to_png_uri(np.zeros((16, 16)))
    assert uri.startswith("data:image/png;base64,")
    flat = _to_png_uri(np.full((16, 16), 7.0))
    assert flat == uri, "a constant plane must render identically regardless of its value"


def test_a_map_is_omitted_rather_than_faked_when_it_cannot_be_computed():
    assert build_maps(b"not an image at all") == []
    tiny = io.BytesIO()
    Image.new("RGB", (8, 8)).save(tiny, format="PNG")
    names = {m["name"] for m in build_maps(tiny.getvalue())}
    assert "noise" not in names and "detail" not in names, (
        "an 8-pixel image cannot support a cell grid; those maps must be absent"
    )


def test_every_map_carries_its_caveat_and_a_peak():
    maps = build_maps(_frame(flatten_patch=True))
    assert maps
    for m in maps:
        assert m["caveat"], f"{m['name']} has no caveat"
        assert m["what_it_shows"]
        assert set(m["peak_location"]) == {"x", "y"}
        assert m["image"].startswith("data:image/png;base64,")


def test_the_maps_are_not_described_as_model_output():
    """Guards the wording. These are residual maps, not detector saliency."""
    for m in build_maps(_frame(flatten_patch=True)):
        blob = (m["title"] + m["what_it_shows"] + m["caveat"]).lower()
        for forbidden in ("saliency", "the detector thinks", "model believes", "confidence"):
            assert forbidden not in blob, f"{m['name']} claims model output: {forbidden}"


def test_stability_absent_is_reported_as_absent_not_as_nothing_survived():
    """Opt-in must not read as a measurement.

    The stability pass re-encodes the image ten times and is most of the analysis time, so
    it is off by default. An empty list with `stability_available` still True would render
    as "no signal survived any transform", which is the opposite of "not measured".
    """
    from forge.image.report import build_report

    frame = _frame(flatten_patch=False)
    off = build_report(frame, with_stability=False)
    assert off.stability == [] and off.stability_available is False

    on = build_report(frame, with_stability=True)
    assert on.stability and on.stability_available is True


def test_skipping_stability_does_not_change_any_other_finding():
    """The gate must remove work, not alter conclusions."""
    from forge.image.report import build_report

    frame = _frame(flatten_patch=True)
    off, on = build_report(frame, with_stability=False), build_report(frame, with_stability=True)
    assert [f.as_dict() for f in off.findings] == [f.as_dict() for f in on.findings]
    assert off.evidence.as_dict() == on.evidence.as_dict()
    assert off.authenticity == on.authenticity
    assert off.manipulation == on.manipulation


def test_every_stage_reports_its_own_cost():
    """"53 seconds" is not actionable. WHICH stage is.

    Without a per-stage breakdown the only way to find the expensive part is to guess, and
    guessing is how people optimise the stage that was already fast.
    """
    from forge.image.report import build_report

    frame = _frame(flatten_patch=False)
    r = build_report(frame, with_stability=True)
    for stage in ("forensics", "perceptual_hash", "evidence", "transform_stability", "detector"):
        assert stage in r.timings_ms, f"{stage} reports no cost"
        assert isinstance(r.timings_ms[stage], int)
    # This asserted a flat zero on the reasoning that no image model exists. One does now,
    # and on a machine with it cached the stage costs real milliseconds. Pinning zero made
    # the test pass only where the detector was missing and fail on a working build. What
    # must hold is that the number reflects whether the detector ran.
    from forge.image.detector import detector_state

    if detector_state().get("available"):
        assert r.timings_ms["detector"] > 0, "the detector ran; its cost cannot read zero"
    else:
        assert r.timings_ms["detector"] == 0, "no detector ran; the cost must read zero"
    assert r.as_dict()["timings_ms"] == r.timings_ms


def test_the_stage_costs_are_consistent_with_the_total():
    """A breakdown that does not add up is worse than no breakdown."""
    from forge.image.report import build_report

    r = build_report(_frame(flatten_patch=False), with_stability=True)
    measured = sum(r.timings_ms.values())
    assert measured <= r.elapsed_ms + 5, (
        f"stages sum to {measured} ms against a total of {r.elapsed_ms} ms"
    )


def test_a_skipped_stage_reports_no_cost_rather_than_zero_cost():
    """Absent, not free. A 0 ms transform-stability row would read as "instant"."""
    from forge.image.report import build_report

    off = build_report(_frame(flatten_patch=False), with_stability=False)
    assert "transform_stability" not in off.timings_ms


def _mpo(frames: int = 2) -> bytes:
    """An MPO: a JPEG container with more than one frame, as dual-lens phones write."""
    base = Image.new("RGB", (96, 72), (120, 90, 60))
    extra = [Image.new("RGB", (96, 72), (60, 90, 120)) for _ in range(frames - 1)]
    buffer = io.BytesIO()
    base.save(buffer, format="MPO", save_all=True, append_images=extra)
    return buffer.getvalue()


def test_a_phone_photograph_in_an_mpo_container_is_analysed_not_rejected():
    """THE REGRESSION. MPO was refused by an allowlist, not by anything measured.

    Dual-lens phones and depth captures write MPO, which is a JPEG holding several frames.
    Turning those away rejects ordinary photographs, which are exactly the files this
    project exists to protect from a false accusation.
    """
    from api.forge_app import ACCEPTED
    from forge.image.forensics import real_format

    finding = real_format(_mpo())
    assert finding.value == "MPO"
    assert finding.value in ACCEPTED, "an ordinary phone photograph is refused"


def test_a_multi_frame_container_says_only_the_first_frame_was_analysed():
    """A signal measured on frame 0 is not a statement about frames 1..n."""
    from forge.image.forensics import real_format

    finding = real_format(_mpo(frames=3))
    assert finding.detail["frames"] == 3
    assert "only the first is analysed" in finding.caveat


def test_a_single_frame_file_does_not_carry_the_multi_frame_caveat():
    """The accepting side: the caveat must fire on multi-frame files and nowhere else."""
    from forge.image.forensics import real_format

    finding = real_format(_frame(flatten_patch=False))
    assert finding.detail["frames"] == 1
    assert "only the first is analysed" not in finding.caveat


def test_an_mpo_produces_a_full_report_rather_than_a_degraded_one():
    from forge.image.report import build_report

    report = build_report(_mpo(), with_stability=False)
    assert len(report.findings) >= 10
    assert report.by_name("file_type").value == "MPO"


def test_a_format_needing_a_decoder_is_named_rather_than_called_unreadable():
    """"Unreadable" sends a user nowhere. HEIC is readable, this build just cannot."""
    from api.forge_app import NEEDS_DECODER

    assert "HEIC" in NEEDS_DECODER
    assert "pillow-heif" in NEEDS_DECODER["HEIC"]


class _Processor:
    def __call__(self, images, return_tensors="pt"):
        import numpy as np
        import torch

        a = np.asarray(images.resize((32, 32)), dtype="float32") / 255.0
        return {"pixel_values": torch.tensor(a).permute(2, 0, 1).unsqueeze(0)}


class _Model:
    """Only the top-left quadrant moves this model's output."""

    def __call__(self, pixel_values):
        import torch

        corner = pixel_values[:, :, :12, :12].mean(dim=(1, 2, 3))
        logits = torch.stack([corner * 12.0, torch.zeros_like(corner)], dim=-1)
        return type("O", (), {"logits": logits})()


class _FakeDetector:
    """A detector with a planted dependency on one region.

    Hiding the top-left quadrant should collapse the score; hiding anything else should not.
    A map that cannot recover a dependency this obvious recovers nothing.
    """

    ai_index = 0
    processor = _Processor()
    model = _Model()


def test_occlusion_finds_the_region_the_decision_rests_on():
    pytest.importorskip("torch")
    """THE POINT OF THE PANEL. A map that cannot find a planted dependency is decoration."""
    from forge.image.attribution import occlusion_attribution

    built = occlusion_attribution(_FakeDetector(), _frame(flatten_patch=False, size=256), grid=4)
    assert built is not None
    assert built.peak["row"] == 0 and built.peak["col"] == 0, (
        f"the map points at {built.peak}, not the top-left region the model actually uses"
    )
    assert built.peak["drop"] > 0


def test_the_map_reports_the_score_it_started_from():
    pytest.importorskip("torch")
    from forge.image.attribution import occlusion_attribution

    built = occlusion_attribution(_FakeDetector(), _frame(flatten_patch=False, size=256), grid=3)
    assert 0.0 <= built.base_probability <= 1.0
    assert len(built.cells) == 3 and len(built.cells[0]) == 3
    assert built.png_data_uri.startswith("data:image/png;base64,")


def test_a_detector_that_raises_gives_no_map_rather_than_no_report():
    pytest.importorskip("torch")
    from forge.image.attribution import occlusion_attribution

    class Broken:
        ai_index = 0

        def processor(self, **_):
            raise RuntimeError("no")

    assert occlusion_attribution(Broken(), _frame(flatten_patch=False)) is None
