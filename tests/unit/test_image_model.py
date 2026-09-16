"""The image detector's architecture, loss and decision rule, on CPU with a stub backbone.

What is under test is not whether DINOv3 works. It is the four decisions around it that a
download would not reveal: that the loss is computed at patch resolution rather than pixel
resolution, that a whole-image sample still supervises the local head, that the global head
sees patch evidence rather than only the CLS token, and that the model is allowed to say it
does not know.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is in the [train] extra")

from forge.image.model import (  # noqa: E402
    ForgeImageConfig,
    build_model,
    downsample_mask,
    heatmap,
    verdict,
)

CONFIG = ForgeImageConfig(image_size=64, patch_size=16, hidden_size=32, freeze_backbone=False)


class StubBackbone(torch.nn.Module):
    """Returns (cls, patches) with the right shapes and a little learnable structure."""

    def __init__(self, config: ForgeImageConfig) -> None:
        super().__init__()
        self.config = config
        self.project = torch.nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, pixel_values):
        batch, _, _, _ = pixel_values.shape
        grid, patch = self.config.grid, self.config.patch_size
        tiles = (
            pixel_values.unfold(2, patch, patch)
            .unfold(3, patch, patch)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(batch, grid * grid, -1)
        )
        patches = self.project(tiles)
        return patches.mean(dim=1), patches


def _model():
    return build_model(CONFIG, backbone=StubBackbone(CONFIG))


def _batch(n: int = 2):
    return torch.rand(n, 3, CONFIG.image_size, CONFIG.image_size)


# --- shapes ------------------------------------------------------------------------------


def test_grid_must_tile_the_image() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        _ = ForgeImageConfig(image_size=100, patch_size=16).grid


def test_outputs_have_the_expected_shapes() -> None:
    out = _model()(_batch(3))
    assert out["logit"].shape == (3,)
    assert out["patch_logits"].shape == (3, CONFIG.grid, CONFIG.grid)


# --- decision 1: the loss lives at patch resolution ---------------------------------------


def test_mask_is_downsampled_to_the_patch_grid() -> None:
    mask = torch.zeros(2, CONFIG.image_size, CONFIG.image_size)
    mask[:, : CONFIG.image_size // 2, :] = 1.0
    got = downsample_mask(mask, CONFIG.grid)
    assert got.shape == (2, CONFIG.grid, CONFIG.grid)
    assert got[:, 0, :].mean() > 0.9 and got[:, -1, :].mean() < 0.1


def test_a_partly_covered_patch_reads_as_partly_covered() -> None:
    """Average pooling, not nearest neighbour: half a patch should not round to on or off."""
    mask = torch.zeros(1, CONFIG.image_size, CONFIG.image_size)
    mask[0, :, : CONFIG.patch_size // 2] = 1.0
    got = downsample_mask(mask, CONFIG.grid)
    assert 0.3 < got[0, 0, 0].item() < 0.7


def test_the_local_loss_uses_the_patch_grid_not_the_pixel_grid() -> None:
    """Grading against upsampled logits would score the interpolation, not the prediction."""
    model = _model()
    out = model(_batch(2), mask=torch.ones(2, CONFIG.image_size, CONFIG.image_size))
    assert "local_loss" in out
    assert out["local_loss"].dim() == 0


# --- decision 2: whole-image samples still supervise the local head ------------------------


def test_a_fully_generated_image_supervises_the_local_head() -> None:
    model = _model()
    out = model(
        _batch(2),
        mask=torch.ones(2, CONFIG.image_size, CONFIG.image_size),
        label=torch.ones(2),
    )
    assert out["local_loss"].item() > 0
    assert out["loss"].item() > 0


def test_a_photograph_supervises_the_local_head_too() -> None:
    model = _model()
    out = model(
        _batch(2),
        mask=torch.zeros(2, CONFIG.image_size, CONFIG.image_size),
        label=torch.zeros(2),
    )
    assert "local_loss" in out


def test_loss_combines_both_heads_with_the_configured_weight() -> None:
    model = _model()
    mask = torch.ones(2, CONFIG.image_size, CONFIG.image_size)
    out = model(_batch(2), mask=mask, label=torch.ones(2))
    expected = out["global_loss"] + CONFIG.local_loss_weight * out["local_loss"]
    assert torch.allclose(out["loss"], expected)


def test_inference_needs_no_labels() -> None:
    out = _model()(_batch(1))
    assert "loss" not in out and "logit" in out


# --- decision 3: the global head sees patch evidence ---------------------------------------


def test_the_global_head_reads_cls_and_pooled_patches() -> None:
    """CLS alone summarises the scene, which is what we are trying NOT to classify on."""
    model = _model()
    first = model.global_head[0]
    assert first.in_features == CONFIG.hidden_size * 2


# --- decision 4: the model may abstain ------------------------------------------------------


@pytest.mark.parametrize(
    "probability,expected",
    [(0.99, "ai"), (0.70, "ai"), (0.50, "uncertain"), (0.40, "uncertain"), (0.05, "human")],
)
def test_verdicts(probability: float, expected: str) -> None:
    assert verdict(probability, abstain_below=0.65) == expected


def test_the_abstention_band_is_symmetric() -> None:
    """An asymmetric band quietly defaults to the majority class when unsure."""
    assert verdict(0.66, 0.65) == "ai"
    assert verdict(0.34, 0.65) == "human"
    assert verdict(0.60, 0.65) == verdict(0.40, 0.65) == "uncertain"


@pytest.mark.parametrize("bad", [0.5, 0.2, 1.5])
def test_a_nonsense_abstention_threshold_is_refused(bad: float) -> None:
    with pytest.raises(ValueError):
        verdict(0.9, bad)


# --- training and display -------------------------------------------------------------------


def test_gradients_reach_both_heads() -> None:
    model = _model()
    out = model(
        _batch(2),
        mask=torch.ones(2, CONFIG.image_size, CONFIG.image_size),
        label=torch.ones(2),
    )
    out["loss"].backward()
    assert model.local_head.weight.grad is not None
    assert model.global_head[0].weight.grad is not None


def test_a_frozen_backbone_receives_no_gradients() -> None:
    frozen = build_model(
        ForgeImageConfig(image_size=64, patch_size=16, hidden_size=32, freeze_backbone=True),
        backbone=StubBackbone(CONFIG),
    )
    assert all(not p.requires_grad for p in frozen.backbone.parameters())
    assert any(p.requires_grad for p in frozen.local_head.parameters())


def test_heatmap_upsamples_for_display_only() -> None:
    """Upsampling belongs at display time; the loss never sees it."""
    out = _model()(_batch(1))
    displayed = heatmap(out["patch_logits"], size=CONFIG.image_size)
    assert displayed.shape == (1, CONFIG.image_size, CONFIG.image_size)
    assert float(displayed.min()) >= 0.0 and float(displayed.max()) <= 1.0
