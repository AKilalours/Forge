"""Where the detector's decision comes from, measured by occlusion. CPU, one forward pass.

WHY NOT THE FORENSIC MAPS. Compression residual, noise floor and local contrast are pixel
statistics. They describe the file and they know nothing about the model, so putting them
where an attribution map belongs invites the reader to treat "bright here" as "the detector
thinks this region is generated". It never meant that.

WHY OCCLUSION RATHER THAN GRAD-CAM. Grad-CAM and attention rollout need to know where a
particular architecture keeps its spatial tokens, so they break silently when the detector
is swapped, and this project swaps detectors. Occlusion asks the model a question it cannot
misunderstand: hide this region, does the AI probability fall? A region whose removal drops
the score is a region the decision rested on. It works on any classifier, it is exact rather
than an approximation of a gradient, and it is trivially explainable to a reader.

WHY IT IS FAST ENOUGH ON A CPU. The occluded variants are built in the model's own
preprocessed tensor space, after the single expensive resize, and scored in ONE batched
forward pass rather than one call per tile. A 5x5 grid is 26 images through a ViT at 224
pixels, which is a couple of seconds, against roughly half a minute for the naive loop.

WHAT THE VALUES MEAN. Each cell is `P(AI) with the region visible` minus `P(AI) with it
hidden`, so a positive cell is evidence the model used FOR its AI call and a negative cell
is evidence against. Occlusion replaces a region with the dataset mean colour, which is
itself an unusual thing to show a model, so a cell says "this region mattered", not "this
region is generated".
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

GRID = 5              # 25 occlusions plus the original: one batch, a couple of CPU seconds
MAX_BATCH = 32


@dataclass
class Attribution:
    grid: int
    base_probability: float
    cells: list[list[float]]        # P(AI) with the region visible minus without it
    png_data_uri: str
    peak: dict
    method: str = "occlusion"

    def as_dict(self) -> dict:
        return {
            "grid": self.grid,
            "base_probability": round(self.base_probability, 6),
            "cells": [[round(v, 6) for v in row] for row in self.cells],
            "image": self.png_data_uri,
            "peak": self.peak,
            "method": self.method,
            "reading": (
                "Each cell is the drop in AI probability when that region is hidden. A warm "
                "cell is a region the decision rested on. It marks what the model used, not "
                "what is generated."
            ),
        }


def _overlay(preview_rgb, cells) -> str:
    """The map drawn over a dimmed copy of the image, so regions can be located by eye.

    A bare grid of colour is unreadable without the picture underneath: a reader cannot tell
    which part of the frame a hot cell refers to, which is most of what makes a heatmap
    worth showing at all.
    """
    import numpy as np
    from PIL import Image

    a = np.asarray(cells, dtype=np.float64)
    scale = float(np.max(np.abs(a)))
    norm = a / scale if scale > 1e-9 else np.zeros_like(a)

    # Diverging: cool where hiding the region RAISED the score, warm where it dropped it.
    warm = np.clip(norm, 0, 1)
    cool = np.clip(-norm, 0, 1)
    rgb = np.dstack([
        0.16 + 0.84 * warm,
        0.18 + 0.34 * warm + 0.30 * cool,
        0.30 + 0.70 * cool - 0.20 * warm,
    ])
    heat = Image.fromarray(np.clip(rgb * 255, 0, 255).astype("uint8"), "RGB")
    heat = heat.resize(preview_rgb.size, Image.Resampling.BICUBIC)

    dimmed = Image.eval(preview_rgb, lambda v: int(v * 0.45))
    blended = Image.blend(dimmed, heat, 0.55)

    buffer = io.BytesIO()
    blended.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def occlusion_attribution(detector, data: bytes, grid: int = GRID) -> Attribution | None:
    """Score the image and `grid * grid` copies with one region hidden, in one batch.

    Returns None rather than raising: an explanation is worth having and never worth taking
    the report down for.
    """
    try:
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            rgb = img.convert("RGB")
            inputs = detector.processor(images=rgb, return_tensors="pt")
            preview = rgb.copy()
            preview.thumbnail((384, 384))

        pixel = inputs["pixel_values"]
        if pixel.ndim != 4:
            return None
        _, _, height, width = pixel.shape
        cell_h, cell_w = height // grid, width // grid
        if cell_h < 1 or cell_w < 1:
            return None

        variants = [pixel]
        for row in range(grid):
            for col in range(grid):
                hidden = pixel.clone()
                # Zero in the model's normalized space is the dataset mean colour, which is
                # the least informative thing that can occupy the region.
                hidden[:, :, row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = 0.0
                variants.append(hidden)

        probabilities = []
        with torch.no_grad():
            for start in range(0, len(variants), MAX_BATCH):
                batch = torch.cat(variants[start:start + MAX_BATCH], dim=0)
                logits = detector.model(pixel_values=batch).logits
                probabilities.append(torch.softmax(logits.float(), dim=-1)[:, detector.ai_index])
        scores = torch.cat(probabilities).numpy()

        base = float(scores[0])
        cells = (base - scores[1:]).reshape(grid, grid)
        row, col = (
            int(value) for value in np.unravel_index(int(np.argmax(cells)), cells.shape)
        )
        return Attribution(
            grid=grid,
            base_probability=base,
            cells=cells.tolist(),
            png_data_uri=_overlay(preview, cells),
            peak={
                "row": int(row), "col": int(col),
                "x": round((col + 0.5) / grid, 4), "y": round((row + 0.5) / grid, 4),
                "drop": round(float(cells[row, col]), 6),
            },
        )
    except Exception:  # noqa: BLE001 - a missing explanation is not a failed report
        return None
