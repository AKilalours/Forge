"""Images that are part real and part generated, with a known mask.

WHY. The localisation head answers "where is the evidence", and that question cannot be
trained or scored without ground truth. Hand-labelling is not an option at this scale, and
eyeballing a heatmap is not a metric. Constructing the composite ourselves gives a mask
that is exact by definition.

THE TRAP, and it is the image analogue of the JPEG confound.

Paste a generated patch into a photograph and you have introduced something far more
learnable than any generation artifact: a seam. A hard rectangular boundary between two
image statistics is trivially detectable, and a model trained on such composites will find
the seam, score beautifully on IoU, and locate nothing at all in a real AI-edited image
whose edit was blended.

Three defences, all applied here:

1. SHAPE AND FEATHER VARY. Rectangles, ellipses, and soft-edged blends, so no single edge
   profile is the signal.

2. HUMAN-HUMAN COMPOSITES ARE BUILT TOO. A region taken from ANOTHER photograph is pasted
   with exactly the same machinery. Those carry the same seam and are labelled as having no
   AI region at all. A model that keys on the seam cannot separate them from the real
   composites, so seam-learning stops paying.

3. THE WHOLE COMPOSITE IS RENORMALISED. One JPEG pass over the finished image, so the
   pasted region does not carry its own compression history.

Defence 2 is the important one. It is the same instrument as the negative control: build the
shortcut into BOTH classes and it stops being a shortcut.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from forge.common.splits import assign_split
from forge.image.normalize import POLICY_V1, NormalizationPolicy, normalize_bytes

COMPOSITE_VERSION = "image_composite_v1"

SHAPES = ("rectangle", "ellipse")
# Feather radius in pixels. Zero is included deliberately: a model must cope with hard
# edges too, since real edits are sometimes crude.
FEATHERS = (0, 3, 9, 18)
# Fraction of the image area the pasted region covers.
AREA_FRACTIONS = (0.08, 0.15, 0.25, 0.40)


@dataclass(frozen=True)
class Composite:
    sample_id: str
    source_image_id: str
    source_group_id: str
    split: str
    kind: str                # "ai_region" or "human_region", the control
    label: int               # 1 if any region is AI, else 0
    shape: str
    feather: int
    area_fraction: float
    box: tuple[int, int, int, int]
    norm_policy: str
    version: str = COMPOSITE_VERSION
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _pick(sequence, key: str):
    digest = hashlib.sha256(f"composite-v1:{key}".encode()).digest()
    return sequence[int.from_bytes(digest[:4], "big") % len(sequence)]


def _geometry(image_id: str, width: int, height: int) -> tuple[str, int, float, tuple[int, int, int, int]]:
    shape = _pick(SHAPES, f"shape:{image_id}")
    feather = _pick(FEATHERS, f"feather:{image_id}")
    fraction = _pick(AREA_FRACTIONS, f"area:{image_id}")

    side = max(int((width * height * fraction) ** 0.5), 16)
    side = min(side, width - 2, height - 2)
    digest = hashlib.sha256(f"composite-v1:pos:{image_id}".encode()).digest()
    left = int.from_bytes(digest[:2], "big") % max(width - side, 1)
    top = int.from_bytes(digest[2:4], "big") % max(height - side, 1)
    return shape, feather, fraction, (left, top, left + side, top + side)


def _mask(size: tuple[int, int], box, shape: str, feather: int):
    from PIL import Image, ImageDraw, ImageFilter

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if shape == "ellipse":
        draw.ellipse(box, fill=255)
    else:
        draw.rectangle(box, fill=255)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def build_composite(
    base: bytes,
    patch_source: bytes,
    image_id: str,
    kind: str,
    policy: NormalizationPolicy = POLICY_V1,
) -> tuple[bytes, bytes, Composite]:
    """Return (composite image bytes, mask PNG bytes, record).

    `patch_source` supplies the pasted pixels: a generated image for kind="ai_region", and
    ANOTHER photograph for kind="human_region". The machinery is identical on purpose, so
    the seam cannot distinguish them.
    """
    if kind not in ("ai_region", "human_region"):
        raise ValueError(f"unknown composite kind {kind!r}")

    from PIL import Image

    with Image.open(io.BytesIO(base)) as base_img, Image.open(io.BytesIO(patch_source)) as patch_img:
        base_img = base_img.convert("RGB")
        patch_img = patch_img.convert("RGB").resize(base_img.size)

        shape, feather, fraction, box = _geometry(image_id, *base_img.size)
        mask = _mask(base_img.size, box, shape, feather)
        blended = Image.composite(patch_img, base_img, mask)

        out = io.BytesIO()
        blended.save(out, format="PNG")
        mask_bytes = io.BytesIO()
        # The mask is stored UNFEATHERED for scoring. A soft mask would make IoU depend on
        # an arbitrary threshold, and the question being scored is which pixels came from
        # the patch, which is a hard fact.
        _mask(base_img.size, box, shape, 0).save(mask_bytes, format="PNG")

    composite = normalize_bytes(out.getvalue(), policy, allow_upscale=True)
    record = Composite(
        sample_id=f"comp_{kind}_{image_id}",
        source_image_id=image_id,
        # Inherited: a composite shares its base photograph's group, so a photograph and
        # every composite built on it land in the same split.
        source_group_id=image_id,
        split=assign_split(image_id).value,
        kind=kind,
        label=1 if kind == "ai_region" else 0,
        shape=shape,
        feather=feather,
        area_fraction=fraction,
        box=box,
        norm_policy=policy.version,
    )
    return composite, mask_bytes.getvalue(), record


def build_pairs(
    photographs: Sequence[tuple[str, bytes]],
    generated: Sequence[bytes],
    policy: NormalizationPolicy = POLICY_V1,
) -> tuple[list[tuple[bytes, bytes, Composite]], dict]:
    """Build one AI-region composite and one human-region control per photograph.

    Equal numbers by construction. If the control class were smaller, the seam would
    correlate with the label again, just more weakly, and a weak shortcut is harder to
    notice than a strong one.
    """
    if not generated:
        raise ValueError("no generated images supplied; the AI regions have nothing to paste")
    if len(photographs) < 2:
        raise ValueError("need at least two photographs so a control patch can come from another")

    out: list[tuple[bytes, bytes, Composite]] = []
    for index, (image_id, raw) in enumerate(photographs):
        out.append(
            build_composite(raw, generated[index % len(generated)], image_id, "ai_region", policy)
        )
        # The control patch comes from the NEXT photograph, never from itself, or the
        # composite would be indistinguishable from the original and teach nothing.
        other = photographs[(index + 1) % len(photographs)][1]
        out.append(build_composite(raw, other, image_id, "human_region", policy))

    labels = [record.label for _, _, record in out]
    return out, {
        "composites": len(out),
        "ai_region": sum(labels),
        "human_region": len(labels) - sum(labels),
        "version": COMPOSITE_VERSION,
    }


def mask_area_fraction(mask_png: bytes) -> float:
    """Fraction of pixels marked as patch. Used to sanity-check a built dataset."""
    from PIL import Image

    with Image.open(io.BytesIO(mask_png)) as mask:
        grey = mask.convert("L")
        # get_flattened_data replaces getdata in Pillow 14. Read through whichever exists,
        # so a library upgrade cannot change what a stored mask means.
        reader = getattr(grey, "get_flattened_data", None) or grey.getdata
        pixels = list(reader())
    return sum(1 for value in pixels if value > 127) / len(pixels)
