"""Transformations an AI image picks up on its way to being posted, and to being judged.

WHY THIS EXISTS. A detector that works on pristine generator output and fails on a
screenshot of it is not a detector, because nobody uploads pristine generator output. The
adversarial suite measures the gap.

THE ORDERING TRAP, which is the whole reason this module has a long docstring.

FORGE normalises every image before the model sees it: resize, centre crop, re-encode as
JPEG at a fixed quality, strip metadata. That normalisation is what stops the detector
learning that photographs are JPEG and generations are PNG.

But normalisation ALSO destroys most of what an attack does. Run an attack after it and
"JPEG quality 40" becomes "JPEG quality 40 then re-encoded at 90", which measures almost
nothing; "metadata strip" becomes a no-op, because normalisation already stripped it.

So the order is fixed and non-negotiable:

    generator output -> ATTACK -> normalisation -> model

which is the order a real upload experiences: someone screenshots an image, then a website
resizes and re-encodes it, then a detector reads it. Every function here returns raw bytes
intended to be passed through forge.image.normalize afterwards, never instead.

A test in this module's suite asserts that at least one attack still changes the normalised
result. If normalisation flattened every attack, the robustness table would be a column of
zeros that looked like excellent robustness.

WHAT COUNTS AS AN ATTACK. Only transformations that leave the image recognisably the same
picture. Destroying an image until a detector fails is not an attack, it is vandalism, and
it tells you nothing about robustness. Each function documents what it preserves.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass

ATTACK_VERSION = "image_attacks_v1"


def _open(raw: bytes):
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _to_bytes(img, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def jpeg(raw: bytes, quality: int = 50) -> bytes:
    """Re-encode as JPEG. The single most common thing that happens to any image online.

    Preserves: composition, colour, subject. Removes: high-frequency detail, which is
    where several generation artifacts live, so this is the attack most likely to hurt.
    """
    return _to_bytes(_open(raw), "JPEG", quality=quality, subsampling=2)


def resize(raw: bytes, scale: float = 0.5) -> bytes:
    """Downscale then leave it downscaled. Resampling smooths generator fingerprints."""
    from PIL import Image

    img = _open(raw)
    size = (max(int(img.width * scale), 32), max(int(img.height * scale), 32))
    return _to_bytes(img.resize(size, Image.LANCZOS))


def crop(raw: bytes, keep: float = 0.8) -> bytes:
    """Centre crop, keeping `keep` of each dimension.

    Preserves the subject. Removes borders, which is where some generators leave the most
    consistent artifacts, and changes the aspect ratio a normaliser will later square off.
    """
    img = _open(raw)
    w, h = int(img.width * keep), int(img.height * keep)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return _to_bytes(img.crop((left, top, left + w, top + h)))


def screenshot(raw: bytes, scale: float = 0.85, quality: int = 65) -> bytes:
    """Approximate a screenshot: rescale, requantise, re-encode lossily.

    Not a literal screen capture. It is the composition of the operations a screen capture
    performs on the pixels, which is what the detector actually sees. Named for the user
    behaviour rather than the mechanism, because the robustness table is read by people
    asking "does it survive a screenshot".
    """
    return jpeg(resize(raw, scale), quality=quality)


def blur(raw: bytes, radius: float = 0.8) -> bytes:
    """Mild Gaussian blur. Kept small on purpose: heavy blur is vandalism, not an attack."""
    from PIL import ImageFilter

    return _to_bytes(_open(raw).filter(ImageFilter.GaussianBlur(radius)))


def sharpen(raw: bytes, factor: float = 1.6) -> bytes:
    """Sharpening, which phone pipelines and editing apps apply by default."""
    from PIL import ImageEnhance

    return _to_bytes(ImageEnhance.Sharpness(_open(raw)).enhance(factor))


def strip_metadata(raw: bytes) -> bytes:
    """Remove EXIF and every other container field, keeping the pixels exactly.

    Included because it is what a person trying to hide provenance would do first, and
    because measuring it makes an important point: FORGE's normalisation already strips
    metadata, so a detector that lost accuracy here would have been reading metadata
    rather than pixels. The expected result is no change, and that is the finding.
    """
    return _to_bytes(_open(raw))


def recompress_twice(raw: bytes, first: int = 85, second: int = 45) -> bytes:
    """Two lossy generations, as an image picks up passing through two platforms."""
    return jpeg(jpeg(raw, quality=first), quality=second)


@dataclass(frozen=True)
class Attack:
    name: str
    apply: Callable[[bytes], bytes]
    description: str
    # Geometric attacks change the FRAME: which pixels are present and at what scale.
    # Photometric attacks change the pixels while keeping the frame. The distinction
    # matters because the recognisability check is a global fingerprint comparison, and a
    # crop legitimately changes the global layout. Judging a crop by that check calls
    # correct behaviour vandalism. See preserves_content.
    geometric: bool = False


# The registry. Ordered from mildest to most aggressive, so a robustness table reads as a
# progression rather than an arbitrary list.
ATTACKS: tuple[Attack, ...] = (
    Attack("metadata_strip", strip_metadata, "remove EXIF, keep pixels"),
    Attack("sharpen", lambda b: sharpen(b), "default phone or app sharpening"),
    Attack("blur_mild", lambda b: blur(b, 0.8), "mild gaussian blur"),
    Attack("jpeg_85", lambda b: jpeg(b, 85), "light lossy re-encode"),
    Attack("crop_80", lambda b: crop(b, 0.8), "centre crop, borders removed", True),
    Attack("resize_half", lambda b: resize(b, 0.5), "downscale to 50 percent", True),
    Attack("jpeg_50", lambda b: jpeg(b, 50), "moderate lossy re-encode"),
    Attack("screenshot", lambda b: screenshot(b), "rescale plus lossy re-encode", True),
    Attack("recompress_twice", lambda b: recompress_twice(b), "two lossy generations"),
    Attack("jpeg_25", lambda b: jpeg(b, 25), "heavy lossy re-encode"),
)

ATTACKS_BY_NAME = {attack.name: attack for attack in ATTACKS}


def apply_attack(raw: bytes, name: str) -> bytes:
    if name not in ATTACKS_BY_NAME:
        raise KeyError(f"unknown attack {name!r}; known: {sorted(ATTACKS_BY_NAME)}")
    return ATTACKS_BY_NAME[name].apply(raw)


def preserves_content(
    original: bytes,
    attacked: bytes,
    max_distance: int = 12,
    geometric: bool = False,
) -> bool:
    """Is the attacked image still recognisably the same picture?

    PHOTOMETRIC attacks keep the frame and change the pixels, so a global perceptual hash
    is the right instrument: a JPEG at quality 25 should still fingerprint close to the
    original, and one that does not has destroyed the image.

    GEOMETRIC attacks change the frame by construction. A centre crop removes a fifth of
    every edge, so the global fingerprint moves a long way while the picture remains
    obviously the same picture. Judging a crop by the photometric bound calls correct
    behaviour vandalism, which is how a robustness suite ends up excluding the attacks that
    matter most. The text track did precisely this: its validity check discarded the
    homoglyph attacks that worked.

    For geometric attacks the question is instead whether enough of the original survives,
    which is answered by comparing the attacked image against the same region of the
    original rather than against the whole of it.
    """
    from forge.image.phash import dhash, distance

    if not geometric:
        return distance(dhash(original), dhash(attacked)) <= max_distance

    from PIL import Image

    with Image.open(io.BytesIO(original)) as source, Image.open(io.BytesIO(attacked)) as result:
        source = source.convert("RGB")
        result = result.convert("RGB")
        # Compare like with like: put the attacked image and the CENTRE of the original at
        # the same size, so scale and border differences are removed and only the content
        # is being judged.
        side = min(*result.size)
        keep = min(source.size)
        left = (source.width - keep) // 2
        top = (source.height - keep) // 2
        reference = source.crop((left, top, left + keep, top + keep)).resize((side, side))
        candidate = result.resize((side, side))
        buf_a, buf_b = io.BytesIO(), io.BytesIO()
        reference.save(buf_a, format="PNG")
        candidate.save(buf_b, format="PNG")

    return distance(dhash(buf_a.getvalue()), dhash(buf_b.getvalue())) <= max_distance
