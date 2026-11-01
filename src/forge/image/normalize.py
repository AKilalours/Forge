"""One normalisation, used by every stage that touches an image.

WHY THIS IS THE FIRST FILE IN THE IMAGE TRACK.

Real photographs arrive as JPEG: shot by a camera, resized by a hosting service, often
recompressed several times, carrying EXIF. Diffusion output arrives as a lossless PNG at
exactly 1024x1024, never compressed, never resized, with no camera pipeline behind it and
no EXIF at all.

A detector trained on those two populations reaches near-perfect accuracy WITHOUT EVER
LOOKING at whether an image was generated. It learns JPEG quantisation, resampling
signatures, aspect-ratio priors and the presence or absence of metadata. It then collapses
the first time someone screenshots an AI image or saves a real photograph as PNG.

That is the image-track equivalent of a text detector learning the topic, and it is worse,
because accuracy, AUROC and every other headline metric look excellent while it happens.

So: every image, human or generated, passes through THIS function before anything sees it.
One implementation, one policy object, no per-call-site variation. If ingestion and
generation normalise differently, the difference between them becomes the signal.

Nothing here is clever. Being boring and shared is the entire point.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

# Frozen with the spec. Changing any field changes what every model has ever seen, so it
# is a dataset-version change, not a tweak.
POLICY_VERSION = "image_norm_v1"


@dataclass(frozen=True)
class NormalizationPolicy:
    size: int = 512            # output is size x size, always square
    jpeg_quality: int = 90     # one quality for every image, human or generated
    subsampling: int = 2       # 4:2:0, the common web default
    version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        if self.size < 32:
            raise ValueError(f"size {self.size} is too small to be useful")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality {self.jpeg_quality} out of range")


POLICY_V1 = NormalizationPolicy()


class ImageTooSmallError(ValueError):
    """Upscaling invents detail, and invented detail is a generation artifact."""


def normalize_bytes(
    raw: bytes,
    policy: NormalizationPolicy = POLICY_V1,
    allow_upscale: bool = False,
) -> bytes:
    """Return JPEG bytes that carry no trace of how the input was encoded.

    Steps, in this order and for these reasons:

    1. Apply EXIF orientation, THEN discard EXIF. A phone photograph is often stored
       sideways with a rotation flag. Stripping the flag first would leave the crop taking
       the wrong region, which is a silent, systematic corruption of one class only.
    2. Convert to RGB. Alpha and palette channels appear in generated PNGs and almost never
       in photographs, so the channel count is itself a source signature.
    3. Centre crop to a square BEFORE resizing, so aspect ratio cannot survive as a
       feature. Photographs are 4:3 and 3:2; diffusion output is 1:1.
    4. Resize to the policy size with one resampling filter.
    5. Re-encode as JPEG at one fixed quality with fixed subsampling, no metadata.

    UPSCALING: TWO CALLERS, TWO ANSWERS.

    When BUILDING A CORPUS, an image below the policy size is rejected. Interpolation
    fabricates high-frequency content, which is precisely what the detector is being asked
    to judge, and there are nine million candidate images, so discarding the small ones
    costs nothing.

    When SERVING or EVALUATING, refusing is not available. A user uploads what they upload,
    and a detector that raises on a 400px image is not a detector. Worse, half the
    adversarial suite produces small images by construction: downscaling is one of the most
    common things that happens to an image online, so a pipeline that cannot process a
    downscaled image cannot measure robustness to downscaling at all.

    So `allow_upscale` is an explicit choice by the caller rather than a default, and the
    upscaled path is a documented compromise: it fabricates detail, and that is a reason to
    treat such a prediction with less confidence, not a reason to refuse the request.
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(raw)) as img:
        oriented = ImageOps.exif_transpose(img)  # 1
        assert oriented is not None
        rgb = oriented.convert("RGB")            # 2

        short_side = min(rgb.size)
        if short_side < policy.size and not allow_upscale:
            raise ImageTooSmallError(
                f"image is {rgb.size[0]}x{rgb.size[1]}, short side {short_side} is below "
                f"the policy size {policy.size}. Upscaling would fabricate detail. Pass "
                "allow_upscale=True on the serving and evaluation paths, where refusing "
                "an image is not an option."
            )

        left = (rgb.size[0] - short_side) // 2
        top = (rgb.size[1] - short_side) // 2
        cropped = rgb.crop((left, top, left + short_side, top + short_side))  # 3
        resized = cropped.resize(
            (policy.size, policy.size), Image.Resampling.LANCZOS
        )  # 4

        buf = io.BytesIO()
        resized.save(                                                      # 5
            buf,
            format="JPEG",
            quality=policy.jpeg_quality,
            subsampling=policy.subsampling,
            optimize=False,
            progressive=False,
        )
    return buf.getvalue()


def was_upscaled(raw: bytes, policy: NormalizationPolicy = POLICY_V1) -> bool:
    """Would normalising this image have fabricated detail?

    Recorded alongside a prediction so an upscaled input can be identified later. A model
    is not being lied to here, but it is being shown interpolated pixels, and a failure on
    such an image means something different from a failure on a native-resolution one.
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        return min(img.size) < policy.size


def describe(raw: bytes) -> dict:
    """Facts about an image, for the manifest. Never fed to the model in v1.

    Resolution and format are recorded because provenance needs them, and withheld from
    training because a model given the resolution would use it instead of the pixels.
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        return {
            "format": img.format,
            "mode": img.mode,
            "width": img.size[0],
            "height": img.size[1],
            "has_exif": bool(getattr(img, "_getexif", lambda: None)()),
        }
