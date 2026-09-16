"""Classical image forensics: everything about a file that needs no trained model.

WHAT THIS IS FOR. A detector's verdict is one signal among several. A JPEG carries its own
history in its quantisation tables, its EXIF, its compression residual and its container
structure, and none of that requires a GPU, a model, or an internet connection. Those
signals are also the ones a reader can verify themselves, which makes them the right thing
to show first.

THE RULE THIS FILE IS BUILT AROUND, and it is not decoration.

    EVERY VALUE IS READ FROM THE FILE OR REPORTED AS ABSENT.

There is no placeholder camera, no invented capture date, no default "looks fine". A tool
whose job is establishing whether an image is authentic cannot itself display a fabricated
provenance record. Where a signal is unavailable, the field says `None` and the UI says
"not available", which is information; a plausible-looking default is a lie.

WHAT THESE SIGNALS CAN AND CANNOT DO. Each finding carries a `caveat` naming its own limit,
because forensic signals are suggestive, not conclusive:

  - A stripped EXIF block is normal for anything that has been through a social platform,
    a screenshot, or a chat app. It is not evidence of generation.
  - Standard quantisation tables mean the file was written by a library using defaults,
    which describes most AI pipelines AND most re-saved photographs.
  - Error level analysis responds to any re-encode, including an innocent one.
  - Resampling periodicity is destroyed by a subsequent JPEG pass.

Reporting these as "evidence of AI" would be exactly the false-positive behaviour the whole
project exists to avoid. They are reported as what they are: facts about the file.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import Any

FORENSICS_VERSION = "image_forensics_v1"

# Signal strength vocabulary, used everywhere so the UI never invents its own wording.
NOT_CHECKED = "not_checked"
NOT_FOUND = "not_found"
PRESENT = "present"
LOW = "low"
MEDIUM = "medium"
HIGH = "high"

# JPEG marker for JUMBF, the container C2PA provenance manifests live in.
_APP11 = b"\xff\xeb"
_C2PA_MARKERS = (b"jumb", b"c2pa", b"urn:uuid:c2pa")

# The IJG standard luminance quantisation table at quality 50. Encoders that use libjpeg's
# defaults scale this table; encoders that tune their own (most cameras) do not match it.
_IJG_LUMA_Q50 = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)


@dataclass(frozen=True)
class Finding:
    """One forensic observation.

    `value` is what was measured or None when the signal is unavailable. `status` is the
    vocabulary above. `caveat` states what this finding cannot establish, and it is required
    rather than optional, because a signal presented without its limits invites the reader
    to over-read it.
    """

    name: str
    value: Any
    status: str
    caveat: str
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _fail_soft(name: str, error: Exception, caveat: str) -> Finding:
    """A signal that could not be computed reports that, rather than a default.

    Swallowing an exception and returning "low" would be the project's recurring bug in a
    new costume: a check that reports a reassuring answer without measuring anything.
    """
    return Finding(
        name=name,
        value=None,
        status=NOT_CHECKED,
        caveat=f"could not be computed on this file: {type(error).__name__}",
        detail={"error": str(error)[:200]},
    )


# --------------------------------------------------------------------------- container


def real_format(data: bytes) -> Finding:
    """The format the BYTES say, not the extension the filename claims.

    A mismatch is worth surfacing: it means something renamed the file, which is common in
    scraped or re-shared images and occasionally deliberate.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt, size, mode = img.format, img.size, img.mode
            frames = int(getattr(img, "n_frames", 1) or 1)
    except Exception as error:  # noqa: BLE001
        return _fail_soft("file_type", error, "")
    caveat = "the container format says how the file was written, not who wrote it"
    if frames > 1:
        # MPO (dual-lens phones, 3D cameras) and multi-frame TIFF hold several images in one
        # file. Everything downstream reads frame 0, which is the frame a viewer shows. Said
        # here because a signal measured on one frame is not a statement about the others.
        caveat += f"; {frames} frames present and only the first is analysed"
    return Finding(
        name="file_type",
        value=fmt,
        status=PRESENT,
        caveat=caveat,
        detail={"width": size[0], "height": size[1], "mode": mode, "bytes": len(data),
                "frames": frames},
    )


def c2pa_status(data: bytes) -> Finding:
    """Is there a C2PA / JUMBF provenance manifest in this file?

    Scanned structurally rather than parsed. A full C2PA validation needs the c2pa library
    and a trust list; what is reported here is only PRESENCE, which is the honest limit of
    a dependency-free scan. Absence is the overwhelmingly common case and means nothing on
    its own: almost no camera or phone signs its output today.
    """
    lowered = data[:4_000_000].lower()
    found = any(marker in lowered for marker in _C2PA_MARKERS) or _APP11 in data[:65536]
    return Finding(
        name="c2pa",
        value="manifest markers present" if found else None,
        status=PRESENT if found else NOT_FOUND,
        caveat=(
            "presence is detected structurally, not cryptographically verified; absence is "
            "normal, as almost no camera signs its output"
        ),
    )


# ------------------------------------------------------------------------------- EXIF


_INTERESTING_EXIF = {
    271: "camera_make",
    272: "camera_model",
    305: "software",
    306: "datetime",
    36867: "datetime_original",
    33437: "f_number",
    33434: "exposure_time",
    34855: "iso",
    37386: "focal_length",
    274: "orientation",
}


def exif(data: bytes) -> Finding:
    """Camera and capture metadata, read from the file.

    Every field here is either in the file or absent. Nothing is inferred, and nothing is
    filled in with a plausible default.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            raw = img.getexif()
            fields = {
                label: str(raw.get(tag)).strip()
                for tag, label in _INTERESTING_EXIF.items()
                if raw.get(tag) not in (None, "")
            }
    except Exception as error:  # noqa: BLE001
        return _fail_soft("exif", error, "")

    if not fields:
        return Finding(
            name="exif",
            value=None,
            status=NOT_FOUND,
            caveat=(
                "no EXIF block. Normal for screenshots, social-platform downloads and chat "
                "apps, which strip metadata. NOT evidence of generation."
            ),
        )
    return Finding(
        name="exif",
        value=fields,
        status=PRESENT,
        caveat="EXIF is trivially editable and is a claim by the writer, not proof",
        detail={"field_count": len(fields)},
    )


def camera_consistency(exif_finding: Finding) -> Finding:
    """Do the EXIF fields hang together the way a camera writes them?

    A real camera writes a make, a model, and exposure triplet together. A file carrying a
    model but no exposure data has usually been through an editor that preserved some tags
    and dropped others. This is a CONSISTENCY observation, not an authenticity verdict.
    """
    if exif_finding.status != PRESENT or not isinstance(exif_finding.value, dict):
        return Finding(
            name="camera_consistency",
            value=None,
            status=NOT_CHECKED,
            caveat="no EXIF to check consistency against",
        )
    fields = exif_finding.value
    capture = {"f_number", "exposure_time", "iso", "focal_length"}
    present = capture & set(fields)
    has_body = "camera_make" in fields or "camera_model" in fields

    if has_body and len(present) >= 3:
        status, value = PRESENT, "consistent"
    elif has_body or present:
        status, value = MEDIUM, "partial"
    else:
        status, value = NOT_FOUND, "no camera fields"
    return Finding(
        name="camera_consistency",
        value=value,
        status=status,
        caveat="consistency describes the metadata's internal shape, not the pixels",
        detail={"capture_fields_present": sorted(present), "body_identified": has_body},
    )


# -------------------------------------------------------------------------- JPEG tables


def _quality_from_table(table: tuple[int, ...]) -> int | None:
    """Estimate libjpeg quality from a luminance quantisation table.

    Inverts the standard scaling of the IJG table. Only meaningful when the encoder used
    that table, which `standard_tables` reports separately.
    """
    if len(table) < 64:
        return None
    ratios = [t / b for t, b in zip(table[:64], _IJG_LUMA_Q50, strict=False) if b]
    if not ratios:
        return None
    scale = sum(ratios) / len(ratios) * 100
    # libjpeg's scaling, inverted. Forward:  Q >= 50 -> scale = 200 - 2Q  (so scale <= 100)
    #                                        Q <  50 -> scale = 5000 / Q  (so scale >  100)
    # The two branches were the wrong way round in the first version. Every save above
    # about q=78 came back as 100, and the test that should have caught it only asserted
    # that a low quality estimates lower than a high one, which stayed true while the
    # function was wrong. Monotonic tests pass on broken monotonic functions.
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return max(1, min(100, int(round(quality))))


def jpeg_tables(data: bytes) -> Finding:
    """Quantisation tables: how, and how hard, this file was compressed.

    Cameras ship tuned tables. Libraries (PIL, OpenCV, most generation pipelines) use scaled
    IJG defaults. So "standard tables" says a library wrote this file, which is true of
    almost every AI image AND of every photograph anyone has re-saved. It narrows the
    writer, it does not identify the source.
    """
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != "JPEG":
                return Finding(
                    name="jpeg_tables",
                    value=None,
                    status=NOT_CHECKED,
                    caveat=f"not a JPEG ({img.format}); quantisation tables do not apply",
                )
            tables = {k: tuple(v) for k, v in (getattr(img, "quantization", {}) or {}).items()}
    except Exception as error:  # noqa: BLE001
        return _fail_soft("jpeg_tables", error, "")

    if not tables:
        return Finding(
            name="jpeg_tables",
            value=None,
            status=NOT_FOUND,
            caveat="no quantisation tables could be read",
        )

    luma = tables.get(0, ())
    quality = _quality_from_table(luma)
    ratios = [t / b for t, b in zip(luma[:64], _IJG_LUMA_Q50, strict=False) if b] if luma else []
    spread = (max(ratios) - min(ratios)) if ratios else None
    # A scaled IJG table has a near-constant ratio to the base table. A camera's tuned table
    # does not. 0.35 is a deliberately loose threshold: false "custom" is harmless here,
    # false "standard" would over-claim.
    standard = spread is not None and spread < 0.35

    return Finding(
        name="jpeg_tables",
        value={"estimated_quality": quality, "standard_tables": standard},
        status=PRESENT,
        caveat=(
            "standard tables mean a software library wrote the file, which is true of most "
            "generated images and of every re-saved photograph alike"
        ),
        detail={"table_count": len(tables), "ratio_spread": round(spread, 4) if spread else None},
    )


# ------------------------------------------------------------------- compression residual


def error_level(data: bytes, quality: int = 90) -> Finding:
    """Error level analysis: how much the image changes when re-encoded.

    Re-save at a fixed quality and measure the residual. Regions edited after the last
    save often sit at a different compression level and show a different residual. This
    responds to ANY re-encode, so a high value means "this file has compression history",
    not "this file was manipulated".

    Reported as the residual's mean and its dispersion across a grid, because a uniform
    residual is unremarkable and a patchy one is the thing worth looking at.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            rgb = img.convert("RGB")
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=quality)
            with Image.open(io.BytesIO(buffer.getvalue())) as again:
                a = np.asarray(rgb, dtype=np.int16)
                b = np.asarray(again.convert("RGB"), dtype=np.int16)
        residual = np.abs(a - b).mean(axis=2)
    except Exception as error:  # noqa: BLE001
        return _fail_soft("error_level", error, "")

    # Grid dispersion: split into 8x8 cells and look at how much cell means vary.
    h, w = residual.shape
    ch, cw = max(1, h // 8), max(1, w // 8)
    cells = [
        float(residual[y : y + ch, x : x + cw].mean())
        for y in range(0, h - ch + 1, ch)
        for x in range(0, w - cw + 1, cw)
    ]
    mean = float(residual.mean())
    dispersion = float(np.std(cells)) if cells else 0.0
    relative = dispersion / mean if mean > 1e-6 else 0.0

    status = HIGH if relative > 0.8 else MEDIUM if relative > 0.45 else LOW
    return Finding(
        name="error_level",
        value={"mean_residual": round(mean, 3), "patchiness": round(relative, 3)},
        status=status,
        caveat=(
            "responds to any re-encode, including innocent ones. Patchiness suggests "
            "regions with different compression history, not necessarily editing"
        ),
        detail={"cells": len(cells), "quality_probe": quality},
    )


def resample_evidence(data: bytes) -> Finding:
    """Periodic structure left behind by resizing.

    Interpolation correlates neighbouring pixels in a periodic way, which shows up as peaks
    in the spectrum of the second difference. The signal is real but fragile: a JPEG pass
    after the resize largely destroys it, so absence proves nothing at all.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            grey = np.asarray(img.convert("L"), dtype=np.float64)
        if min(grey.shape) < 64:
            return Finding(
                name="resample",
                value=None,
                status=NOT_CHECKED,
                caveat="image too small for spectral analysis",
            )
        # Second difference along rows, averaged down columns, then its spectrum.
        second = np.diff(grey, n=2, axis=1)
        profile = np.abs(second).mean(axis=0)
        profile = profile - profile.mean()
        spectrum = np.abs(np.fft.rfft(profile))[1:]
        if spectrum.size < 8:
            raise ValueError("spectrum too short")
        peak = float(spectrum.max())
        median = float(np.median(spectrum)) or 1e-9
        ratio = peak / median
    except Exception as error:  # noqa: BLE001
        return _fail_soft("resample", error, "")

    status = HIGH if ratio > 12 else MEDIUM if ratio > 7 else LOW
    return Finding(
        name="resample",
        value={"peak_to_median": round(ratio, 2)},
        status=status,
        caveat=(
            "a JPEG pass after resizing destroys this signal, so a low value does not mean "
            "the image was never resized"
        ),
    )


# ------------------------------------------------------------------------------ summary


_ORDER = {LOW: 0, NOT_FOUND: 0, NOT_CHECKED: 0, MEDIUM: 1, PRESENT: 1, HIGH: 2}


def manipulation_summary(findings: list[Finding]) -> Finding:
    """Aggregate the manipulation-related signals into one honest sentence.

    Deliberately NOT a score out of 100. These signals do not combine into a probability,
    and presenting them as one invites a reader to treat a guess as a measurement. What is
    reported is the strongest individual signal and which signals drove it.
    """
    relevant = [
        f for f in findings
        if f.name in {"error_level", "resample", "noise_consistency", "colour"}
    ]
    scored = [(f, _ORDER.get(f.status, 0)) for f in relevant if f.status != NOT_CHECKED]
    if not scored:
        return Finding(
            name="manipulation_summary",
            value=None,
            status=NOT_CHECKED,
            caveat="no manipulation signal could be computed on this file",
        )
    worst = max(scored, key=lambda pair: pair[1])
    status = [LOW, MEDIUM, HIGH][worst[1]]
    return Finding(
        name="manipulation_summary",
        value=status,
        status=status,
        caveat=(
            "the strongest single signal, not a combined probability. These signals do not "
            "multiply into a likelihood and are not presented as one"
        ),
        detail={f.name: f.status for f, _ in scored},
    )


def analyze(data: bytes) -> list[Finding]:
    """Run every model-free signal over one file. No GPU, no network, no weights."""
    fmt = real_format(data)
    ex = exif(data)
    findings = [
        fmt,
        ex,
        camera_consistency(ex),
        c2pa_status(data),
        ai_declaration(data),
        jpeg_tables(data),
        error_level(data),
        resample_evidence(data),
        noise_consistency(data),
        colour_statistics(data),
    ]
    findings.append(manipulation_summary(findings))
    return findings


# ------------------------------------------------------- self-declared generation markers


# Generators increasingly stamp their output, and these are the real, checkable
# declarations. This is the ONE model-free signal that can positively indicate AI rather
# than merely describing how a file was written.
#
#   IPTC digitalSourceType = trainedAlgorithmicMedia  is the published standard for
#   declaring synthetic media. Adobe Firefly, Google and others write it.
#
#   PNG tEXt "parameters" is what AUTOMATIC1111 and most Stable Diffusion front-ends
#   write: the literal prompt, sampler, seed and model hash.
#
#   The rest are software strings generators leave in EXIF or XMP.
_AI_DECLARATIONS = (
    (b"trainedalgorithmicmedia", "IPTC digitalSourceType: trained algorithmic media"),
    (b"compositewithtrainedalgorithmicmedia", "IPTC: composite with algorithmic media"),
    (b"algorithmicmedia", "IPTC digitalSourceType: algorithmic media"),
    (b"stable diffusion", "Stable Diffusion software string"),
    (b"stablediffusion", "Stable Diffusion software string"),
    (b"midjourney", "Midjourney software string"),
    (b"dall-e", "DALL-E software string"),
    (b"firefly", "Adobe Firefly software string"),
    (b"imagen", "Google Imagen software string"),
    (b"comfyui", "ComfyUI workflow marker"),
    (b"automatic1111", "AUTOMATIC1111 marker"),
    (b"invokeai", "InvokeAI marker"),
    (b"novelai", "NovelAI marker"),
)

# Keys Stable Diffusion pipelines write into PNG text chunks.
_SD_PARAM_KEYS = (b"parameters", b"negative prompt", b"sampler", b"cfg scale", b"model hash")


def ai_declaration(data: bytes) -> Finding:
    """Does the file DECLARE that it was generated?

    The only model-free signal that can positively indicate AI, and a strong one when
    present, because nothing else writes these strings. Scanned over the metadata region
    rather than the whole file so compressed pixel data cannot produce a coincidental hit.

    THE ASYMMETRY, which the UI must preserve:

        PRESENT -> the file says it is generated. Believe it, and note it is a claim by
                   the producing software rather than a measurement of the pixels.
        ABSENT  -> nothing at all. Most generators write nothing, every editor strips
                   metadata, and a screenshot of an AI image carries none of it.

    Reading absence as evidence of human authorship would be the same false-positive engine
    this project exists to avoid, pointed the other way.
    """
    head = data[:1_048_576].lower()
    hits = [label for marker, label in _AI_DECLARATIONS if marker in head]
    sd_keys = [k.decode() for k in _SD_PARAM_KEYS if k in head]
    if len(sd_keys) >= 2:
        hits.append("Stable Diffusion parameter block (" + ", ".join(sd_keys[:3]) + ")")

    if hits:
        return Finding(
            name="ai_declaration",
            value=hits,
            status=HIGH,
            caveat=(
                "the file declares generated content. A claim written by the producing "
                "software, and the strongest model-free indicator available"
            ),
            detail={"markers": len(hits)},
        )
    return Finding(
        name="ai_declaration",
        value=None,
        status=NOT_FOUND,
        caveat=(
            "no self-declaration. This means NOTHING: most generators write none, every "
            "editor strips metadata, and a screenshot of an AI image carries none"
        ),
    )


def noise_consistency(data: bytes) -> Finding:
    """Is the high-frequency noise floor uniform across the frame?

    A photograph carries roughly uniform sensor noise. Regions pasted, inpainted or heavily
    denoised sit at a different level. Measured as the dispersion of a Laplacian high-pass
    residual across a grid.

    Smooth sky, shallow depth of field and denoising all produce uneven noise in entirely
    genuine photographs, so this describes the frame rather than accusing it.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            grey = np.asarray(img.convert("L"), dtype=np.float64)
        if min(grey.shape) < 48:
            return Finding(
                "noise_consistency", None, NOT_CHECKED,
                "image too small to estimate a noise floor",
            )
        hp = (
            grey[1:-1, 1:-1] * 4
            - grey[:-2, 1:-1] - grey[2:, 1:-1] - grey[1:-1, :-2] - grey[1:-1, 2:]
        )
        h, w = hp.shape
        ch, cw = max(1, h // 6), max(1, w // 6)
        cells = [
            float(hp[y : y + ch, x : x + cw].std())
            for y in range(0, h - ch + 1, ch)
            for x in range(0, w - cw + 1, cw)
        ]
        cells = [c for c in cells if c > 1e-6]
        if len(cells) < 4:
            raise ValueError("too few usable cells")
        spread = float(np.std(cells) / np.mean(cells))
    except Exception as error:  # noqa: BLE001
        return _fail_soft("noise_consistency", error, "")

    status = HIGH if spread > 1.1 else MEDIUM if spread > 0.7 else LOW
    return Finding(
        name="noise_consistency",
        value={"variation": round(spread, 3)},
        status=status,
        caveat=(
            "smooth skies, shallow depth of field and denoising all produce uneven noise "
            "in genuine photographs; this describes the frame, it does not accuse it"
        ),
    )


def colour_statistics(data: bytes) -> Finding:
    """Channel balance, saturation and clipping.

    A reader looking at a colour and lighting row deserves the actual numbers rather than a
    verdict. Heavy clipping suggests aggressive processing and nothing more: no colour
    statistic distinguishes a generator from a camera.
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float64)
        means = rgb.reshape(-1, 3).mean(axis=0)
        maxc, minc = rgb.max(axis=2), rgb.min(axis=2)
        saturation = float(np.mean((maxc - minc) / np.maximum(maxc, 1e-6)))
        clipped = float((rgb >= 254).all(axis=2).mean() + (rgb <= 1).all(axis=2).mean())
    except Exception as error:  # noqa: BLE001
        return _fail_soft("colour", error, "")

    status = HIGH if clipped > 0.12 else MEDIUM if clipped > 0.04 else LOW
    return Finding(
        name="colour",
        value={
            "channel_means": [round(float(m), 1) for m in means],
            "saturation": round(saturation, 3),
            "clipped_fraction": round(clipped, 4),
        },
        status=status,
        caveat="describes grading and exposure; no colour statistic separates AI from a camera",
    )
