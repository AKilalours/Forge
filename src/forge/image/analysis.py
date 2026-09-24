"""One image analysis, two shells.

The FastAPI route and the Streamlit page must produce the same payload, because they render
the same cards from it. This is the body of `POST /v1/image/analyze` with the HTTP parts
lifted out: the route keeps the upload limits and the status codes, this keeps the work.

`UnsupportedImage` carries the same distinction the route made: a format this build cannot
decode is NAMED, because "unreadable" sends a reader nowhere while "HEIC needs a decoder"
tells them what to do.
"""

from __future__ import annotations

import base64
import io
import time

# Containers the analysis is known to handle. MPO is here because dual-lens phones write it
# and it is a JPEG holding several frames; turning it away rejected ordinary photographs,
# which are exactly the files this project exists to protect from a false accusation.
ACCEPTED = {"JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF", "MPO"}
NEEDS_DECODER = {
    "HEIC": "HEIC needs pillow-heif installed to decode.",
    "HEIF": "HEIF needs pillow-heif installed to decode.",
    "AVIF": "AVIF needs pillow-avif-plugin installed to decode.",
}


class UnsupportedImage(RuntimeError):
    """Raised with a reason a reader can act on, never a bare failure."""


def thumbnail(data: bytes, box: int = 640) -> str | None:
    """A downscaled preview as a data URI. Uploads are never written to disk."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            img.thumbnail((box, box))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:  # noqa: BLE001 - a missing preview is cosmetic, never fatal
        return None


def analyse(data: bytes, filename: str = "upload", with_stability: bool = False,
            with_diagnostics: bool = True) -> dict:
    """Analyse one image and return the payload both pages render.

    The transform-survival pass is opt-in: it re-encodes the image ten times and dominates
    the wall clock. Its absence is reported as absence, never as an empty list that reads as
    "nothing survived". Detector robustness is separate and always runs, because it asks
    whether the VERDICT survives rather than whether a forensic signal does.
    """
    from forge.image.maps import build_maps
    from forge.image.report import build_report

    if not data:
        raise UnsupportedImage("empty upload")

    mark = time.perf_counter()
    preview = thumbnail(data)
    preview_ms = int((time.perf_counter() - mark) * 1000)

    report = build_report(data, filename=filename, preview=preview,
                          with_stability=with_stability,
                          with_diagnostics=with_diagnostics)
    fmt = report.by_name("file_type")
    detected = fmt.value if fmt else None
    if detected not in ACCEPTED:
        hint = NEEDS_DECODER.get(str(detected or "").upper())
        raise UnsupportedImage(
            f"cannot read this file (detected: {detected}). {hint}" if hint
            else f"unsupported or unreadable image (detected: {detected}). "
                 f"Supported: {', '.join(sorted(ACCEPTED))}."
        )

    payload = report.as_dict()
    mark = time.perf_counter()
    # The pixel maps are model-free and cheap, but the deployed page does not render them
    # either, so they follow the same switch.
    payload["maps"] = build_maps(data) if with_diagnostics else []
    payload["timings_ms"] = dict(payload.get("timings_ms") or {})
    payload["timings_ms"]["preview"] = preview_ms
    payload["timings_ms"]["forensic_maps"] = int((time.perf_counter() - mark) * 1000)
    payload["maps_note"] = (
        "Forensic residual maps, not detector saliency. No model is involved. Each map is "
        "normalized within this image, so brightness is relative to this frame and is not "
        "comparable across images."
    )
    return payload
