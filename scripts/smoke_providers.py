"""Run explicit, redacted Pangram and/or Reducto integration smoke tests.

Only fixed fictional content is sent. The script requires a CLI flag and a typed
confirmation before any network request, reads keys only from environment variables, and
writes a report without request or response text. Set required keys in a trusted local
secret manager or shell environment, then run:

    python scripts/smoke_providers.py --provider both --confirm-external

The script prompts for the exact phrase SEND FICTIONAL SAMPLE before sending anything.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/experiments/providers"
CONFIRMATION = "SEND FICTIONAL SAMPLE"
FICTIONAL_TEXT = (
    "The blue cup stayed beside the window through every Sunday afternoon. "
    "Its chipped rim caught the last warm light, and the small ritual made an empty "
    "chair feel like a place where a conversation could continue."
)
PDF_TEXT = (
    "Fictional provider smoke test. The blue cup stayed beside the window on Sunday. "
    "This sample contains no real person, private document, or personal information."
)


def _pdf_escape(text: str) -> bytes:
    encoded = text.encode("ascii", errors="strict")
    return encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def make_fictional_pdf() -> bytes:
    """Build a tiny valid one-page PDF using only the standard library."""
    stream = b"BT /F1 12 Tf 72 720 Td (" + _pdf_escape(PDF_TEXT) + b") Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
        + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    document = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode("ascii"))
        document.extend(body)
        document.extend(b"\nendobj\n")
    xref_offset = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    document.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    document.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(document)


def _run_pangram(api_key: str) -> dict[str, Any]:
    from forge.experience.providers import pangram_analyze

    started = time.perf_counter()
    result = pangram_analyze(FICTIONAL_TEXT, api_key=api_key)
    return {
        "status": "success",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "model": result.get("model"),
        "version": result.get("version"),
        "prediction": result.get("prediction"),
        "window_count": len(result.get("windows", [])),
    }


def _run_reducto(api_key: str) -> dict[str, Any]:
    from forge.experience.providers import reducto_parse

    started = time.perf_counter()
    result = reducto_parse(make_fictional_pdf(), "forge-fictional-smoke.pdf", api_key=api_key)
    return {
        "status": "success",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "pages": result.get("pages"),
        "chunk_count": len(result.get("chunks", [])),
        "text_extracted": bool(result.get("text", "").strip()),
    }


def run_smoke(provider: str, api_key: str) -> dict[str, Any]:
    """Call one selected provider and return sanitized result metadata only."""
    run = _run_pangram if provider == "pangram" else _run_reducto
    try:
        return run(api_key)
    except Exception as error:  # SDK errors can include response text; never persist them.
        return {"status": "failure", "error_type": type(error).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("pangram", "reducto", "both"), required=True)
    parser.add_argument("--confirm-external", action="store_true",
                        help="also type the confirmation phrase before any provider call")
    parser.add_argument("--out", type=Path, default=None,
                        help="redacted JSON report; defaults to reports/experiments/providers/")
    args = parser.parse_args()

    selected = ("pangram", "reducto") if args.provider == "both" else (args.provider,)
    key_names = {"pangram": "PANGRAM_API_KEY", "reducto": "REDUCTO_API_KEY"}
    missing = [key_names[name] for name in selected if not os.environ.get(key_names[name])]
    if missing:
        parser.error("missing environment variable(s): " + ", ".join(missing))
    if not args.confirm_external:
        parser.error("external calls require --confirm-external")
    try:
        typed = input(
            "This sends only fixed fictional text/PDF content to "
            f"{', '.join(selected)}. Type {CONFIRMATION!r} to continue: "
        )
    except EOFError:
        parser.error("confirmation must be typed in an interactive terminal")
    if typed != CONFIRMATION:
        parser.error("confirmation did not match; no provider calls were made")

    results = {name: run_smoke(name, os.environ[key_names[name]]) for name in selected}
    report = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "workload": "fixed fictional passage and generated one-page fictional PDF",
        "content_saved": False,
        "secrets_saved": False,
        "providers": results,
    }
    output = args.out or DEFAULT_OUT / f"live-smoke-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Redacted report: {output}")
    if any(item["status"] != "success" for item in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
