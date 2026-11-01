"""Small, opt-in adapters for Pangram analysis and Reducto document parsing.

The adapters never persist document text. Callers must collect explicit user consent
before calling them because both providers process uploaded content outside this app.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_PROVIDER_TEXT_CHARS = 50_000
MAX_PROVIDER_FILE_BYTES = 15 * 1024 * 1024
ALLOWED_DOCUMENT_SUFFIXES = {".pdf", ".docx"}


class ProviderError(RuntimeError):
    """A provider call failed or returned an unusable response."""


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def pangram_analyze(
    text: str,
    *,
    api_key: str,
    model_preference: str = "pangram-4",
    client_factory: Callable[..., Any] | None = None,
) -> dict:
    """Get Pangram's model and window signals without publishing a dashboard link."""
    if not text.strip() or len(text) > MAX_PROVIDER_TEXT_CHARS:
        raise ValueError(f"text must contain 1–{MAX_PROVIDER_TEXT_CHARS:,} characters")
    if not api_key.strip():
        raise ValueError("PANGRAM_API_KEY is not configured")
    try:
        if client_factory is None:
            from pangram import Pangram

            client_factory = Pangram
        client = client_factory(api_key=api_key)
        models = client.list_models()
        if not models:
            raise ProviderError("Pangram returned no models for this API key")
        model = model_preference if model_preference in models else models[0]
        result = client.predict(text, model=model, public_dashboard_link=False)
        if _value(result, "stage") not in (None, "STAGE_SUCCESS"):
            raise ProviderError(f"Pangram analysis did not complete: {_value(result, 'stage')}")
        windows = []
        for window in (_value(result, "windows", []) or [])[:40]:
            windows.append({
                "label": _value(window, "label"),
                "score": _value(window, "ai_assistance_score"),
                "confidence": _value(window, "confidence"),
                "start": _value(window, "start_index"),
                "end": _value(window, "end_index"),
                "text": str(_value(window, "text", ""))[:MAX_PROVIDER_TEXT_CHARS],
            })
        return {
            "model": model,
            "version": _value(result, "version"),
            "prediction": _value(result, "prediction_short"),
            "fraction_ai": _value(result, "fraction_ai"),
            "fraction_ai_assisted": _value(result, "fraction_ai_assisted"),
            "fraction_human": _value(result, "fraction_human"),
            "windows": windows,
        }
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(f"Pangram request failed ({type(error).__name__}): {error}") from error


def reducto_parse(
    data: bytes,
    filename: str,
    *,
    api_key: str,
    client_factory: Callable[..., Any] | None = None,
) -> dict:
    """Temporarily stage a document for Reducto, then remove the local temporary file."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_SUFFIXES:
        raise ValueError(f"Reducto accepts: {', '.join(sorted(ALLOWED_DOCUMENT_SUFFIXES))}")
    if not data or len(data) > MAX_PROVIDER_FILE_BYTES:
        raise ValueError(f"document must be between 1 byte and {MAX_PROVIDER_FILE_BYTES:,} bytes")
    if not api_key.strip():
        raise ValueError("REDUCTO_API_KEY is not configured")

    try:
        if client_factory is None:
            from reducto import Reducto

            client_factory = Reducto
        with tempfile.NamedTemporaryFile(prefix="forge-parse-", suffix=suffix) as temp:
            temp.write(data)
            temp.flush()
            client = client_factory(api_key=api_key)
            uploaded = client.upload(file=Path(temp.name))
            parsed = client.parse.run(input=uploaded.file_id)
        chunks = []
        text_parts: list[str] = []
        for chunk in (_value(_value(parsed, "result"), "chunks", []) or []):
            blocks = []
            for block in (_value(chunk, "blocks", []) or []):
                bbox = _value(block, "bbox")
                page = _value(bbox, "page") if bbox is not None else None
                blocks.append({
                    "type": _value(block, "type"),
                    "page": page,
                    "text": str(_value(block, "content", ""))[:2000],
                })
            chunk_text = str(_value(chunk, "content", ""))[:MAX_PROVIDER_TEXT_CHARS]
            if chunk_text.strip():
                text_parts.append(chunk_text)
            chunks.append({
                "text": chunk_text,
                "blocks": blocks,
            })
        full_text = "\n\n".join(text_parts)
        if not full_text:
            raise ProviderError("Reducto completed without extractable text")
        if len(full_text) > MAX_PROVIDER_TEXT_CHARS:
            full_text = full_text[:MAX_PROVIDER_TEXT_CHARS]
        usage = _value(parsed, "usage")
        return {
            "text": full_text,
            "chunks": chunks,
            "job_id": _value(parsed, "job_id"),
            "pages": _value(usage, "num_pages"),
        }
    except ProviderError:
        raise
    except Exception as error:
        raise ProviderError(f"Reducto request failed ({type(error).__name__}): {error}") from error
