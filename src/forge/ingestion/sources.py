"""Source adapters. One per corpus, all yielding RawRecord.

Each adapter records provenance at the moment of acquisition: what was fetched, from
where, under what license, on what date. Provenance reconstructed later is guesswork.

The HF adapters stream rather than download. FineWeb sample-10BT is 27.6 GB on disk;
streaming means we read only the documents we keep.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class RawRecord:
    """Exactly what the upstream gave us, plus where it came from. No cleaning yet."""

    source_id: str
    source_record_id: str
    text: str
    license: str
    domain: str
    register: str
    acquired_at: datetime
    source_config: str | None = None
    date: str | None = None
    upstream_language: tuple[str, float] | None = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Source(Protocol):
    id: str
    license: str

    def stream(self, limit: int | None = None) -> Iterator[RawRecord]: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _HFStreamingSource:
    """Shared base for the two FineWeb variants."""

    id = "hf"
    license = "ODC-By-1.0"
    hf_repo = ""
    domain = "web"
    register = "informational"

    def __init__(self, config: str = "sample-10BT", filters: dict | None = None) -> None:
        self.config = config
        self.filters = filters or {}

    def _keep(self, row: dict) -> bool:
        min_lang = self.filters.get("language_score_min")
        if min_lang is not None and float(row.get("language_score", 0.0)) < min_lang:
            return False
        min_edu = self.filters.get("int_score_min")
        if min_edu is not None and int(row.get("int_score", 0)) < min_edu:
            return False
        return True

    def stream(self, limit: int | None = None) -> Iterator[RawRecord]:
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Streaming a HuggingFace corpus needs the `data` extra. "
                "Run: pip install -e '.[data]'"
            ) from e

        acquired = _now()
        ds = load_dataset(self.hf_repo, name=self.config, split="train", streaming=True)
        kept = 0
        for row in ds:
            if not self._keep(row):
                continue
            yield RawRecord(
                source_id=self.id,
                source_record_id=str(row.get("id", "")),
                text=row.get("text", ""),
                license=self.license,
                domain=self.domain,
                register=self.register,
                acquired_at=acquired,
                source_config=self.config,
                date=str(row.get("date")) if row.get("date") else None,
                upstream_language=(row.get("language", "en"), float(row.get("language_score", 0.0))),
                extra={"url": row.get("url"), "edu_score": row.get("score")},
            )
            kept += 1
            if limit is not None and kept >= limit:
                return


class FineWebSource(_HFStreamingSource):
    id = "fw"
    hf_repo = "HuggingFaceFW/fineweb"
    domain = "web"
    register = "informational"


class FineWebEduSource(_HFStreamingSource):
    id = "fwe"
    hf_repo = "HuggingFaceFW/fineweb-edu"
    domain = "academic"
    register = "explanatory"


class GutenbergSource:
    """Public-domain books. Long-form literary register, which Common Crawl underrepresents
    and which is a known source of human false positives in production detectors."""

    id = "gut"
    license = "public-domain-us"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def stream(self, limit: int | None = None) -> Iterator[RawRecord]:
        acquired = _now()
        n = 0
        for p in sorted(self.root.rglob("*.txt")):
            yield RawRecord(
                source_id=self.id,
                source_record_id=p.stem,
                text=_strip_gutenberg_boilerplate(p.read_text(errors="replace")),
                license=self.license,
                domain="books",
                register="literary",
                acquired_at=acquired,
            )
            n += 1
            if limit is not None and n >= limit:
                return


_PG_START = "*** START OF"
_PG_END = "*** END OF"


def _strip_gutenberg_boilerplate(text: str) -> str:
    """The PG header and license footer are identical across thousands of books.
    Leaving them in would give the deduper thousands of false near-duplicate hits and
    teach the detector that boilerplate means human."""
    i = text.find(_PG_START)
    if i != -1:
        nl = text.find("\n", i)
        text = text[nl + 1 :] if nl != -1 else text
    j = text.find(_PG_END)
    if j != -1:
        text = text[:j]
    return text.strip()


class GovInfoSource:
    """US federal documents. Formal/bureaucratic register."""

    id = "gov"
    license = "public-domain-us-gov"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def stream(self, limit: int | None = None) -> Iterator[RawRecord]:
        acquired = _now()
        n = 0
        for p in sorted(self.root.rglob("*.txt")):
            yield RawRecord(
                source_id=self.id,
                source_record_id=p.stem,
                text=p.read_text(errors="replace"),
                license=self.license,
                domain="government",
                register="formal",
                acquired_at=acquired,
            )
            n += 1
            if limit is not None and n >= limit:
                return


class LocalJSONLSource:
    """Any local JSONL with a `text` field.

    This exists so the pipeline can be exercised end to end without network access, and
    so a corpus acquired by hand can be ingested through the same validated path rather
    than a one-off script.
    """

    def __init__(
        self,
        path: str | Path,
        source_id: str = "local",
        license: str = "unknown",
        domain: str = "web",
        register: str = "informational",
    ) -> None:
        self.path = Path(path)
        self.id = source_id
        self.license = license
        self.domain = domain
        self.register = register

    def stream(self, limit: int | None = None) -> Iterator[RawRecord]:
        acquired = _now()
        with self.path.open() as fh:
            for n, line in enumerate(fh):
                if limit is not None and n >= limit:
                    return
                row = json.loads(line)
                yield RawRecord(
                    source_id=self.id,
                    source_record_id=str(row.get("id", n)),
                    text=row.get("text", ""),
                    license=row.get("license", self.license),
                    domain=row.get("domain", self.domain),
                    register=row.get("register", self.register),
                    acquired_at=acquired,
                    date=row.get("date"),
                )


REGISTRY = {
    "fw": FineWebSource,
    "fwe": FineWebEduSource,
    "gut": GutenbergSource,
    "gov": GovInfoSource,
    "local": LocalJSONLSource,
}
