"""Common Crawl as a FORGE source: the actual web, not a curated redistribution of it.

WHY THIS EXISTS AND FINEWEB DOES NOT COVER IT. `sources.FineWebSource` streams a corpus
somebody else already mined, filtered and deduplicated. That is the right input for
training and it stays the right input. It is not a mining pipeline: the selection, the
extraction, the language decision and the quality decision were all made upstream, and the
only engineering left is `load_dataset(..., streaming=True)`. This module does the part
FineWeb skipped, against the same crawl FineWeb is derived from.

WHAT MINING ACTUALLY INVOLVES HERE, in the order the code does it:

  1. Resolve a crawl. `collinfo.json` lists what exists. Guessing a crawl id and finding
     out from a 404 ninety thousand paths later is not a resolution strategy.
  2. Read the segment index. One crawl is roughly 90,000 WET files of about 150 MB each,
     so the index is the unit of parallelism and the file is the unit of work.
  3. Stream each segment. Never download it. A segment is decompressed through a pipe and
     discarded as it goes, so peak disk is the records kept, not the crawl touched.
  4. Parse WARC records, skip the `warcinfo` header record, and carry per record
     provenance: target URI, fetch date, record id, and the segment it came from.
  5. Hand each one to the existing cleaner as a `RawRecord`. Language, length, quality,
     PII and dedup decisions stay where they already live.

WET OR WARC, AND WHY WET IS THE DEFAULT. WET is Common Crawl's own plain-text extraction.
WARC is the raw HTTP response, HTML and all. Extracting from WARC ourselves with selectolax
is supported here (`fmt="warc"`) and it is the more impressive-sounding path, but it is not
automatically the better one: our boilerplate removal would differ from the extraction every
public Common Crawl derivative used, which means a corpus that is different without being
demonstrably better, and a detector trained on it could not be compared to anything. So WET
is the default and WARC is available for the case where the HTML itself matters. Saying
which one you used, and why, is the part that survives being asked about.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not decide language, length, quality or
duplication. Those are `forge.cleaning`, they are governed by the frozen spec, and a source
that filtered on its own would make two sources produce differently-shaped corpora from the
same policy. A source's job is to yield what upstream gave us plus where it came from.

NETWORK FAILURE IS NOT END OF FILE. A truncated gzip stream looks exactly like a segment
that ended, and a mining run that silently keeps 40% of a segment produces a corpus whose
size nobody can reproduce. Every segment is counted and verified against a clean stream
termination, and a short read raises. See `CrawlStreamError`.
"""

from __future__ import annotations

import gzip
import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from forge.ingestion.sources import RawRecord

# Common Crawl serves these over plain HTTPS with no credentials and no rate limit worth
# working around. The S3 mirror (s3://commoncrawl/...) is the same bytes and needs boto3.
BASE = "https://data.commoncrawl.org"
COLLINFO = "https://index.commoncrawl.org/collinfo.json"

# ODC-BY. Common Crawl publishes the crawl under it; the underlying pages keep their own
# terms, which is why documents from here are written with redistributable=False and the
# publishable artifact is metadata only. See writer.write_metadata_only.
LICENSE = "CC-BY-SA-4.0-crawl/ODC-By-1.0"

FORMATS = ("wet", "warc")


class CrawlStreamError(RuntimeError):
    """A segment did not stream to a clean end, so its record count cannot be trusted."""


@dataclass(frozen=True)
class Segment:
    """One WET or WARC file in one crawl. The unit of parallel work."""

    crawl: str
    path: str

    @property
    def url(self) -> str:
        return f"{BASE}/{self.path}"

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


def _requests():
    try:
        import requests
    except ImportError as error:  # pragma: no cover - environment, not logic
        raise RuntimeError(
            "Mining Common Crawl needs the `data` extra. Run: pip install -e '.[data]'"
        ) from error
    return requests


def available_crawls() -> list[str]:
    """Crawl ids newest first, e.g. CC-MAIN-2026-05.

    Resolved rather than assumed. A crawl id is a date-derived string and the obvious
    guess for "the current one" is wrong for most of any given quarter.
    """
    requests = _requests()
    response = requests.get(COLLINFO, timeout=60)
    response.raise_for_status()
    return [entry["id"] for entry in response.json()]


def segment_paths(crawl: str, fmt: str = "wet", limit: int | None = None) -> list[Segment]:
    """The crawl's segment index, which is itself a gzipped list about 2 MB compressed.

    `limit` takes the FIRST n rather than a sample. The index is ordered by segment and
    the segments are not topically sorted, so the head of the list is not a biased slice,
    and taking the head is reproducible in a way that a random sample is not unless the
    seed is also pinned and recorded. If you want a different slice, slice the return.
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    requests = _requests()
    url = f"{BASE}/crawl-data/{crawl}/{fmt}.paths.gz"
    response = requests.get(url, timeout=120)
    if response.status_code == 404:
        raise CrawlStreamError(
            f"no {fmt} index for crawl {crawl!r} at {url}. "
            f"Call available_crawls() for the ids that exist."
        )
    response.raise_for_status()
    paths = gzip.decompress(response.content).decode().split()
    if limit is not None:
        paths = paths[:limit]
    return [Segment(crawl=crawl, path=path) for path in paths]


def _headers(record) -> dict:
    """WARC headers as a plain dict, so nothing downstream holds a warcio object."""
    return {key.lower(): value for key, value in record.rec_headers.headers}


class _CountedReader(io.RawIOBase):
    """Counts the compressed bytes actually pulled off the socket.

    This exists because of a real defect. The first version of `stream_segment` guarded
    truncation with `except (EOFError, OSError, BadGzipFile)` and that guard could never
    fire: Common Crawl gzips every WARC record as its own gzip member, so a stream that
    stops mid-member looks to warcio like the end of the archive and iteration finishes
    cleanly. A dropped connection therefore produced a short segment that reported
    success. The test that serves half a file caught it.

    So completeness is established positively rather than by the absence of an error.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        chunk = self._inner.read(len(buffer))
        if not chunk:
            return 0
        buffer[: len(chunk)] = chunk
        self.count += len(chunk)
        return len(chunk)


def stream_segment(
    segment: Segment,
    fmt: str = "wet",
    timeout: int = 300,
    verify_length: bool = True,
) -> Iterator[RawRecord]:
    """Stream one segment and yield a RawRecord per document.

    Streamed, not downloaded: the response body is fed straight into warcio through a
    buffered reader, so a 150 MB segment costs a few MB of memory and no disk at all.
    That is what makes running this over hundreds of segments on a laptop possible.

    `verify_length=False` disables the completeness check. It exists for a mirror that
    serves these chunked, and it is off the honest path: a segment whose byte count
    cannot be checked is a segment whose record count cannot be defended.
    """
    try:
        from warcio.archiveiterator import ArchiveIterator
        from warcio.exceptions import ArchiveLoadFailed
    except ImportError as error:  # pragma: no cover - environment, not logic
        raise RuntimeError(
            "Parsing Common Crawl archives needs warcio, in the `data` extra. "
            "Run: pip install -e '.[data]'"
        ) from error

    acquired = datetime.now(timezone.utc)

    with _open_segment(segment, timeout) as (body, declared):
        if verify_length and declared is None:
            raise CrawlStreamError(
                f"{segment.name} was served without a Content-Length, so a truncated "
                f"download cannot be told from a complete one. Re-run against "
                f"{BASE}, or pass verify_length=False and accept that this segment's "
                f"record count is unverified."
            )

        counted = _CountedReader(body)
        # Raw bytes, not decoded: warcio does the gzip itself, per record, which is what
        # makes a WARC seekable. Letting requests decode first would hand it plain bytes
        # and break record framing on the multi-member gzip Common Crawl uses.
        raw = io.BufferedReader(counted, buffer_size=1 << 20)
        try:
            for record in ArchiveIterator(raw):
                built = _record_to_raw(record, segment, fmt, acquired)
                if built is not None:
                    yield built
        except (ArchiveLoadFailed, EOFError, OSError, gzip.BadGzipFile) as error:
            # Kept, because warcio DOES raise on some malformed framing. It just does
            # not raise on the common case, which is why the byte check below exists.
            raise CrawlStreamError(
                f"{segment.name} failed to parse ({type(error).__name__}: {error}). "
                f"Its record count is incomplete, so it must be re-run rather than kept."
            ) from error

        if verify_length and counted.count != declared:
            raise CrawlStreamError(
                f"{segment.name} stopped mid-stream: read {counted.count} of "
                f"{declared} bytes. warcio ended the archive cleanly at the break, "
                f"so the records yielded look complete and are not. Re-run this segment."
            )


def is_local(path: str) -> bool:
    """A segment path that names a file on this machine rather than a crawl object."""
    return path.startswith("/") or path.startswith("file://")


@contextmanager
def _open_segment(segment: Segment, timeout: int):
    """Yield (byte stream, declared length) for a crawl object or a local WET file.

    LOCAL FILES ARE NOT A TEST HOOK BOLTED ON. Two real uses: re-running a crawl ingest
    over segments already on disk, without paying for the network twice; and driving the
    Spark and Beam runners through their actual worker processes in tests. A monkeypatch
    of `requests` lives in the test process and never reaches a Spark Python worker, so a
    runner test built on one would only ever have tested the driver.

    The declared length of a local file is its size on disk, so the completeness check
    runs identically on both paths rather than being skipped for the convenient one.
    """
    if is_local(segment.path):
        path = segment.path.removeprefix("file://")
        with open(path, "rb") as handle:
            yield handle, os.path.getsize(path)
        return

    requests = _requests()
    with requests.get(segment.url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        yield response.raw, (int(declared) if declared is not None else None)


def _record_to_raw(record, segment: Segment, fmt: str, acquired: datetime) -> RawRecord | None:
    """One WARC record to a RawRecord, or None for records that are not documents."""
    wanted = "conversion" if fmt == "wet" else "response"
    if record.rec_type != wanted:
        # warcinfo (one per file) and, in WARC, request/metadata records.
        return None

    headers = _headers(record)
    payload = record.content_stream().read()
    if fmt == "warc":
        text = _extract_html(payload)
    else:
        text = payload.decode("utf-8", errors="replace")
    if not text.strip():
        return None

    uri = headers.get("warc-target-uri", "")
    return RawRecord(
        source_id="cc",
        # The WARC record id is globally unique and stable across re-downloads of the
        # same crawl, which a URL is not: the same page appears in many crawls.
        source_record_id=headers.get("warc-record-id", "").strip("<>"),
        text=text,
        license=LICENSE,
        domain="web",
        register="informational",
        acquired_at=acquired,
        source_config=segment.crawl,
        date=headers.get("warc-date"),
        # NOT set from WARC-Identified-Content-Language. Common Crawl reports a language
        # with no confidence attached, and upstream_language is a (label, score) pair.
        # Inventing a score to fill the tuple would put a fabricated number into the
        # corpus metadata, so the language decision stays with cleaning/langid.py and the
        # crawl's own guess is kept as provenance only.
        upstream_language=None,
        extra={
            "url": uri,
            "segment": segment.path,
            "crawl": segment.crawl,
            "cc_language": headers.get("warc-identified-content-language"),
            "content_length": headers.get("content-length"),
        },
    )


def _extract_html(payload: bytes) -> str:
    """Boilerplate-stripped text from a raw HTTP response body.

    Only reached with fmt="warc". selectolax is already a `data` extra dependency because
    cleaning uses it for markup stripping, so this adds no new requirement.
    """
    from selectolax.parser import HTMLParser

    body = payload.split(b"\r\n\r\n", 1)[-1]
    tree = HTMLParser(body.decode("utf-8", errors="replace"))
    for tag in tree.css("script, style, nav, header, footer, aside, noscript"):
        tag.decompose()
    node = tree.body or tree.root
    return node.text(separator="\n", strip=True) if node else ""


class CommonCrawlSource:
    """A `sources.Source` over one or more crawl segments.

    Conforms to the same Protocol as the FineWeb and Gutenberg adapters, which is what
    lets `ingestion.run` treat mined web text and downloaded corpora identically: one
    cleaning policy, one dedup pass across all sources, one split assignment, one
    manifest. A source that needed its own runner would be a second pipeline.
    """

    id = "cc"
    license = LICENSE

    def __init__(
        self,
        crawl: str,
        *,
        fmt: str = "wet",
        segments: int = 1,
        segment_paths_override: list[str] | None = None,
    ) -> None:
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        self.crawl = crawl
        self.fmt = fmt
        self.segments = segments
        self._override = segment_paths_override

    def plan(self) -> list[Segment]:
        """The segments this source will read. Resolved once, so a run is reproducible.

        Separated from `stream` on purpose: this list is the work unit for the Spark and
        Beam drivers, which partition segments across workers. Holding it as data rather
        than as a generator is what lets the same plan be executed serially here or
        distributed there and produce the same corpus.
        """
        if self._override is not None:
            return [Segment(crawl=self.crawl, path=path) for path in self._override]
        return segment_paths(self.crawl, fmt=self.fmt, limit=self.segments)

    def stream(self, limit: int | None = None) -> Iterator[RawRecord]:
        kept = 0
        for segment in self.plan():
            for record in stream_segment(segment, fmt=self.fmt):
                yield record
                kept += 1
                if limit is not None and kept >= limit:
                    return
