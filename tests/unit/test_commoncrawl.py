"""What the Common Crawl source must get right before it is pointed at 90,000 segments.

No network. Every test builds a real gzipped WARC in memory with warcio's own writer and
serves it through a fake `requests`, so what is exercised is the parsing and the framing
rather than a hand-written approximation of them.

The test that matters most is `test_a_truncated_segment_raises`. A dropped connection
produces a short gzip stream that decompresses cleanly up to the break, which is
indistinguishable from a segment that ended, and a mining run that quietly keeps 40% of a
segment produces a corpus size nobody can reproduce. That failure has no symptom, so it
needs a test rather than a code review.
"""

from __future__ import annotations

import io

import pytest

from forge.ingestion.commoncrawl import (
    CommonCrawlSource,
    CrawlStreamError,
    Segment,
    stream_segment,
)

warcio = pytest.importorskip("warcio", reason="warcio is in the [data] extra")

SEGMENT = Segment(crawl="CC-MAIN-2026-05", path="crawl-data/CC-MAIN-2026-05/segments/x.wet.gz")


def _wet(records: list[tuple[str, str, str | None]]) -> bytes:
    """A gzipped WET file: one warcinfo record then one conversion record per entry."""
    from warcio.warcwriter import WARCWriter

    buffer = io.BytesIO()
    writer = WARCWriter(buffer, gzip=True)
    writer.write_record(
        writer.create_warcinfo_record("x.wet.gz", {"software": "test"})
    )
    for uri, text, language in records:
        headers = {
            "WARC-Target-URI": uri,
            "WARC-Date": "2026-01-15T00:00:00Z",
            "WARC-Record-ID": f"<urn:uuid:{abs(hash(uri)) % 10**12}>",
        }
        if language is not None:
            headers["WARC-Identified-Content-Language"] = language
        writer.write_record(
            writer.create_warc_record(
                uri,
                "conversion",
                payload=io.BytesIO(text.encode()),
                warc_headers_dict=headers,
                http_headers=None,
            )
        )
    return buffer.getvalue()


class _Response:
    """`declared` is the Content-Length the server advertises.

    Kept separate from the body on purpose: a truncated download is exactly the case
    where the two disagree, and a fake that derived one from the other could not
    express it.
    """

    def __init__(self, body: bytes, declared: int | None) -> None:
        self.raw = io.BytesIO(body)
        self.status_code = 200
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def raise_for_status(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_) -> bool:
        return False


def _serve(monkeypatch, body: bytes, declared: int | None = -1) -> None:
    """Serve `body`. By default the Content-Length matches it, as a healthy server's does."""
    length = len(body) if declared == -1 else declared

    class _Requests:
        @staticmethod
        def get(url, **kwargs):
            return _Response(body, length)

    monkeypatch.setattr(
        "forge.ingestion.commoncrawl._requests", lambda: _Requests, raising=True
    )


def test_conversion_records_become_raw_records(monkeypatch):
    _serve(monkeypatch, _wet([
        ("https://a.example/page", "First document body.", "eng"),
        ("https://b.example/page", "Second document body.", "eng"),
    ]))

    records = list(stream_segment(SEGMENT))

    assert [r.text for r in records] == ["First document body.", "Second document body."]
    assert {r.source_id for r in records} == {"cc"}


def test_the_warcinfo_header_record_is_not_a_document(monkeypatch):
    # Every WET file opens with one. Counting it would inflate every per-segment count
    # by exactly one, which is small enough to never look wrong.
    _serve(monkeypatch, _wet([("https://a.example/", "Body.", None)]))

    assert len(list(stream_segment(SEGMENT))) == 1


def test_provenance_survives(monkeypatch):
    _serve(monkeypatch, _wet([("https://a.example/page", "Body.", "eng")]))

    record = next(iter(stream_segment(SEGMENT)))

    assert record.extra["url"] == "https://a.example/page"
    assert record.extra["segment"] == SEGMENT.path
    assert record.source_config == "CC-MAIN-2026-05"
    assert record.date == "2026-01-15T00:00:00Z"
    assert record.source_record_id and "<" not in record.source_record_id


def test_the_crawls_language_guess_is_provenance_and_not_a_score(monkeypatch):
    # WARC-Identified-Content-Language carries no confidence. upstream_language is a
    # (label, score) pair, so filling it here would mean inventing the score and writing
    # a fabricated number into corpus metadata. langid.py decides; the crawl only reports.
    _serve(monkeypatch, _wet([("https://a.example/", "Body.", "eng")]))

    record = next(iter(stream_segment(SEGMENT)))

    assert record.upstream_language is None
    assert record.extra["cc_language"] == "eng"


def test_empty_documents_are_dropped(monkeypatch):
    _serve(monkeypatch, _wet([
        ("https://a.example/", "   \n  ", None),
        ("https://b.example/", "Real body.", None),
    ]))

    assert [r.extra["url"] for r in stream_segment(SEGMENT)] == ["https://b.example/"]


def test_a_truncated_segment_raises(monkeypatch):
    """The whole point, and the test that found the defect this module now documents.

    Common Crawl gzips each record as its own member, so a stream that stops mid-member
    ends warcio's iteration CLEANLY. Nothing is raised and nothing is logged; the caller
    receives a short list of well-formed records. The server still declared the full
    length, so the byte count is what exposes it.
    """
    body = _wet([(f"https://{n}.example/", f"Body {n}." * 50, None) for n in range(20)])
    _serve(monkeypatch, body[: len(body) // 2], declared=len(body))

    with pytest.raises(CrawlStreamError, match="stopped mid-stream"):
        list(stream_segment(SEGMENT))


def test_truncation_is_silent_without_the_byte_check(monkeypatch):
    """Proof that the check is load-bearing rather than belt and braces.

    With verification off, the same truncated stream yields a plausible, short, entirely
    error-free result. If this test ever fails because warcio started raising, the guard
    can be simplified; until then it cannot.
    """
    body = _wet([(f"https://{n}.example/", f"Body {n}." * 50, None) for n in range(20)])
    _serve(monkeypatch, body[: len(body) // 2], declared=len(body))

    records = list(stream_segment(SEGMENT, verify_length=False))

    assert 0 < len(records) < 20


def test_a_server_that_declares_no_length_is_refused(monkeypatch):
    # Unverifiable completeness is not the same as verified completeness, and the
    # difference must not be decided silently.
    _serve(monkeypatch, _wet([("https://a.example/", "Body.", None)]), declared=None)

    with pytest.raises(CrawlStreamError, match="without a Content-Length"):
        list(stream_segment(SEGMENT))


def test_an_unknown_format_is_refused_before_any_request():
    with pytest.raises(ValueError, match="fmt must be one of"):
        CommonCrawlSource("CC-MAIN-2026-05", fmt="wat")


def test_an_explicit_segment_list_needs_no_index_fetch():
    # Reproducing a run means re-reading the same segments, not re-resolving "the first
    # n" against an index that may have been republished.
    source = CommonCrawlSource(
        "CC-MAIN-2026-05", segment_paths_override=["a/b/one.wet.gz", "a/b/two.wet.gz"]
    )

    assert [s.name for s in source.plan()] == ["one.wet.gz", "two.wet.gz"]


def test_stream_stops_at_the_limit(monkeypatch):
    _serve(monkeypatch, _wet([(f"https://{n}.example/", f"Body {n}.", None) for n in range(5)]))
    source = CommonCrawlSource("CC-MAIN-2026-05", segment_paths_override=["a/one.wet.gz"])

    assert len(list(source.stream(limit=3))) == 3


def test_a_urllib3_protocol_error_is_converted_not_leaked(monkeypatch):
    """The regression that killed a 284-segment run at minute 37.

    urllib3.exceptions.ProtocolError does not inherit from OSError. The first version of
    the guard listed (ArchiveLoadFailed, EOFError, OSError, BadGzipFile), so a dropped
    connection raised something no clause matched, escaped stream_segment without being
    converted, and went straight through the retry loop written for exactly this failure.
    The retry was fine. The exception list was wrong.
    """
    urllib3 = pytest.importorskip("urllib3")

    body = _wet([("https://a.example/", "Body." * 100, None)])

    class _Broken(io.BytesIO):
        def read(self, size=-1):
            raise urllib3.exceptions.ProtocolError(
                "Connection broken: IncompleteRead(62323511 bytes read, 3189908 more expected)"
            )

    class _Response:
        def __init__(self) -> None:
            self.raw = _Broken(body)
            self.status_code = 200
            self.headers = {"Content-Length": str(len(body))}

        def raise_for_status(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        "forge.ingestion.commoncrawl._requests",
        lambda: type("R", (), {"get": staticmethod(lambda url, **kw: _Response())}),
    )

    with pytest.raises(CrawlStreamError, match="failed to parse"):
        list(stream_segment(SEGMENT))


def test_the_transport_error_list_covers_the_library_actually_used():
    # A rename upstream would silently shrink this tuple back to the shape that leaked.
    urllib3 = pytest.importorskip("urllib3")
    from forge.ingestion.commoncrawl import TRANSPORT_ERRORS

    assert issubclass(urllib3.exceptions.ProtocolError, TRANSPORT_ERRORS)


def test_a_503_while_opening_becomes_a_retryable_error(monkeypatch):
    """The second run's failure: a 503 at minute 22, from OPENING the stream.

    The guard wrapped the read loop and not the open, so requests.HTTPError escaped
    unconverted and the per-segment retry never saw it. A 503 from Common Crawl is a
    transient overload response, which is precisely what the retry exists for.
    """
    requests = pytest.importorskip("requests")

    class _Response:
        status_code = 503
        headers: dict = {}

        def raise_for_status(self):
            raise requests.exceptions.HTTPError("503 Server Error: Service Unavailable")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        "forge.ingestion.commoncrawl._requests",
        lambda: type("R", (), {"get": staticmethod(lambda url, **kw: _Response())}),
    )

    with pytest.raises(CrawlStreamError, match="could not be opened"):
        list(stream_segment(SEGMENT))
