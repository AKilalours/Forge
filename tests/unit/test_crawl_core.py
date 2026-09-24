"""The crawl ingest's decisions, tested on real WET files and the real cleaner.

No network: every segment here is a gzipped WARC written to tmp_path with warcio's own
writer, read back through `stream_segment`'s local-file path. The cleaner is the real one,
run under the real v0.1-min policy, so a document that survives here survives for the
reasons it would in production.

The failures this module can produce do not crash. A wrong survivor rule, a shard that
depends on the executor, a training document let into the reserve pool, a manifest that
hashes the wrong ids: each one writes a plausible corpus. So the tests assert on the
corpus, and especially on `corpus_fingerprint`, which is what makes "same input, same
output" a single comparison.
"""

from __future__ import annotations

import io
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from forge.cleaning.pipeline import Cleaner, CleaningPolicy
from forge.ingestion.commoncrawl import CrawlStreamError, stream_segment
from forge.ingestion.crawl_core import (
    CrawlError,
    Resolution,
    build_report,
    clean_segment,
    corpus_fingerprint,
    fingerprint_segments,
    is_stats,
    load_policy,
    merge_stats,
    partition_segments,
    plan_crawl,
    resolve_group,
    run_serial,
    shard_of,
    survivor,
    training_hashes,
    write_crawl_manifest,
)
from forge.ingestion.sources import RawRecord

pytest.importorskip("warcio", reason="warcio is in the [data] extra")
pq = pytest.importorskip("pyarrow.parquet")

ROOT = Path(__file__).resolve().parents[2]
CRAWL = "CC-MAIN-2026-34"

BASE = (
    "The harbour authority published its annual dredging schedule in March, listing "
    "eleven separate operations across the estuary and the two tidal basins. Silt "
    "accumulation had increased noticeably since the previous survey, and the deeper "
    "berths required attention before the autumn shipping season began in earnest. "
    "Contractors were appointed in two lots, with the smaller basins handled under a "
    "framework agreement that had been renewed the year before. A short consultation "
    "period followed, during which several fishing associations objected to the "
    "proposed timing on the grounds that it overlapped with the spawning season. "
    "Officials agreed to move the work on the northern channel by three weeks, and the "
    "revised plan was circulated to every berth holder along with a map of the areas "
    "that would be closed. Survey vessels returned in the summer to measure the depth "
    "again, and their readings were compared against the figures from the spring. "
)

TOPICS = {
    "unique_a": "The council later confirmed that the railway footbridge would be repainted.",
    "unique_b": "A local choir performed at the reopening of the library in late October.",
    "shared": "Walkers were reminded that the coastal path is closed at high tide.",
    "training": "The ferry timetable changed twice that winter because of storms.",
}
FRENCH = (
    "Le conseil municipal a approuvé le nouveau budget pour la rénovation des écoles, "
    "et les travaux commenceront au printemps prochain dans plusieurs quartiers. " * 8
)


def _doc(topic: str) -> str:
    return BASE + TOPICS[topic]


def _write_wet(path: Path, records: list[tuple[str, str]]) -> str:
    """A gzipped WET file with deterministic record ids, as `(record_id, text)` pairs."""
    from warcio.warcwriter import WARCWriter

    with open(path, "wb") as handle:
        writer = WARCWriter(handle, gzip=True)
        writer.write_record(writer.create_warcinfo_record(path.name, {"software": "test"}))
        for record_id, text in records:
            writer.write_record(writer.create_warc_record(
                f"https://example.org/{record_id}",
                "conversion",
                payload=io.BytesIO(text.encode()),
                warc_headers_dict={
                    "WARC-Target-URI": f"https://example.org/{record_id}",
                    "WARC-Date": "2026-08-07T10:00:00Z",
                    "WARC-Record-ID": f"<urn:uuid:{record_id}>",
                },
            ))
    return str(path)


def _cleaned_hash(text: str, policy: CleaningPolicy) -> str:
    """The content hash the cleaner assigns to `text`, which is what training stores."""
    record = RawRecord(
        source_id="fw", source_record_id="x", text=text, license="ODC-By-1.0",
        domain="web", register="informational", acquired_at=datetime.now(timezone.utc),
    )
    return next(iter(Cleaner(policy, dedup=False).process([record]))).content_sha256


@pytest.fixture
def policy() -> CleaningPolicy:
    return load_policy(ROOT / "configs" / "data" / "human_minimal.yaml")


@pytest.fixture
def segments(tmp_path) -> list[str]:
    """Two segments. `shared` appears in both; `training` is in the training corpus."""
    return [
        _write_wet(tmp_path / "seg-a.warc.wet.gz", [
            ("0a", _doc("unique_a")),
            ("9s", _doc("shared")),
            ("5t", _doc("training")),
        ]),
        _write_wet(tmp_path / "seg-b.warc.wet.gz", [
            ("1b", _doc("unique_b")),
            ("2s", _doc("shared")),
            ("3f", FRENCH),
        ]),
    ]


@pytest.fixture
def training_root(tmp_path, policy) -> Path:
    """A training corpus laid out as the writer lays it out, holding one known document."""
    import pyarrow as pa

    root = tmp_path / "silver"
    shard = root / "source=fw" / "split=train"
    shard.mkdir(parents=True)
    pq.write_table(
        pa.table({"content_sha256": [_cleaned_hash(_doc("training"), policy)]}),
        shard / "part-000.parquet",
    )
    return root


def _written_rows(out: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(out.glob("source=*/split=*/*.parquet")):
        rows.extend(pq.ParquetFile(path).read(columns=["doc_id", "content_sha256"]).to_pylist())
    return rows


# --------------------------------------------------------------------------- the policy

def test_the_policy_is_the_training_policy_not_the_default(policy):
    # The measurement that sized this pipeline first ran under CleaningPolicy(), which
    # admits 50 to 20,000 tokens. The models were trained on 150 to 400. A reserve pool
    # built under the default would be full of documents longer than anything the model
    # saw, and its "false positives" would measure that, not the detector.
    default = CleaningPolicy()

    assert (policy.length.min_chars, policy.length.min_tokens, policy.length.max_tokens) == (
        800, 150, 400)
    assert policy.length != default.length


# ----------------------------------------------------------------------------- the plan

def test_the_fingerprint_does_not_depend_on_segment_order():
    assert fingerprint_segments(["a/x.gz", "a/y.gz"]) == fingerprint_segments(["a/y.gz", "a/x.gz"])


def test_a_local_segment_fingerprints_by_name_so_two_machines_agree():
    assert fingerprint_segments(["/Users/one/seg.gz"]) == fingerprint_segments(["/tmp/two/seg.gz"])


def test_a_segment_listed_twice_is_refused(segments):
    # Otherwise dedup removes every document of that segment as a duplicate of itself
    # and the duplicate count reports an artefact of the plan as a property of the crawl.
    with pytest.raises(CrawlError, match="repeats"):
        plan_crawl(CRAWL, paths=[segments[0], segments[0]])


def test_more_partitions_than_segments_is_refused(segments):
    with pytest.raises(CrawlError, match="idle"):
        plan_crawl(CRAWL, paths=segments, partitions=3)


def test_partitioning_is_round_robin():
    plan = plan_crawl(CRAWL, paths=[f"/s/{i}.gz" for i in range(5)], partitions=2)

    assert partition_segments(plan) == [["/s/0.gz", "/s/2.gz", "/s/4.gz"], ["/s/1.gz", "/s/3.gz"]]


# ---------------------------------------------------------------------------- stage one

def test_cleaning_a_segment_ends_with_its_stats(segments, policy):
    plan = plan_crawl(CRAWL, paths=segments)

    rows = list(clean_segment(segments[1], plan, policy))

    assert rows[-1]["record_type"] == "stats"
    assert rows[-1]["dedup_applied"] is False
    assert rows[-1]["rejected"].get("language") == 1
    assert [r["record_type"] for r in rows[:-1]] == ["document", "document"]


# ---------------------------------------------------------------------------- stage two

def test_the_survivor_does_not_depend_on_arrival_order():
    # The rule that lets two runners agree. Every order a shuffle could deliver.
    group = [{"doc_id": d, "content_sha256": "h"} for d in ("cc_c", "cc_a", "cc_b")]

    winners = {survivor(list(order))["doc_id"] for order in itertools.permutations(group)}

    assert winners == {"cc_a"}


def test_a_training_document_loses_every_copy_and_is_counted_as_overlap():
    rows = [{"doc_id": "cc_a", "content_sha256": "h"}, {"doc_id": "cc_b", "content_sha256": "h"}]

    resolution = resolve_group(rows, excluded=frozenset({"h"}))

    assert resolution == Resolution(kept=None, duplicates=0, overlaps_training=2)


def test_duplicates_are_counted_apart_from_training_overlap():
    rows = [{"doc_id": "cc_b", "content_sha256": "h"}, {"doc_id": "cc_a", "content_sha256": "h"}]

    resolution = resolve_group(rows, excluded=frozenset())

    assert resolution.kept["doc_id"] == "cc_a"
    assert (resolution.duplicates, resolution.overlaps_training) == (1, 0)


def test_no_training_corpus_means_no_reserve_pool(tmp_path):
    # An empty exclusion set would let every training page through and look exactly like
    # a crawl that happened not to overlap with training.
    with pytest.raises(CrawlError, match="Refusing"):
        training_hashes(tmp_path)


def test_shard_placement_is_a_function_of_the_content():
    """Several hashes with known placements, not one.

    The first version checked a single hash, and a mutant that sharded by process id
    passed it by luck: one pid in four lands in the right shard. Four different expected
    shards cannot all be matched by a value that is the same for every row.
    """
    expected = {"00000000": 0, "00000001": 1, "00000006": 2, "0000000b": 3}

    placed = {prefix: shard_of({"content_sha256": prefix + "0" * 56}, 4) for prefix in expected}

    assert placed == expected


# -------------------------------------------------------------------------- end to end

def test_serial_run_counts_reconcile(segments, policy, training_root, tmp_path):
    plan = plan_crawl(CRAWL, paths=segments, shards=2)

    report = run_serial(plan, tmp_path / "out", policy, training_hashes(training_root))

    # 6 records: 1 French rejected, 5 kept. Of those, one of the two `shared` copies is a
    # duplicate and the `training` page is overlap, so 3 are written.
    assert report["documents_seen"] == 6
    assert report["documents_kept_before_dedup"] == 5
    assert report["duplicates_removed"] == 1
    assert report["overlaps_training"] == 1
    assert report["documents_written"] == 3
    assert report["policy"]["max_tokens"] == 400


def test_the_training_page_is_not_in_the_reserve_pool(segments, policy, training_root, tmp_path):
    excluded = training_hashes(training_root)
    run_serial(plan_crawl(CRAWL, paths=segments), tmp_path / "out", policy, excluded)

    written = {row["content_sha256"] for row in _written_rows(tmp_path / "out")}

    assert not written & excluded


def test_the_shared_page_survives_once_as_the_lowest_doc_id(segments, policy, training_root, tmp_path):
    run_serial(plan_crawl(CRAWL, paths=segments), tmp_path / "out", policy,
               training_hashes(training_root))

    ids = sorted(row["doc_id"] for row in _written_rows(tmp_path / "out"))

    # "2s" < "9s": the copy from the SECOND segment wins, because the rule is about the
    # document and not about which segment was read first.
    assert "cc_urn:uuid:2s" in ids and "cc_urn:uuid:9s" not in ids


def test_segment_order_does_not_change_the_corpus(segments, policy, training_root, tmp_path):
    excluded = training_hashes(training_root)

    forward = run_serial(plan_crawl(CRAWL, paths=segments, shards=3),
                         tmp_path / "f", policy, excluded)
    backward = run_serial(plan_crawl(CRAWL, paths=segments[::-1], shards=3),
                          tmp_path / "b", policy, excluded)

    assert forward["corpus_fingerprint"] == backward["corpus_fingerprint"]
    assert sorted(p.relative_to(tmp_path / "f") for p in (tmp_path / "f").rglob("*.parquet")) == \
        sorted(p.relative_to(tmp_path / "b") for p in (tmp_path / "b").rglob("*.parquet"))


def test_every_shard_gets_its_own_file(segments, policy, training_root, tmp_path):
    # With the writer's default part name, two shards in one split would both write
    # part-000.parquet and the second would silently replace the first.
    report = run_serial(plan_crawl(CRAWL, paths=segments, shards=8), tmp_path / "out",
                        policy, training_hashes(training_root))

    assert len(_written_rows(tmp_path / "out")) == report["documents_written"]


def test_the_manifest_hashes_the_real_ids(segments, policy, training_root, tmp_path):
    """Regression: the first draft split identity lines on ':'.

    A Common Crawl doc_id is "cc_urn:uuid:...", so every id in the manifest came out as
    "cc_urn": the count was right, the hash of ids was a hash of one repeated string,
    and nothing failed.
    """
    from forge.common.hashing import content_sha256

    out = tmp_path / "out"
    report = run_serial(plan_crawl(CRAWL, paths=segments), out, policy,
                        training_hashes(training_root))
    identities = [
        f"{row['doc_id']}\t{row['content_sha256']}\tx" for row in _written_rows(out)
    ]

    manifest = json.loads(write_crawl_manifest(out, "v0.2-reserve", report, identities).read_text())

    ids = sorted(row["doc_id"] for row in _written_rows(out))
    assert manifest["sources"][0]["sha256_of_ids"] == content_sha256("\n".join(ids))
    assert len(set(ids)) == manifest["counts"]["human"] == 3


def test_a_report_that_does_not_add_up_is_refused(segments, policy):
    plan = plan_crawl(CRAWL, paths=segments)
    stats = merge_stats([{"seen": 10, "kept": 5, "segment": "s", "pid": 1}])

    with pytest.raises(CrawlError, match="do not reconcile"):
        build_report(
            runner="test", plan=plan, policy=policy, stats=stats,
            resolutions=[Resolution(kept=None, duplicates=1, overlaps_training=0)],
            shard_results=[{"shard": 0, "rows": 2, "identity": ["a\th\ttrain", "b\th\ttrain"]}],
            excluded=frozenset(), elapsed=1.0, out="x",
        )


def test_corpus_fingerprint_ignores_order():
    assert corpus_fingerprint(["b", "a"]) == corpus_fingerprint(["a", "b"])


# ------------------------------------------------------------------ transient failures

def _flaky(failures: int):
    """A stream_segment that raises CrawlStreamError `failures` times, then succeeds."""
    state = {"calls": 0}
    real = stream_segment

    def wrapped(segment, **kwargs):
        state["calls"] += 1
        if state["calls"] <= failures:
            # Yield one record first, so the partial output of a failed attempt is real
            # and would be visible if it were not discarded.
            yield next(iter(real(segment, **kwargs)))
            raise CrawlStreamError("stopped mid-stream (simulated)")
        yield from real(segment, **kwargs)

    return wrapped, state


def test_a_segment_that_fails_once_is_retried(monkeypatch, segments, policy):
    # An hour-long run meets a dropped connection eventually. Without a retry, fifty
    # minutes of good work is lost to one bad minute.
    wrapped, state = _flaky(1)
    monkeypatch.setattr("forge.ingestion.crawl_core.stream_segment", wrapped)
    plan = plan_crawl(CRAWL, paths=segments)

    rows = list(clean_segment(segments[0], plan, policy))

    assert state["calls"] == 2
    assert rows[-1]["failed"] is False and rows[-1]["attempts"] == 2


def test_a_failed_attempt_emits_nothing(monkeypatch, segments, policy):
    """The reason an attempt's output is buffered rather than streamed.

    A retry cannot resume mid-segment, so records from the failed attempt would be read
    twice. Dedup would keep one copy of each and the double read would leave no trace,
    making the corpus smaller than the counts claim with nothing to show for it.
    """
    wrapped, _ = _flaky(1)
    monkeypatch.setattr("forge.ingestion.crawl_core.stream_segment", wrapped)
    plan = plan_crawl(CRAWL, paths=segments)

    documents = [r for r in clean_segment(segments[0], plan, policy) if not is_stats(r)]

    assert len(documents) == len({d["doc_id"] for d in documents})


def test_a_segment_that_never_recovers_is_recorded_not_raised(monkeypatch, segments, policy,
                                                              training_root, tmp_path):
    wrapped, _ = _flaky(99)
    monkeypatch.setattr("forge.ingestion.crawl_core.stream_segment", wrapped)
    plan = plan_crawl(CRAWL, paths=segments)

    report = run_serial(plan, tmp_path / "out", policy, training_hashes(training_root), )

    # 283 good segments must not be lost to one bad one, and the loss must be visible:
    # a corpus built from fewer segments than planned is not reproducible from the plan.
    assert report["segments_read"] == 0
    assert len(report["segments_failed"]) == 2
    assert "stopped mid-stream" in report["segments_failed"][0]["error"]
