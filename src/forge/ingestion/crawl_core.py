"""The crawl ingest, written once, so Spark and Beam can each distribute it.

Same arrangement as `hard_negative.scan_core`, for the same reason. Everything that
decides what the corpus contains lives here; a runner module decides only how the work is
spread across workers. If a decision that changes the output can only be found by reading
`spark_crawl.py`, then the two runners are not running the same job and comparing them
means nothing. `run_serial` is the reference: both runners are tested against it and must
produce a corpus with the same `corpus_fingerprint`.

THE SHAPE OF THIS JOB, AND THE ONE STAGE THAT IS NOT EMBARRASSINGLY PARALLEL.

    segments -> clean each one independently        (parallel, no coordination)
             -> deduplicate across ALL of them      (a shuffle on content hash)
             -> shard and write                     (a second shuffle, then parallel)

Cleaning is a pure function of one document, so a worker needs nothing but its segment.
Deduplication is not: it is a function of a document and every other document in the
corpus. A worker that deduplicates its own partition produces a partition with no
internal duplicates and says nothing about the pages it shares with the other workers.
Nothing raises. You get duplicate pages spread across train and test, which is the leak
`cleaning.pipeline`'s fixed stage order exists to prevent, and it makes every metric
better than it should be. That is why `Cleaner` has a `dedup=False` mode rather than this
module growing its own copy of the cleaning sequence.

THREE RULES THAT MAKE THE OUTPUT A PROPERTY OF THE INPUT, NOT OF THE EXECUTION.

1. The survivor of a duplicate group is chosen by a total order over the documents
   (`survivor`), never "first seen". After a shuffle there is no first.
2. A document's output shard is a function of its content hash (`shard_of`), never of
   which worker happened to hold it.
3. Rows inside a shard are sorted by doc_id before writing.

Together these mean Spark, Beam and the serial reference write the same documents to the
same files. The tests hold them to it.

THE RESERVE POOL MUST EXCLUDE TRAINING DATA. FineWeb is derived from Common Crawl, so a
page in a new crawl can be the same page as a training document. Mining scans the reserve
pool for confident false positives; a "false positive" on text the model was trained on
measures memorisation, not production behaviour. `training_hashes` loads the content
hashes of the training corpus and the dedup stage drops any match, counted separately as
`overlaps_training` so the report says how many there were rather than hiding them inside
the duplicate count.

THE POLICY MUST BE THE TRAINING POLICY. The models were trained on v0.1-min, whose config
sets 800 chars and 150 to 400 tokens. `CleaningPolicy()` defaults to 200 chars and 50 to
20,000 tokens. A reserve pool cleaned under the defaults would contain documents far
longer than anything the model saw, and a false positive on one of those would say
"out of distribution", not "the detector is wrong". So the policy is always loaded from a
config (`load_policy`) and recorded in the report. The same class of mistake is written
up in `ingestion.run.policy_from_config`.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.cleaning.pipeline import Cleaner, CleaningPolicy
from forge.ingestion.commoncrawl import (
    FORMATS,
    CommonCrawlSource,
    CrawlStreamError,
    Segment,
    stream_segment,
)

SOURCE_ID = "cc"


class CrawlError(RuntimeError):
    """A crawl ingest cannot proceed, with a reason a reader can act on."""


# --------------------------------------------------------------------------------- plan

@dataclass(frozen=True)
class CrawlPlan:
    """Exactly which segments this run reads, pinned before any work starts.

    Pinned rather than resolved per worker for the reason `scan_core.PoolAudit` exists:
    two runners can only be compared over the same input, and "the first 20 segments of
    the current crawl" is not the same input twice if the index is republished between
    runs. The fingerprint is what a report cites.
    """

    crawl: str
    fmt: str
    segments: tuple[str, ...]
    partitions: int
    shards: int
    fingerprint: str

    def as_record(self) -> dict:
        return {
            "crawl": self.crawl,
            "format": self.fmt,
            "segments": len(self.segments),
            "partitions": self.partitions,
            "shards": self.shards,
            "fingerprint": self.fingerprint,
        }


def fingerprint_segments(paths: Iterable[str]) -> str:
    """sha256 over the sorted segment paths, first 16 hex characters.

    Sorted, so the identity of the input does not depend on the order the index listed
    it in. Paths rather than contents: a Common Crawl segment is immutable at its path,
    which is what makes this cheap and still meaningful. A local path is fingerprinted by
    its basename, so the same segments downloaded to two machines fingerprint the same.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(p).name if p.startswith("/") else p for p in paths):
        digest.update(path.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def plan_crawl(
    crawl: str,
    segments: int = 1,
    fmt: str = "wet",
    partitions: int = 1,
    shards: int = 1,
    paths: list[str] | None = None,
) -> CrawlPlan:
    """Resolve the segment list once, on the driver, before any worker starts."""
    if fmt not in FORMATS:
        raise CrawlError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    if partitions < 1 or shards < 1:
        raise CrawlError(f"partitions and shards must be at least 1, got {partitions}, {shards}")

    source = CommonCrawlSource(crawl, fmt=fmt, segments=segments, segment_paths_override=paths)
    resolved = [segment.path for segment in source.plan()]
    if not resolved:
        raise CrawlError(f"crawl {crawl!r} resolved to no {fmt} segments")
    if len(set(resolved)) != len(resolved):
        # Reading a segment twice doubles its documents, and dedup would then quietly
        # remove every one of them as a duplicate of itself, inflating the duplicate
        # count with an artefact of the plan.
        raise CrawlError("the segment list contains repeats; each segment must be read once")
    if partitions > len(resolved):
        raise CrawlError(
            f"{partitions} partitions over {len(resolved)} segments would leave workers "
            f"idle and make the throughput number meaningless. Ask for at most "
            f"{len(resolved)} partitions, or plan more segments."
        )
    return CrawlPlan(
        crawl=crawl,
        fmt=fmt,
        segments=tuple(resolved),
        partitions=partitions,
        shards=shards,
        fingerprint=fingerprint_segments(resolved),
    )


def partition_segments(plan: CrawlPlan) -> list[list[str]]:
    """Round robin, not contiguous blocks.

    Segment sizes vary by a factor of several, and contiguous slices of an index ordered
    by capture time put similar segments together. Round robin spreads that variance, so
    skew in a throughput report is a statement about the runner, not about the crawl.
    """
    buckets: list[list[str]] = [[] for _ in range(plan.partitions)]
    for index, path in enumerate(plan.segments):
        buckets[index % plan.partitions].append(path)
    return buckets


# ------------------------------------------------------------------------------- policy

def load_policy(config: str | Path) -> CleaningPolicy:
    """The cleaning policy the training corpus was built with, read from its config.

    Through `policy_from_config`, not re-implemented here, because that function carries
    the fix for the bug where config values were silently ignored.
    """
    from forge.common.config import load
    from forge.ingestion.run import policy_from_config

    return policy_from_config(load(str(config)))


def policy_record(policy: CleaningPolicy) -> dict:
    """What the report says was enforced, so a reader can check it against the config."""
    return {
        "min_chars": policy.length.min_chars,
        "min_tokens": policy.length.min_tokens,
        "max_tokens": policy.length.max_tokens,
        "language": policy.language,
        "language_score_min": policy.language_score_min,
    }


# ----------------------------------------------------------------------- stage 1: clean

def clean_segment(path: str, plan: CrawlPlan, policy: CleaningPolicy,
                  retries: int = 2) -> Iterator[dict]:
    """Every document one segment yields, cleaned but NOT deduplicated.

    `policy` is required, not defaulted. See the module docstring: the default policy is
    not the training policy, and a default here is how the difference would go unnoticed.

    Yields plain dicts rather than HumanDocument because a pydantic model has to cross a
    worker boundary by pickle in Spark and by a coder in Beam, and a schema that only
    survives one of those is a schema that silently differs between runners.

    The last item yielded is a stats record, marked by `record_type`. A worker that
    reports nothing about what it rejected leaves the pipeline unable to tell "this
    segment was mostly non-English" from "this segment failed".
    """
    segment = Segment(crawl=plan.crawl, path=path)
    started = time.perf_counter()

    # RETRIED, AND BUFFERED UNTIL THE SEGMENT COMPLETES. Two things forced this shape.
    #
    # A 284-segment run is an hour of continuous streaming, and one dropped connection
    # raises CrawlStreamError, which without a retry takes the whole run with it after
    # fifty minutes of work. The completeness check is right to be loud; it just must not
    # be fatal on the first try.
    #
    # And a retry cannot resume mid-segment: the records already yielded from the failed
    # attempt would be yielded again by the next one, and they would not be duplicates of
    # each other in the dedup stage either, because dedup keeps one copy of each hash and
    # would simply hide the double read. So an attempt's output is held until the segment
    # finishes, and a failed attempt's partial output is discarded rather than emitted.
    # A segment is a couple of thousand kept documents at most, so the memory is bounded.
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        cleaner = Cleaner(policy, dedup=False)
        rows: list[dict] = []
        try:
            for document in cleaner.process(stream_segment(segment, fmt=plan.fmt)):
                row = document.model_dump(mode="json", by_alias=True)
                row["record_type"] = "document"
                rows.append(row)
        except CrawlStreamError as error:
            last_error = error
            time.sleep(min(2 ** attempt, 8))
            continue

        yield from rows
        stats = cleaner.stats.as_dict()
        stats.update({
            "record_type": "stats",
            "segment": path,
            "kept": len(rows),
            "attempts": attempt + 1,
            "failed": False,
            "seconds": round(time.perf_counter() - started, 3),
            "pid": os.getpid(),
            "fingerprint": plan.fingerprint,
        })
        yield stats
        return

    # Out of retries. The run continues, because losing 283 good segments to one bad one
    # is worse, but the failure is recorded per segment and the CLI exits non-zero. A
    # corpus built from fewer segments than were planned must say so: its size is not
    # reproducible from the plan alone.
    yield {
        "record_type": "stats",
        "segment": path,
        "seen": 0,
        "kept": 0,
        "rejected": {},
        "dedup_applied": False,
        "attempts": retries + 1,
        "failed": True,
        "error": f"{type(last_error).__name__}: {last_error}",
        "seconds": round(time.perf_counter() - started, 3),
        "pid": os.getpid(),
        "fingerprint": plan.fingerprint,
    }


def is_stats(row: dict) -> bool:
    return row.get("record_type") == "stats"


# ----------------------------------------------------------------------- stage 2: dedup

def training_hashes(root: str | Path) -> frozenset[str]:
    """Content hashes of every document in the training corpus under `root`.

    Read with `pq.ParquetFile(...).read(columns=...)` rather than `pq.read_table(path)`:
    the writer lays the corpus out as source=*/split=* AND stores physical source and
    split columns, and read_table applies hive partition discovery to a single file path,
    which collides string against dictionary types. `training/data.py` hit that and
    switched to the same call.
    """
    import pyarrow.parquet as pq

    from forge.ingestion.writer import PARTITION_GLOB

    files = sorted(Path(root).glob(PARTITION_GLOB))
    if not files:
        raise CrawlError(
            f"no training shards match {PARTITION_GLOB} under {root}. Refusing to build "
            f"a reserve pool that cannot be checked for overlap with training data."
        )
    hashes: set[str] = set()
    for path in files:
        column = pq.ParquetFile(path).read(columns=["content_sha256"]).column(0)
        hashes.update(column.to_pylist())
    return frozenset(hashes)


def dedup_key(row: dict) -> str:
    """What the shuffle groups on: the content hash, computed post-redaction.

    Post-redaction on purpose. Two pages that differ only in an email address are the
    same document once the address is a token, and hashing before redaction keeps both.
    """
    return row["content_sha256"]


def survivor(rows: Iterable[dict]) -> dict:
    """One document from a group of exact duplicates, chosen without reference to order.

    Lowest doc_id wins. Any total order over the documents would do; what matters is that
    it is a function of the DATA and not of the order the shuffle delivered. The
    sequential cleaner keeps "the first", which is well defined only because it saw an
    order. Using that here would let Spark and Beam write different corpora from the same
    segments with neither being wrong.
    """
    return min(rows, key=lambda row: row["doc_id"])


@dataclass(frozen=True)
class Resolution:
    """The outcome for one content hash: what survives, and why the rest did not."""

    kept: dict | None
    duplicates: int
    overlaps_training: int


def resolve_group(rows: Iterable[dict], excluded: frozenset[str]) -> Resolution:
    """Apply both dedup rules to every copy of one content hash.

    A hash found in the training corpus loses every copy, and those copies are counted as
    training overlap, not as duplicates: the two numbers answer different questions (how
    redundant is the crawl, and how much of it the model has already seen), and folding
    one into the other would make both unreadable.
    """
    group = list(rows)
    if not group:
        return Resolution(kept=None, duplicates=0, overlaps_training=0)
    if dedup_key(group[0]) in excluded:
        return Resolution(kept=None, duplicates=0, overlaps_training=len(group))
    return Resolution(kept=survivor(group), duplicates=len(group) - 1, overlaps_training=0)


# ----------------------------------------------------------------------- stage 3: write

def shard_of(row: dict, shards: int) -> int:
    """Output shard from the content hash, so placement never depends on the executor."""
    return int(dedup_key(row)[:8], 16) % shards


def part_name(shard: int) -> str:
    return f"part-{shard:05d}"


def finalise(rows: Iterable[dict]) -> list[Any]:
    """Rebuild HumanDocument objects from surviving rows, validated at the write boundary.

    A row that has been through two serialisation boundaries is not the object that left
    the worker, and the writer's schema contract is worth re-checking at the one point
    where the corpus becomes a file.
    """
    from forge.common.schemas import HumanDocument

    return [
        HumanDocument.model_validate({k: v for k, v in row.items() if k != "record_type"})
        for row in rows
    ]


def write_shard(shard: int, rows: Iterable[dict], root: str | Path, policy: CleaningPolicy) -> dict:
    """Validate, sort and write one shard. Returns what the driver needs for the manifest.

    `assert_corpus_matches_policy` runs here, on the written documents, for the reason it
    exists in `ingestion.run`: rejection counts describe the policy that RAN, and only the
    output can show whether that was the policy that was ASKED for.
    """
    from forge.ingestion.run import assert_corpus_matches_policy
    from forge.ingestion.writer import write_parquet

    ordered = sorted(rows, key=lambda row: row["doc_id"])
    documents = finalise(ordered)
    assert_corpus_matches_policy(documents, policy)
    written = write_parquet(documents, root, part=part_name(shard)) if documents else {}
    return {
        "shard": shard,
        "rows": len(documents),
        "partitions": written,
        # Tab-separated: a Common Crawl doc_id is "cc_urn:uuid:...", so a colon separator
        # would split the id itself and every id in the manifest would read "cc_urn".
        "identity": [f"{d.doc_id}\t{d.content_sha256}\t{d.split.value}" for d in documents],
    }


# ----------------------------------------------------------------------------- report

def corpus_fingerprint(identities: Iterable[str]) -> str:
    """One hash over every written (doc_id, content hash, split), independent of order.

    The field the runner tests compare. Equal fingerprints mean the same documents with
    the same text in the same splits, which is the claim "these runners are equivalent"
    reduced to one string.
    """
    digest = hashlib.sha256()
    for line in sorted(identities):
        digest.update(line.encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def merge_stats(records: Iterable[dict]) -> dict:
    """Fold the per-segment stats records into one."""
    seen = kept = 0
    rejected: Counter = Counter()
    segments: list[str] = []
    failed: list[dict] = []
    pids: set[int] = set()
    seconds: list[float] = []

    for record in records:
        if record.get("failed"):
            failed.append({"segment": record.get("segment"), "error": record.get("error")})
        seen += record.get("seen", 0)
        kept += record.get("kept", 0)
        segments.append(record.get("segment", ""))
        pids.add(record.get("pid", -1))
        seconds.append(record.get("seconds", 0.0))
        rejected.update(record.get("rejected") or {})

    return {
        "segments_read": len([s for s in segments if s]) - len(failed),
        "segments_failed": failed,
        "documents_seen": seen,
        "documents_kept_before_dedup": kept,
        "rejected": dict(sorted(rejected.items())),
        "keep_rate_before_dedup": round(kept / seen, 4) if seen else 0.0,
        "distinct_worker_pids": len(pids - {-1}),
        "segment_seconds_max": round(max(seconds), 3) if seconds else 0.0,
        "segment_seconds_sum": round(sum(seconds), 3),
    }


def build_report(
    *,
    runner: str,
    plan: CrawlPlan,
    policy: CleaningPolicy,
    stats: dict,
    resolutions: Iterable[Resolution],
    shard_results: Iterable[dict],
    excluded: frozenset[str],
    elapsed: float,
    out: str | Path,
) -> dict:
    """Everything a reader needs to check this run, in one record.

    Counts must reconcile: kept_before_dedup = written + duplicates + overlaps_training.
    The function raises if they do not, because a report that does not add up is a
    report of a pipeline that lost documents somewhere, and publishing it would publish
    that loss as a result.
    """
    duplicates = overlaps = 0
    for resolution in resolutions:
        duplicates += resolution.duplicates
        overlaps += resolution.overlaps_training

    shard_results = list(shard_results)
    identities = [line for result in shard_results for line in result["identity"]]
    by_split = Counter(line.rsplit("\t", 1)[1] for line in identities)
    written = len(identities)

    before = stats["documents_kept_before_dedup"]
    if before != written + duplicates + overlaps:
        raise CrawlError(
            f"counts do not reconcile: {before} kept before dedup, but {written} written "
            f"+ {duplicates} duplicates + {overlaps} training overlaps = "
            f"{written + duplicates + overlaps}. A document was lost or double counted."
        )

    return {
        "runner": runner,
        "plan": plan.as_record(),
        "policy": policy_record(policy),
        **stats,
        "duplicates_removed": duplicates,
        "overlaps_training": overlaps,
        "training_hashes_checked": len(excluded),
        "documents_written": written,
        "by_split": dict(sorted(by_split.items())),
        "shards_written": sum(1 for r in shard_results if r["rows"]),
        "dedup_applied": True,
        "corpus_fingerprint": corpus_fingerprint(identities),
        "elapsed_seconds": round(elapsed, 3),
        "documents_seen_per_second": round(stats["documents_seen"] / elapsed, 1) if elapsed else 0.0,
        "out": str(out),
    }


def write_crawl_manifest(root: str | Path, dataset_version: str, report: dict,
                         identities: Iterable[str]) -> Path:
    """MANIFEST.json in the same shape `writer.write_manifest` produces.

    Built from ids rather than from HumanDocument objects, because the distributed
    runners never hold the whole corpus in one process and must not have to.
    """
    import json
    from datetime import datetime, timezone

    from forge.common.hashing import content_sha256
    from forge.common.splits import SPLIT_SALT
    from forge.ingestion.writer import _code_commit

    ids = sorted(line.split("\t", 1)[0] for line in identities)
    manifest = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _code_commit(),
        "spec_version": "data_spec_v1",
        "processing_version": "clean_v1",
        "split_salt": SPLIT_SALT,
        "sources": [{
            "id": SOURCE_ID,
            "n_docs": len(ids),
            "sha256_of_ids": content_sha256("\n".join(ids)),
            "cleaning": {k: report[k] for k in (
                "documents_seen", "documents_kept_before_dedup", "rejected",
                "duplicates_removed", "overlaps_training",
            )},
        }],
        "crawl": report["plan"],
        "policy": report["policy"],
        "corpus_fingerprint": report["corpus_fingerprint"],
        "counts": {
            "human": len(ids),
            "ai": 0,
            "hard_negative": 0,
            "adversarial": 0,
            "by_split": report["by_split"],
        },
    }
    path = Path(root) / "MANIFEST.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- reference

def run_serial(plan: CrawlPlan, out: str | Path, policy: CleaningPolicy,
               excluded: frozenset[str]) -> dict:
    """The reference implementation. One process, no framework, same three stages.

    Not a fallback for production: it holds the whole cleaned corpus in memory. It exists
    so that "Spark and Beam are correct" can mean "they agree with this", which is short
    enough to read in full.
    """
    started = time.perf_counter()
    groups: dict[str, list[dict]] = {}
    stats_records: list[dict] = []

    for path in plan.segments:
        for row in clean_segment(path, plan, policy):
            if is_stats(row):
                stats_records.append(row)
            else:
                groups.setdefault(dedup_key(row), []).append(row)

    resolutions = [resolve_group(rows, excluded) for rows in groups.values()]
    shards: dict[int, list[dict]] = {}
    for resolution in resolutions:
        if resolution.kept is not None:
            shards.setdefault(shard_of(resolution.kept, plan.shards), []).append(resolution.kept)

    results = [write_shard(i, rows, out, policy) for i, rows in sorted(shards.items())]
    return build_report(
        runner="serial", plan=plan, policy=policy, stats=merge_stats(stats_records),
        resolutions=resolutions, shard_results=results, excluded=excluded,
        elapsed=time.perf_counter() - started, out=out,
    )
