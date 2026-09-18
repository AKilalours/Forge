"""The reserve-pool scan, as functions no distributed runner is allowed to own.

WHY THIS FILE EXISTS SEPARATELY. The scan was written first as a Spark job, and everything
that decides whether it finishes ended up inside a module called spark_scan: how shards are
split across partitions, how many documents each partition may take, how many intra-op
threads it may ask for, and the per-process model cache. None of that is Spark. When the
same scan was ported to Beam, the choice was to import those from spark_scan, which would
have made a Beam pipeline depend on a module named after a different runner, or to copy
them, which would have produced two partitioning implementations that drift and then two
"speedups" measured over quietly different work. Both are worse than moving them here.

So: this module is the scan. forge.hard_negative.spark_scan and
forge.hard_negative.beam_scan are two ways of distributing it, and neither owns any
decision that affects the result. That is also what makes the two measurements comparable
at all, because the thing being measured is provably the same code.

THE MISTAKE THIS FILE IS SHAPED AROUND. The obvious implementation maps the scorer over
rows, which loads or serialises a 184M-parameter model per record. Both runners will
happily do that and the job simply never finishes, with no error to read. The model is
therefore loaded ONCE PER PROCESS and cached, and the scan is expressed per PARTITION
rather than per row. Every design choice below follows from that one constraint.

SCOPE, stated because it is the honest limit. Everything here has only ever run on one
machine, under Spark local mode and the Beam DirectRunner. It has never run on a cluster,
and the 5-million-document pool it is written for does not exist on disk: data/reserve/ is
gitignored and the corpus is not redistributed.
"""

from __future__ import annotations

import glob
import os
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

RESERVE_COLUMNS = ("doc_id", "source_group_id", "text", "domain", "source", "register")

# What a shard must have for the scan to read it at all.
REQUIRED_COLUMNS = ("doc_id", "source_group_id", "text")

# What a shard must NOT have, and this is a correctness boundary rather than a schema one.
# These columns are written by forge.generation: `generator` names the model that produced
# the text, `label` marks it as AI, `sample_id` is the generated document's key. The mining
# scan exists to find documents the detector scores as AI when they are HUMAN, so every hit
# is by construction a false positive. On a generated shard every hit is a TRUE positive,
# and mining those as hard negatives would train the detector to call its own synthetic
# text human. That is not a degraded measurement, it is the opposite of the intended one.
GENERATED_MARKER_COLUMNS = ("generator", "label", "sample_id")


class ScanError(RuntimeError):
    pass


@dataclass(frozen=True)
class PoolAudit:
    """What the pool actually contained, read from parquet footers before any worker starts.

    WHY A FINGERPRINT AND NOT A DOCUMENT COUNT. The Spark sweep and the Beam sweep are
    published side by side, and the only thing that makes that comparison legitimate is
    that they scanned the same work. The first version of the claim gate checked that both
    artifacts reported the same `documents_scanned`, which felt like the invariant and was
    not: the Spark sweep ran when data/silver held 6 human shards, and by the time the Beam
    sweep ran the v0.2-min regeneration had added 24 generated shards to the same tree.
    Both runs would have reported 40 documents. They would have been 40 documents from
    different corpora, and the gate would have passed.

    A count is a property of the cap. The fingerprint is a property of the pool.
    """

    files: tuple[str, ...]
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.files) != len(self.rows):
            raise ScanError("every audited file needs its row count")

    @property
    def total_rows(self) -> int:
        return sum(self.rows)

    @property
    def fingerprint(self) -> str:
        """sha256 over the sorted file list AND each file's row count.

        Paths alone would not notice a shard being rewritten in place with different
        content under the same name, which is exactly what a regeneration does.
        """
        import hashlib

        payload = "\n".join(f"{f}:{n}" for f, n in zip(self.files, self.rows, strict=True))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_record(self) -> dict[str, Any]:
        return {"n_files": len(self.files), "total_rows": self.total_rows,
                "fingerprint": self.fingerprint, "files": list(self.files)}


def audit_pool(files: Iterable[str]) -> PoolAudit:
    """Read every shard's footer and refuse the pool before a single worker starts.

    THE FAILURE THIS REPLACES. The column check used to live in _rows_from_files, inside
    the worker, at read time. Pointing the scan at data/silver after the v0.2-min
    regeneration therefore produced four partitions each dying separately on a generated
    shard, roughly two hundred lines of Beam traceback wrapping a one-line cause, and no
    artifact. ScanPlan already refuses an empty file list and an impossible partition count
    on the driver, on the stated principle that a plan which is obviously wrong should be
    visible before the compute is spent. The schema was the one part of "obviously wrong"
    that had been left to the workers.

    Footers only. Row counts and column names come from parquet metadata, so auditing
    thousands of shards costs a directory's worth of seeks and no row reads.
    """
    import pyarrow.parquet as pq

    audited: list[tuple[str, int]] = []
    generated: list[str] = []
    incomplete: list[tuple[str, list[str]]] = []
    unreadable: list[tuple[str, str]] = []

    for path in files:
        try:
            pf = pq.ParquetFile(path)
            names = set(pf.schema_arrow.names)
            rows = pf.metadata.num_rows
        except Exception as exc:                 # noqa: BLE001 - reported, not swallowed
            unreadable.append((path, type(exc).__name__))
            continue
        marker = sorted(names & set(GENERATED_MARKER_COLUMNS))
        if marker:
            generated.append(f"{path} (has {', '.join(marker)})")
            continue
        missing = sorted(set(REQUIRED_COLUMNS) - names)
        if missing:
            incomplete.append((path, missing))
            continue
        audited.append((path, rows))

    if generated:
        listed = "\n  ".join(generated)
        raise ScanError(
            f"{len(generated)} of {len(generated) + len(audited) + len(incomplete)} shards "
            f"under this root are GENERATED text, not a human pool:\n  {listed}\n"
            "The mining scan keeps documents the detector scores as AI when they are human, "
            "so a hit on a generated shard is a true positive and mining it as a hard "
            "negative would teach the detector to call its own synthetic text human. "
            "Narrow the scan with a pattern that selects only the human shards, for example "
            'pattern="source=*/split=*/*.parquet".'
        )
    if incomplete:
        listed = "\n  ".join(f"{p} is missing {m}" for p, m in incomplete)
        raise ScanError(f"{len(incomplete)} shards cannot be scanned:\n  {listed}")
    if unreadable:
        listed = "\n  ".join(f"{p} ({e})" for p, e in unreadable)
        raise ScanError(f"{len(unreadable)} shards could not be opened:\n  {listed}")

    return PoolAudit(files=tuple(f for f, _ in audited),
                     rows=tuple(n for _, n in audited))


@dataclass(frozen=True)
class ScanPlan:
    """How the scan will be laid out, decided before any executor starts.

    Separated from execution so the partitioning can be tested without Spark, and so a
    plan that is obviously wrong (one partition for four hundred shards, or four hundred
    partitions for one shard) is visible before an hour of compute goes into it.
    """

    files: tuple[str, ...]
    partitions: int
    threshold: float
    pool: PoolAudit | None = None

    def __post_init__(self) -> None:
        if not self.files:
            raise ScanError(
                "no parquet files to scan. The reserve pool is written by ingestion with "
                "with_reserve=True; without it there is nothing to mine."
            )
        if self.partitions < 1:
            raise ScanError("partitions must be >= 1")
        if not 0.0 < self.threshold < 1.0:
            raise ScanError(f"threshold {self.threshold} is not a probability")

    @property
    def files_per_partition(self) -> float:
        return len(self.files) / self.partitions


def plan_scan(root: str, partitions: int, threshold: float,
              pattern: str = "**/*.parquet", audit: bool = True) -> ScanPlan:
    """Build the plan. Files are SORTED, so two runs of the same pool scan in the same
    order and a re-run is comparable to the one before it.

    The pattern is a parameter because a root is not always a homogeneous pool. data/silver
    holds the human corpus under source=*/ and the generated arms under mirrors/ and
    random/, and the default recursive glob picks up all three. `audit` exists only so a
    test can build a plan over paths that are not real files; a run never turns it off.
    """
    files = tuple(sorted(glob.glob(f"{root}/{pattern}", recursive=True)))
    pool = audit_pool(files) if audit and files else None
    return ScanPlan(files=files, partitions=partitions, threshold=threshold, pool=pool)


def balanced_partitions(plan: ScanPlan) -> list[list[str]]:
    """Split files across partitions round-robin rather than in contiguous blocks.

    Contiguous blocks are the intuitive split and they are wrong here: parquet shards
    within a source are written in ingestion order and are correlated in size and content,
    so block N gets all the long documents and the job's wall clock becomes that one
    partition. Round-robin interleaves them. It does not balance perfectly, because
    nothing does without reading sizes first, but it removes the systematic skew.
    """
    buckets: list[list[str]] = [[] for _ in range(plan.partitions)]
    for i, f in enumerate(plan.files):
        buckets[i % plan.partitions].append(f)
    return [b for b in buckets if b]


@dataclass
class ScanStats:
    scanned: int = 0
    errors: int = 0
    above_threshold: int = 0
    partitions: int = 0
    files: int = 0
    wall_seconds: float = 0.0
    per_partition_seconds: list[float] = field(default_factory=list)

    @property
    def skew(self) -> float:
        """Slowest partition divided by the mean. 1.0 is perfect balance.

        Reported because it is the number that decides whether adding executors will help.
        A skew of 3 means two thirds of the cluster is idle waiting for one partition, and
        no amount of extra hardware fixes that; repartitioning does.
        """
        if not self.per_partition_seconds:
            return 1.0
        mean = sum(self.per_partition_seconds) / len(self.per_partition_seconds)
        return max(self.per_partition_seconds) / mean if mean else 1.0


def _rows_from_files(files: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Stream rows out of parquet without materialising a whole shard.

    iter_batches rather than read_table: a shard is read in record batches, so peak memory
    is one batch instead of one file. At 5M documents that difference is the job running
    or the executor being killed.
    """
    import pyarrow.parquet as pq

    for path in files:
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        missing = {"doc_id", "source_group_id", "text"} - available
        if missing:
            raise ScanError(f"{path} is missing required columns {sorted(missing)}")
        cols = [c for c in RESERVE_COLUMNS if c in available]
        for batch in pf.iter_batches(columns=cols, batch_size=256):
            yield from batch.to_pylist()



def partition_files(element_iter: Iterable[Any]) -> list[str]:
    """Unwrap whatever a runner hands a partition function into a flat list of paths.

    THE BUG THIS EXISTS TO PREVENT. The first version of the runner did
    `sc.parallelize([[bucket] for bucket in buckets])` and then `list(files_iter)[0]`:
    two levels of wrapping, one level of unwrapping. scan_partition received
    [["a.parquet", "b.parquet"]] and passed a LIST to pyarrow, which failed 14 seconds
    into a Spark job with "Cannot convert list to pyarrow.lib.NativeFile" buried in a
    Py4J traceback.

    Nothing caught it, because every test covered a pure function and the Spark path was
    left untested as "needs a JVM". The glue between the tested parts was the one piece
    with no test. So the glue is a pure function now, and it accepts either shape: a
    partition of path strings, or a partition holding one bucket of paths.
    """
    flat: list[str] = []
    for element in element_iter:
        if isinstance(element, str):
            flat.append(element)
        elif isinstance(element, (list, tuple)):
            for inner in element:
                if isinstance(inner, str):
                    flat.append(inner)
                elif isinstance(inner, (list, tuple)):
                    flat.extend(x for x in inner if isinstance(x, str))
                else:
                    raise ScanError(f"unexpected partition element {inner!r}")
        else:
            raise ScanError(f"unexpected partition element {element!r}")
    return flat

_SCORER_CACHE: dict[str, Any] = {}

# THE RACE THE SPARK RUNNER COULD NOT EXPOSE, found by porting to a threaded one.
#
# The cache below was written for Spark local mode, where every partition is a separate
# worker PROCESS. A plain `if arm not in cache` is safe there, because there is nothing to
# race with. Beam's local runner in multi_threading mode makes every partition a THREAD in
# one process, and four threads starting together all evaluate `arm not in cache` before
# any of them has finished loading. The measured result was four model loads at four
# partitions with the cache in place and doing nothing: on the stub that cost seconds, and
# with the real 184M parameter arm it is four copies of the weights resident at once, which
# on a laptop is the difference between a slow run and an OOM.
#
# It produced no error and no warning. It was visible only because scan_partition reports
# whether its own call paid for the load, which was added for a different reason entirely.
_SCORER_LOCK = threading.Lock()


def _acquire_scorer(arm: str) -> tuple[Any, bool]:
    """Return the scorer, and whether THIS call is the one that loaded it.

    THE REPORTING BUG THIS SHAPE EXISTS TO PREVENT, which is a second-order version of the
    race below and was measured before it was understood. The first attempt counted loads
    in a module-level integer and had each partition compare the count before and after
    acquiring its scorer. That reads as obviously correct and is not: a thread that blocked
    on the lock while another thread loaded the model sees the counter move and reports
    that IT paid for the load. Four partitions therefore reported four loads on a run where
    the lock had correctly performed exactly one, and the artifact said the cache was
    useless at the moment the cache started working.

    A before-and-after reading of shared state cannot answer a question about which caller
    did something. The answer has to come back from the operation itself, so it does.
    """
    # Fast path outside the lock: once the model is cached, every subsequent partition
    # reads it without contending. Serialising every scorer lookup would make the lock a
    # bottleneck on the very runner it exists for.
    cached = _SCORER_CACHE.get(arm)
    if cached is not None:
        return cached, False

    with _SCORER_LOCK:
        # Checked again inside the lock. This second check is the whole fix: the thread
        # that waited here must not load a model the thread ahead of it already loaded,
        # and it must not claim the load either.
        if arm in _SCORER_CACHE:
            return _SCORER_CACHE[arm], False

        from forge.inference.scorer import load_arm

        _SCORER_CACHE[arm] = load_arm(arm)
        return _SCORER_CACHE[arm], True


def _scorer_for(arm: str):
    """One scorer per executor PROCESS, not per partition and never per row.

    This cache is the whole performance story. A runner hands each partition to a worker
    that survives across partitions, so a module-level dict means the 184M parameter model
    is loaded once per process. Loading inside the row loop, which is what a naive map()
    does, turns a compute-bound job into a model-loading job that produces no error and
    never finishes.

    WHAT THE TWO RUNNERS DO DIFFERENTLY, and it is visible here rather than anywhere else.
    Spark local mode gives each partition its own Python worker process, so this dict is
    per process and the model is loaded once per worker. Beam's local runner in
    multi_threading mode gives each partition a THREAD in one process, so this same dict is
    shared and the model is loaded once for the whole job. That is not a bug in either
    runner and it is not something to normalise away; it is the single largest reason their
    startup costs differ, which is why scan_partition reports whether it was the call that
    paid for the load.

    It is also why this function is locked. See the comment above the cache: without the
    lock the threaded runner loaded the model once per thread and the cache was decorative.
    """
    return _acquire_scorer(arm)[0]


def scan_partition(files: list[str], arm: str, threshold: float,
                   limit: int | None = None,
                   torch_threads: int | None = None) -> Iterator[dict[str, Any]]:
    """Score one partition's shards and yield only the confident false positives.

    Only rows ABOVE the threshold cross the partition boundary. The reserve pool is human
    text, so every positive is by construction a false positive, and the fraction that
    clears a deployed threshold is small. Returning everything and filtering on the driver
    would move the entire pool through the network to discard almost all of it.

    A document that fails to score is counted and skipped, never silently dropped and
    never allowed to kill the partition. One malformed row in five million must not cost
    the run.
    """
    import time

    # OVERSUBSCRIPTION, which is the real ceiling on a single machine. The scan is
    # parallel twice over: Spark runs P partitions, and the torch forward inside each one
    # is itself multi-threaded. Left alone, P partitions times T intra-op threads asks for
    # P*T cores from a machine that has far fewer, and the threads spend their time
    # descheduling each other. The first sweep showed exactly this: skew stayed at 1.08,
    # so the partitions were balanced, while efficiency fell from 84% at two partitions to
    # 43% at four. Balanced work, saturated machine.
    #
    # Pinning intra-op threads per partition is the fix on one host. On a cluster it is
    # unnecessary, because each executor has its own cores, which is the difference
    # between this ceiling and a real one.
    if torch_threads is not None:
        import torch

        torch.set_num_threads(max(1, torch_threads))

    # Whether THIS call paid for the model load. Recorded, not inferred from timings,
    # because it is the one field that distinguishes a runner giving each partition its own
    # process from a runner giving each partition a thread in a shared one. It comes back
    # from _acquire_scorer rather than from a cache check or a load counter, because both
    # of those misattribute it under concurrency in opposite directions: a cache check
    # marks every racing thread as the first, and a counter marks every thread that waited
    # on the lock as the loader. _acquire_scorer is the only place that knows.
    scorer, paid_for_load = _acquire_scorer(arm)
    t0 = time.perf_counter()
    scanned = errors = hits = 0
    for row in _rows_from_files(files):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        if limit is not None and scanned >= limit:
            break
        scanned += 1
        try:
            score = scorer.score(text)
        except Exception:      # noqa: BLE001 - one bad row must not kill the partition
            errors += 1
            continue
        if score.mean >= threshold:
            hits += 1
            yield {
                "doc_id": row["doc_id"],
                "source_group_id": row["source_group_id"],
                "domain": row.get("domain", "unknown"),
                "source": row.get("source", "unknown"),
                "score_mean": float(score.mean),
                "score_max": float(score.maximum),
                "n_windows": int(score.n_windows),
            }
    # The trailing record carries this partition's own accounting. Without it the driver
    # knows how many hits came back and nothing about how many documents were examined,
    # so a partition that silently scanned nothing looks identical to one with no hits.
    yield {
        "doc_id": None, "_stats": True, "scanned": scanned, "errors": errors,
        "above_threshold": hits, "seconds": round(time.perf_counter() - t0, 3),
        "torch_threads": torch_threads, "loaded_model": paid_for_load,
        # WHICH PROCESS RAN THIS PARTITION. Recorded because whether a runner gives a
        # partition a process or a thread is the single fact that most changes how its
        # timings should be read, and it is the fact this code has been wrong about twice:
        # once in a docstring, once in the model-load column. Two runs of the same sweep
        # can now be told apart from their artifacts instead of from a claim about a
        # runner's defaults.
        "pid": os.getpid(),
    }


# ---------------------------------------------------------------------------
# THE INVARIANT, and the placement mistake that made it library code.
#
# These three lived in scripts/spark_scan_reserve.py, where they were only ever reachable
# from a command line. The Beam runner needs the identical arithmetic, and a second copy of
# docs_per_partition is exactly how two runners end up scanning different numbers of
# documents while both printing a speedup. Untested script bodies are also why none of them
# had a test; as library functions they do.
# ---------------------------------------------------------------------------


def docs_per_partition(total_docs: int | None, partitions: int) -> int | None:
    """Split a FIXED total across partitions, so every partition count does the same work.

    This is the third place in this repository that needs this invariant.
    forge.training.scaling holds the global batch fixed across world sizes. DistributedRun
    holds step semantics fixed across strategies. Here it is total documents fixed across
    partition counts, and now across runners as well.

    The first version of the Spark script took --max-docs-per-partition, so four partitions
    scanned four times as many documents as one. Wall time would have stayed roughly flat,
    the speedup would have read as 1.0, and the honest conclusion "Spark does not help"
    would have been drawn from a run that quietly did four times the work. A scaling number
    computed over a workload that grows with the parallelism is not a scaling number.
    """
    if total_docs is None:
        return None
    return -(-total_docs // partitions)      # ceiling, so no document is dropped


def threads_per_partition(partitions: int, override: int | None = None) -> int:
    """Divide the machine's cores among the partitions instead of letting them collide.

    Runner parallelism multiplies with torch's own intra-op threading. Without this, four
    partitions each asking for four threads want sixteen cores from a machine that has
    fewer, and the measured result was efficiency dropping to 43% at a skew of 1.08:
    perfectly balanced partitions on a completely saturated machine.
    """
    import os

    if override is not None:
        return max(1, override)
    return max(1, (os.cpu_count() or 1) // max(1, partitions))


def is_reserve_pool(root: str) -> bool:
    """A mining round may only read the reserve pool. Path-based and deliberately strict:
    a check that tries to be clever here is a check that eventually says yes to training
    data.

    Both runner scripts call this before deciding whether to emit candidate ids. A scan of
    data/silver finds documents the model was TRAINED on, where a "false positive" on
    memorised text says nothing about production behaviour, so those runs publish timing
    and nothing that could later be mistaken for a mining round.
    """
    import pathlib

    return pathlib.Path(root).resolve().name == "reserve"
