"""The reserve-pool scan, as a Spark job, for when the pool stops fitting on one machine.

WHY SPARK HERE AND NOWHERE ELSE IN THIS REPO. Phase 1 cleaning runs on Polars and PyArrow
because 400k documents fit in memory on one machine and Spark would add operational cost
for no throughput. docs/jd_coverage.md argues that, and it is still right. The mining scan
is the one job with a different shape: it reads a reserve pool sized in millions, runs an
independent forward pass per document, and keeps the small fraction the detector is
confidently wrong about. No shuffle, no join, no cross-document state. Embarrassingly
parallel over shards, and bounded by compute rather than by memory.

forge.hard_negative.reserve.load_reserve is the single-machine version: it globs every
parquet file and materialises every document into one Python list. That is correct at
400k and impossible at 5M. This module is the same scan without that constraint.

THE MISTAKE THIS FILE IS SHAPED AROUND. The obvious implementation maps the scorer over
rows, which loads or serialises a 184M-parameter model per record. Spark will happily do
that and the job simply never finishes, with no error to read. The model is therefore
loaded ONCE PER PARTITION, inside mapPartitions, and cached per executor process. Every
design choice below follows from that one constraint.

SCOPE, stated because it is the honest limit. This has been run in Spark LOCAL mode over
the parquet in this repository. It has never run on a cluster, and the 5-million-document
pool it is written for does not exist on disk: data/reserve/ is gitignored and the corpus
is not redistributed. What is measured is the job and its scaling across local partitions.
What is NOT claimed is a cluster run.
"""

from __future__ import annotations

import glob
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

RESERVE_COLUMNS = ("doc_id", "source_group_id", "text", "domain", "source", "register")


class SparkScanError(RuntimeError):
    pass


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

    def __post_init__(self) -> None:
        if not self.files:
            raise SparkScanError(
                "no parquet files to scan. The reserve pool is written by ingestion with "
                "with_reserve=True; without it there is nothing to mine."
            )
        if self.partitions < 1:
            raise SparkScanError("partitions must be >= 1")
        if not 0.0 < self.threshold < 1.0:
            raise SparkScanError(f"threshold {self.threshold} is not a probability")

    @property
    def files_per_partition(self) -> float:
        return len(self.files) / self.partitions


def plan_scan(root: str, partitions: int, threshold: float,
              pattern: str = "**/*.parquet") -> ScanPlan:
    """Build the plan. Files are SORTED, so two runs of the same pool scan in the same
    order and a re-run is comparable to the one before it."""
    files = tuple(sorted(glob.glob(f"{root}/{pattern}", recursive=True)))
    return ScanPlan(files=files, partitions=partitions, threshold=threshold)


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
            raise SparkScanError(f"{path} is missing required columns {sorted(missing)}")
        cols = [c for c in RESERVE_COLUMNS if c in available]
        for batch in pf.iter_batches(columns=cols, batch_size=256):
            yield from batch.to_pylist()



def partition_files(element_iter: Iterable[Any]) -> list[str]:
    """Unwrap what Spark hands a mapPartitions function into a flat list of paths.

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
                    raise SparkScanError(f"unexpected partition element {inner!r}")
        else:
            raise SparkScanError(f"unexpected partition element {element!r}")
    return flat

_SCORER_CACHE: dict[str, Any] = {}


def _scorer_for(arm: str):
    """One scorer per executor PROCESS, not per partition and never per row.

    This cache is the whole performance story. Spark hands each partition to a worker
    process that survives across partitions, so a module-level dict means the 184M
    parameter model is loaded once per process. Loading inside the row loop, which is what
    a naive map() does, turns a compute-bound job into a model-loading job that produces
    no error and never finishes.
    """
    if arm not in _SCORER_CACHE:
        from forge.inference.scorer import load_arm

        _SCORER_CACHE[arm] = load_arm(arm)
    return _SCORER_CACHE[arm]


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

    scorer = _scorer_for(arm)
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
        "torch_threads": torch_threads,
    }
