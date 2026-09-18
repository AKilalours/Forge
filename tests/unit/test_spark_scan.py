"""The Spark scan's decisions, tested without Spark.

Everything that decides whether this job finishes is a pure function: how files are split
across partitions, whether the plan is coherent, and whether a scorer is constructed once
or once per row. Those are the things worth pinning, and none of them need a JVM.

The one test that does need Spark is marked and skips cleanly, because pyspark needs Java
and a machine without it should say so rather than fail.
"""

from __future__ import annotations

import pytest

from forge.hard_negative.spark_scan import (
    ScanPlan,
    ScanStats,
    SparkScanError,
    balanced_partitions,
    plan_scan,
)


def _plan(n_files: int, partitions: int) -> ScanPlan:
    return ScanPlan(files=tuple(f"s{i}.parquet" for i in range(n_files)),
                    partitions=partitions, threshold=0.99)


def test_a_plan_with_no_files_fails_before_a_cluster_is_started() -> None:
    """An empty reserve pool is the likeliest real failure: data/reserve/ is gitignored,
    so a fresh checkout has nothing to scan. Failing at plan time costs nothing; failing
    after the executors are up costs whatever the cluster bills."""
    with pytest.raises(SparkScanError, match="with_reserve=True"):
        ScanPlan(files=(), partitions=4, threshold=0.99)


def test_a_threshold_outside_zero_to_one_is_refused() -> None:
    with pytest.raises(SparkScanError, match="not a probability"):
        _plan(4, 2).__class__(files=("a.parquet",), partitions=1, threshold=1.5)


def test_files_are_interleaved_across_partitions_not_blocked() -> None:
    """THE SKEW REGRESSION.

    Contiguous blocks are the intuitive split. They are wrong here: shards inside one
    source are written in ingestion order and are correlated in size, so a block split
    hands one partition all the long documents and the job's wall clock becomes that
    partition. Round-robin interleaving removes the systematic part of that skew.
    """
    buckets = balanced_partitions(_plan(8, 4))
    assert buckets == [
        ["s0.parquet", "s4.parquet"],
        ["s1.parquet", "s5.parquet"],
        ["s2.parquet", "s6.parquet"],
        ["s3.parquet", "s7.parquet"],
    ]


def test_more_partitions_than_files_does_not_produce_empty_partitions() -> None:
    """An empty partition still costs a task launch and, worse here, a model load."""
    buckets = balanced_partitions(_plan(3, 8))
    assert len(buckets) == 3
    assert all(buckets), "no partition may be empty"


def test_every_file_is_scanned_exactly_once() -> None:
    """Losing a shard silently under-reports the scan; duplicating one double-counts the
    hits. Both look like a successful run."""
    plan = _plan(17, 5)
    flat = [f for bucket in balanced_partitions(plan) for f in bucket]
    assert sorted(flat) == sorted(plan.files)
    assert len(flat) == len(set(flat))


def test_plan_scan_sorts_so_two_runs_are_comparable(tmp_path) -> None:
    """The shards are REAL parquet now, not empty files.

    They used to be `write_bytes(b"")`, which was enough when plan_scan only globbed. It
    now audits every footer on the driver, because the schema check used to live in the
    worker and a mixed pool therefore failed four partitions deep instead of before the
    run. An empty file is not parquet, so the cheap fixture was also the one that would
    have hidden the audit.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    for name in ("b.parquet", "a.parquet", "c.parquet"):
        pq.write_table(pa.table({"doc_id": ["d"], "source_group_id": ["g"],
                                 "text": ["t"]}), tmp_path / name)
    plan = plan_scan(str(tmp_path), partitions=2, threshold=0.9, pattern="*.parquet")
    assert [p.split("/")[-1] for p in plan.files] == ["a.parquet", "b.parquet", "c.parquet"]
    assert plan.pool is not None, "a real run must carry the pool it audited"


def test_skew_is_reported_because_it_decides_whether_more_executors_help() -> None:
    balanced = ScanStats(per_partition_seconds=[10.0, 10.0, 10.0, 10.0])
    skewed = ScanStats(per_partition_seconds=[2.0, 2.0, 2.0, 30.0])
    assert balanced.skew == pytest.approx(1.0)
    assert skewed.skew == pytest.approx(30.0 / 9.0)
    assert ScanStats().skew == 1.0, "no measurements must not divide by zero"


def test_the_scorer_is_cached_per_process_and_not_rebuilt_per_call() -> None:
    """THE PERFORMANCE REGRESSION, and the one that produces no error.

    Constructing the scorer inside the row loop is what a naive map() does. Spark runs it
    without complaint and the job becomes a model-loading job that never finishes. This
    pins the cache so the mistake cannot be reintroduced by someone tidying the code.
    """
    from forge.hard_negative import spark_scan

    calls = {"n": 0}

    class FakeArm:
        pass

    def fake_load_arm(arm: str) -> FakeArm:
        calls["n"] += 1
        return FakeArm()

    import sys
    import types

    stub = types.ModuleType("forge.inference.scorer")
    stub.load_arm = fake_load_arm
    real = sys.modules.get("forge.inference.scorer")
    sys.modules["forge.inference.scorer"] = stub
    spark_scan._SCORER_CACHE.clear()
    try:
        first = spark_scan._scorer_for("baseline")
        second = spark_scan._scorer_for("baseline")
        third = spark_scan._scorer_for("mirror")
    finally:
        spark_scan._SCORER_CACHE.clear()
        if real is not None:
            sys.modules["forge.inference.scorer"] = stub and real
        else:
            del sys.modules["forge.inference.scorer"]

    assert first is second, "the same arm must not be loaded twice in one process"
    assert third is not first, "different arms are different models"
    assert calls["n"] == 2, f"expected one load per arm, got {calls['n']}"


# ---------------------------------------------------------------------------
# THE REGRESSION THAT ACTUALLY HAPPENED, and the reason it got through.
#
# Eight tests above pass and all eight cover pure functions. The Spark path was left
# untested on the grounds that it "needs a JVM". The piece that broke was neither: it was
# the glue between them, which had no test because it was not obviously either kind.
#
# sc.parallelize([[bucket] for bucket in buckets]) wraps each bucket in a further list,
# and the mapPartitions function then did list(iter)[0], which unwraps once. Two wraps,
# one unwrap. scan_partition got [["a.parquet"]] and passed a LIST to pyarrow, failing 14
# seconds into a Spark job with "Cannot convert list to pyarrow.lib.NativeFile" at the
# bottom of a Py4J traceback.
#
# The lesson is not "test Spark". It is that "needs a JVM" was doing more work than it
# should have: it excused leaving the shape-handling untested too. partition_files is now
# a pure function, so the shape is testable even though the cluster is not.
# ---------------------------------------------------------------------------


def test_partition_files_accepts_a_partition_of_plain_paths() -> None:
    from forge.hard_negative.spark_scan import partition_files

    assert partition_files(iter(["a.parquet", "b.parquet"])) == ["a.parquet", "b.parquet"]


def test_partition_files_accepts_a_partition_holding_one_bucket() -> None:
    """The shape sc.parallelize(buckets) actually produces: one element per partition,
    that element being the bucket."""
    from forge.hard_negative.spark_scan import partition_files

    assert partition_files(iter([["a.parquet", "b.parquet"]])) == ["a.parquet", "b.parquet"]


def test_partition_files_survives_the_double_wrapping_that_broke_the_first_run() -> None:
    """The exact shape that failed. Flattened rather than passed through, so the same
    mistake in a future runner produces paths instead of a pyarrow TypeError."""
    from forge.hard_negative.spark_scan import partition_files

    assert partition_files(iter([[["a.parquet", "b.parquet"]]])) == ["a.parquet", "b.parquet"]


def test_partition_files_is_empty_for_an_empty_partition() -> None:
    from forge.hard_negative.spark_scan import partition_files

    assert partition_files(iter([])) == []


def test_partition_files_refuses_a_shape_it_does_not_understand() -> None:
    """Guessing is what produced a list where a path belonged. An unknown shape fails
    here, on the driver's terms, rather than inside pyarrow on an executor."""
    from forge.hard_negative.spark_scan import partition_files

    with pytest.raises(SparkScanError, match="unexpected partition element"):
        partition_files(iter([{"not": "a path"}]))
