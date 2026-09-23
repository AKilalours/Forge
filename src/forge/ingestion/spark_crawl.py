"""The crawl ingest on Spark. Distribution only; every decision is in `crawl_core`.

    segments --flatMap(clean_segment)--> rows      stage 1, one task per segment bucket
             --keyBy(hash).groupByKey--> groups     stage 2, the shuffle dedup needs
             --map(resolve_group)------> survivors
             --keyBy(shard).groupByKey-> shards     stage 3, placement by content hash
             --map(write_shard)--------> files

WHY `cleaned` IS PERSISTED, AND WHAT HAPPENS IF IT IS NOT. Spark is lazy and recomputes
an RDD for every action that needs it. This job has more than one action over the cleaned
rows: the stats are collected, and the dedup chain runs. Without `persist`, the second
action re-runs stage 1 from scratch, and stage 1 is not a cheap map, it is streaming every
segment from Common Crawl again. Nothing fails. The run takes twice as long, downloads the
crawl twice, and reports a throughput that is half the truth. MEMORY_AND_DISK rather than
MEMORY_ONLY because half a million cleaned documents does not fit in executor memory on a
laptop, and a MEMORY_ONLY partition that is evicted is recomputed, which is the same bug.

The later stages reuse Spark's shuffle files, which Spark keeps across jobs on its own, so
only the stage before the first shuffle needs this.
"""

from __future__ import annotations

import time
from pathlib import Path

from forge.cleaning.pipeline import CleaningPolicy
from forge.ingestion.crawl_core import (
    CrawlPlan,
    Resolution,
    build_report,
    clean_segment,
    dedup_key,
    is_stats,
    merge_stats,
    partition_segments,
    resolve_group,
    shard_of,
    write_shard,
)


def spark_session(partitions: int):
    """local[P], the same shape as the reserve scan's session.

    THE WORKER INTERPRETER IS PINNED TO THIS ONE, through PYSPARK_PYTHON. Left unset,
    Spark starts its Python workers with whatever `python3` is first on PATH, which need
    not be the interpreter running the driver. Outside an activated venv that is the
    system or conda base Python: the workers fail to import forge, or worse, import a
    different pyarrow than the driver and disagree about schemas. Found by the runner
    equivalence test, whose workers died with "No module named 'forge'" while the driver
    imported it fine.

    The environment variable and not the `spark.pyspark.python` config key. The first
    version of this fix set the config key, and the workers still reported
    /usr/bin/python3: in local mode PySpark's SparkContext reads PYSPARK_PYTHON when it
    starts and never consults that key. setdefault, so an explicit PYSPARK_PYTHON still
    wins.
    """
    import os
    import sys

    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    return (
        SparkSession.builder
        .appName(f"forge-crawl-p{partitions}")
        .master(f"local[{partitions}]")
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def run_spark(plan: CrawlPlan, out: str | Path, policy: CleaningPolicy,
              excluded: frozenset[str], spark=None) -> dict:
    from pyspark import StorageLevel

    own_session = spark is None
    spark = spark or spark_session(plan.partitions)
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    started = time.perf_counter()
    out = str(out)

    # Broadcast, not closed over: the training hashes are the same for every task, and a
    # closure would ship a copy with each one.
    excluded_bc = sc.broadcast(excluded)
    buckets = partition_segments(plan)

    try:
        cleaned = (
            # One slice per bucket, so partition i holds exactly bucket i and the round
            # robin assignment survives into the executors.
            sc.parallelize(buckets, len(buckets))
            .flatMap(lambda bucket: bucket)
            .flatMap(lambda path: clean_segment(path, plan, policy))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        stats = merge_stats(cleaned.filter(is_stats).collect())

        resolved = (
            cleaned.filter(lambda row: not is_stats(row))
            .keyBy(dedup_key)
            .groupByKey()
            .map(lambda item: resolve_group(item[1], excluded_bc.value))
        )
        # Only the counts come back to the driver. Collecting the resolutions themselves
        # would ship every surviving document to one process.
        duplicates, overlaps = resolved.map(
            lambda r: (r.duplicates, r.overlaps_training)
        ).fold((0, 0), lambda a, b: (a[0] + b[0], a[1] + b[1]))

        results = (
            resolved.filter(lambda r: r.kept is not None)
            .map(lambda r: (shard_of(r.kept, plan.shards), r.kept))
            .groupByKey(numPartitions=max(1, min(plan.shards, plan.partitions)))
            .map(lambda item: write_shard(item[0], item[1], out, policy))
            .collect()
        )
        cleaned.unpersist()
    finally:
        if own_session:
            spark.stop()

    return build_report(
        runner="spark", plan=plan, policy=policy, stats=stats,
        resolutions=[Resolution(kept=None, duplicates=duplicates, overlaps_training=overlaps)],
        shard_results=results, excluded=excluded,
        elapsed=time.perf_counter() - started, out=out,
    )
