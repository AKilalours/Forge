"""The crawl ingest on Beam. Distribution only; every decision is in `crawl_core`.

Same three stages as `spark_crawl`, expressed as a pipeline graph:

    Create(buckets) -> Reshuffle -> FlatMap(clean_segment)          stage 1
        |-> Filter(stats) --------------------------------------> stats sink
        '-> KeyBy(hash) -> GroupByKey -> Map(resolve_group)      stage 2
                |-> counts ------------------------------------> counts sink
                '-> KeyBy(shard) -> GroupByKey -> Map(write)     stage 3 -> shards sink

WHY BEAM DOES NOT NEED SPARK'S PERSIST. A Beam pipeline is a graph built once and executed
once. The cleaned PCollection is consumed by two branches, and the runner materialises it
for both; nothing is recomputed per consumer. The failure Spark's `persist` prevents does
not exist here, which is a real difference between the two models and worth being able to
say out loud.

WHY THE RESULTS GO THROUGH TEXT SINKS. Under `multi_processing` the transforms run in
other processes, so a list appended to in a DoFn stays empty on the driver. The reserve
scan hit this and solved it the same way; see `scripts/beam_scan_reserve.py`.

THE RUNNER IS PINNED, via `hard_negative.beam_scan.beam_pipeline_options`, for the reason
written up there: `--runner=DirectRunner` hands the pipeline to Prism, which ignores the
worker count.
"""

from __future__ import annotations

import json
import tempfile
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


def _read_json_lines(prefix: Path) -> list:
    rows = []
    for path in sorted(prefix.parent.glob(prefix.name + "*")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def run_beam(plan: CrawlPlan, out: str | Path, policy: CleaningPolicy,
             excluded: frozenset[str], running_mode: str = "multi_processing") -> dict:
    import apache_beam as beam

    from forge.hard_negative.beam_scan import beam_pipeline_options

    started = time.perf_counter()
    out = str(out)
    sinks = Path(tempfile.mkdtemp(prefix="forge-beam-crawl-"))
    buckets = list(enumerate(partition_segments(plan)))

    with beam.Pipeline(options=beam_pipeline_options(plan.partitions, running_mode)) as pipeline:
        cleaned = (
            pipeline
            | "Buckets" >> beam.Create(buckets)
            # Without this the runner is free to fuse Create with the FlatMap below and run
            # every bucket in one worker. Correct output, serial execution, and a throughput
            # figure that measures nothing. Found the hard way in the reserve scan.
            | "Spread" >> beam.Reshuffle()
            | "Paths" >> beam.FlatMap(lambda item: item[1])
            | "Clean" >> beam.FlatMap(clean_segment, plan=plan, policy=policy)
        )

        (
            cleaned
            | "OnlyStats" >> beam.Filter(is_stats)
            | "StatsJson" >> beam.Map(json.dumps)
            | "StatsSink" >> beam.io.WriteToText(str(sinks / "stats"))
        )

        resolved = (
            cleaned
            | "OnlyDocs" >> beam.Filter(lambda row: not is_stats(row))
            | "ByHash" >> beam.Map(lambda row: (dedup_key(row), row))
            | "GroupHash" >> beam.GroupByKey()
            | "Resolve" >> beam.Map(lambda item, ex: resolve_group(item[1], ex), ex=excluded)
        )

        (
            resolved
            | "Counts" >> beam.Map(lambda r: json.dumps([r.duplicates, r.overlaps_training]))
            | "CountsSink" >> beam.io.WriteToText(str(sinks / "counts"))
        )

        (
            resolved
            | "Survivors" >> beam.Filter(lambda r: r.kept is not None)
            | "ByShard" >> beam.Map(lambda r: (shard_of(r.kept, plan.shards), r.kept))
            | "GroupShard" >> beam.GroupByKey()
            | "Write" >> beam.Map(lambda item: write_shard(item[0], item[1], out, policy))
            | "ShardJson" >> beam.Map(json.dumps)
            | "ShardSink" >> beam.io.WriteToText(str(sinks / "shards"))
        )

    counts = _read_json_lines(sinks / "counts")
    return build_report(
        runner=f"beam-{running_mode}", plan=plan, policy=policy,
        stats=merge_stats(_read_json_lines(sinks / "stats")),
        resolutions=[Resolution(kept=None,
                                duplicates=sum(c[0] for c in counts),
                                overlaps_training=sum(c[1] for c in counts))],
        shard_results=_read_json_lines(sinks / "shards"), excluded=excluded,
        elapsed=time.perf_counter() - started, out=out,
    )
