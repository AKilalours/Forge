"""Run the reserve-pool scan on Spark, and measure how it partitions.

    python scripts/spark_scan_reserve.py --partitions 1
    python scripts/spark_scan_reserve.py --partitions 4
    python scripts/spark_scan_reserve.py --summarise

This is the Spark half of a pair; scripts/beam_scan_reserve.py is the other, written to
the same shape so the two tables are comparable.

WHAT THIS MEASURES. Throughput and partition skew for the mining scan, in Spark LOCAL
mode, on this machine. It does not measure a cluster, and this repository has never run
one. The reserve pool the job is written for (5M documents) is not on disk: data/reserve/
is gitignored because the corpus is not redistributed.

WHY IT REFUSES TO MINE FROM data/silver. The only parquet in a checkout is the training
pool. Scanning it finds documents the model was TRAINED on, and a "false positive" on
memorised text says nothing about production behaviour. forge.hard_negative.mining is
built around the reserve pool being disjoint from training for exactly this reason. So
when the root is not a reserve pool this script writes timing and nothing else: no
candidate ids, no parquet of hits, nothing that could be mistaken later for a mining
round. The measurement is legitimate; the mining would not be.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# THE SHARED CORE. These three used to be defined here, and the Beam runner needs exactly
# the same arithmetic: the same fixed-total document cap, the same division of cores among
# partitions, the same strict reserve-pool check. A second copy is how two runners end up
# scanning different numbers of documents while both printing a speedup, so they live in
# forge.hard_negative.scan_core and both scripts import them. Their reasoning is documented
# there.
from forge.hard_negative.scan_core import (
    docs_per_partition,
    is_reserve_pool,
    threads_per_partition,
)

OUT = Path("reports/experiments/spark")
DEFAULT_ROOT = "data/reserve"


def run(root: str, arm: str, threshold: float, partitions: int,
        total_docs: int | None, torch_threads: int | None = None) -> dict:
    from pyspark.sql import SparkSession

    from forge.hard_negative.scan_core import (
        balanced_partitions,
        partition_files,
        plan_scan,
        scan_partition,
    )

    plan = plan_scan(root, partitions=partitions, threshold=threshold)
    buckets = balanced_partitions(plan)
    mining_allowed = is_reserve_pool(root)
    per_partition_cap = docs_per_partition(total_docs, len(buckets))
    threads = threads_per_partition(len(buckets), torch_threads)

    spark = (
        SparkSession.builder
        .appName(f"forge-reserve-scan-p{partitions}")
        .master(f"local[{partitions}]")
        # One Python worker per task and no reuse across tasks would reload the model for
        # every partition. Reuse is what makes the per-process scorer cache mean anything.
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    def work(element_iter):
        files = partition_files(element_iter)
        if not files:
            return iter(())
        return scan_partition(files, arm, threshold, limit=per_partition_cap,
                              torch_threads=threads)

    t0 = time.perf_counter()
    # One bucket per element, one element per partition. The earlier version wrapped each
    # bucket in a further list and then unwrapped once, which handed pyarrow a list.
    rdd = sc.parallelize(buckets, len(buckets))
    rows = rdd.mapPartitions(work).collect()
    wall = time.perf_counter() - t0
    spark.stop()

    stats = [r for r in rows if r.get("_stats")]
    hits = [r for r in rows if not r.get("_stats")]
    scanned = sum(s["scanned"] for s in stats)
    per_partition = [s["seconds"] for s in stats]
    mean = sum(per_partition) / len(per_partition) if per_partition else 0.0

    result = {
        "mode": "spark local",
        "root": root,
        "is_reserve_pool": mining_allowed,
        "mining_candidates_written": False,
        "why": (
            "Timing only. The root is not the reserve pool, so any hit here is a document "
            "the model was trained on and is not a mining candidate."
            if not mining_allowed else
            "Reserve pool scan; candidates are valid mining input."
        ),
        "arm": arm,
        "threshold": threshold,
        "partitions": len(buckets),
        "files": len(plan.files),
        "total_docs_budget": total_docs,
        "docs_per_partition_cap": per_partition_cap,
        "torch_threads_per_partition": threads,
        "host_cpu_count": __import__("os").cpu_count(),
        "documents_scanned": scanned,
        "scoring_errors": sum(s["errors"] for s in stats),
        "above_threshold": sum(s["above_threshold"] for s in stats),
        "wall_seconds": round(wall, 2),
        # Two rates, because they answer different questions. Wall includes JVM startup and
        # one model load per process, a fixed cost that dominates a short run and would
        # make Spark look worse than it is. Compute excludes it and is what a long scan
        # converges on. Reporting only one of them would be a choice about which story to
        # tell.
        "documents_per_second_wall": round(scanned / wall, 3) if wall else 0.0,
        "documents_per_second_compute": (
            round(scanned / max(per_partition), 3) if per_partition else 0.0
        ),
        "startup_seconds": round(wall - max(per_partition), 2) if per_partition else 0.0,
        "per_partition_seconds": [round(x, 2) for x in per_partition],
        "skew": round(max(per_partition) / mean, 3) if mean else 1.0,
    }
    if mining_allowed:
        result["candidate_ids"] = sorted(h["doc_id"] for h in hits)
        result["mining_candidates_written"] = True

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scan_p{len(buckets)}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidate_ids"}, indent=1))
    print(f"wrote {path}")
    return result


def summarise() -> int:
    runs = {}
    for f in sorted(OUT.glob("scan_p*.json")):
        d = json.loads(f.read_text())
        runs[d["partitions"]] = d
    if 1 not in runs:
        print("no scan_p1.json: the single-partition run is the baseline for every speedup")
        return 1

    budgets = {runs[p].get("documents_scanned") for p in runs}
    if len(budgets) != 1:
        print(f"REFUSING TO SUMMARISE: the runs scanned different numbers of documents "
              f"{sorted(budgets)}. A speedup across runs that did different amounts of "
              f"work is not a speedup. Re-run them with the same --max-docs.")
        return 1

    base = runs[1]["documents_per_second_compute"]
    rows = [{
        "partitions": p,
        "documents_per_second_compute": runs[p]["documents_per_second_compute"],
        "documents_per_second_wall": runs[p]["documents_per_second_wall"],
        "wall_seconds": runs[p]["wall_seconds"],
        "startup_seconds": runs[p]["startup_seconds"],
        "speedup": round(runs[p]["documents_per_second_compute"] / base, 3) if base else 0.0,
        "efficiency": (
            round(runs[p]["documents_per_second_compute"] / base / p, 3) if base else 0.0
        ),
        "skew": runs[p]["skew"],
    } for p in sorted(runs)]

    out = {
        "mode": "spark local",
        "root": runs[1]["root"],
        "is_reserve_pool": runs[1]["is_reserve_pool"],
        "documents_scanned": runs[1]["documents_scanned"],
        "invariant": (
            "Every partition count scans the SAME total number of documents; the "
            "per-partition cap absorbs the difference. Speedup is computed on compute "
            "time, which excludes JVM startup and the one model load per process."
        ),
        "caveat": (
            "Spark local mode on one machine. Partitions here are threads sharing RAM and "
            "one disk, not executors on separate hosts, so this measures whether the job "
            "PARALLELISES, not what a cluster would do. No cluster run exists."
        ),
        "rows": rows,
    }
    path = OUT / "spark_summary.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'parts':>6} {'docs/s cpu':>11} {'docs/s wall':>12} {'startup s':>10} "
          f"{'speedup':>9} {'eff':>7} {'skew':>7}")
    for r in rows:
        print(f"{r['partitions']:>6} {r['documents_per_second_compute']:>11.2f} "
              f"{r['documents_per_second_wall']:>12.2f} {r['startup_seconds']:>10.1f} "
              f"{r['speedup']:>9.2f} {r['efficiency']:>7.1%} {r['skew']:>7.2f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--arm", default="baseline")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--partitions", type=int, default=1)
    ap.add_argument("--max-docs", type=int, default=40,
                    help="TOTAL documents across all partitions, not per partition. The "
                         "cap per partition is derived, so every partition count does the "
                         "same work and the speedup means something.")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="intra-op threads PER PARTITION. Default divides the host's "
                         "cores among the partitions so they do not oversubscribe.")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        raise SystemExit(summarise())
    run(a.root, a.arm, a.threshold, a.partitions, a.max_docs, a.torch_threads)
