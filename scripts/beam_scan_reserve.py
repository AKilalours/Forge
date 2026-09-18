"""Run the reserve-pool scan on a local Apache Beam runner, and measure how it partitions.

    python scripts/beam_scan_reserve.py --partitions 1
    python scripts/beam_scan_reserve.py --partitions 4
    python scripts/beam_scan_reserve.py --summarise

This is the Beam half of the pair. scripts/spark_scan_reserve.py is the other half, and
the two are deliberately written to the same shape: same root, same arm, same threshold,
same TOTAL document budget split across partitions, same artifact fields. Anything they do
not share comes from forge.hard_negative.scan_core, so the work being timed is the same
code in both and the two numbers can be put in one table.

WHAT THIS MEASURES. Throughput and partition skew for the mining scan, on Beam's local
portability runner (FnApiRunner), on this machine. It is pinned to that runner on purpose:
`--runner=DirectRunner` on Beam 2.76 resolves to a switching runner that prefers Prism, a
subprocess that ignores direct_num_workers, so a sweep run that way varies a number the
runner never reads. forge.hard_negative.beam_scan documents how that was found. A local
runner adds checks a production runner does not, so the absolute docs/s is a property of
this runner and not of Beam. It does not measure Dataflow or Flink, and this repository has
never run either.

WHY IT REFUSES TO MINE FROM data/silver. Same reason the Spark script does, enforced by the
same function. The only parquet in a checkout is the training pool; scanning it finds
documents the model was TRAINED on, and a "false positive" on memorised text says nothing
about production behaviour. When the root is not a reserve pool this script writes timing
and nothing else: no candidate ids, no parquet of hits, nothing that could be mistaken
later for a mining round.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from forge.hard_negative.beam_scan import RUNNER
from forge.hard_negative.scan_core import (
    docs_per_partition,
    is_reserve_pool,
    threads_per_partition,
)

OUT = Path("reports/experiments/beam")
DEFAULT_ROOT = "data/reserve"


def _collect(pattern_dir: Path) -> list[dict]:
    """Read back what the pipeline wrote.

    WHY A FILE AND NOT A LIST. The obvious driver-side collection is appending to a Python
    list inside a Map, and it works under the default multi_threading mode because the
    workers are threads in this process. Under multi_processing the workers are separate
    processes, the list they append to is a copy, and the driver's copy comes back EMPTY:
    zero documents scanned, no error, and a summary that reports a division by zero rather
    than a failure. Writing JSON lines and reading them back behaves identically in every
    mode, which is the only reason it is worth the temporary directory.
    """
    rows: list[dict] = []
    for f in sorted(pattern_dir.glob("out*")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(root: str, arm: str, threshold: float, partitions: int,
        total_docs: int | None, torch_threads: int | None = None,
        running_mode: str = "multi_threading") -> dict:
    import apache_beam as beam

    from forge.hard_negative.beam_scan import (
        beam_pipeline_options,
        build_scan,
        keyed_buckets,
    )
    from forge.hard_negative.scan_core import plan_scan

    plan = plan_scan(root, partitions=partitions, threshold=threshold)
    buckets = keyed_buckets(plan)
    mining_allowed = is_reserve_pool(root)
    per_partition_cap = docs_per_partition(total_docs, len(buckets))
    threads = threads_per_partition(len(buckets), torch_threads)
    options = beam_pipeline_options(len(buckets), running_mode)

    with tempfile.TemporaryDirectory() as tmp:
        sink = Path(tmp)
        t0 = time.perf_counter()
        with beam.Pipeline(options=options) as pipeline:
            records = build_scan(
                pipeline | "Buckets" >> beam.Create(buckets, reshuffle=False),
                arm, threshold, limit=per_partition_cap, torch_threads=threads,
            )
            (
                records
                | "ToJson" >> beam.Map(json.dumps)
                | "Write" >> beam.io.WriteToText(str(sink / "out"), shard_name_template="-SSS")
            )
        wall = time.perf_counter() - t0
        rows = _collect(sink)

    stats = [r for r in rows if r.get("_stats")]
    hits = [r for r in rows if not r.get("_stats")]
    scanned = sum(s["scanned"] for s in stats)
    per_partition = [s["seconds"] for s in stats]
    mean = sum(per_partition) / len(per_partition) if per_partition else 0.0

    result = {
        "mode": f"beam {RUNNER.lower()} {running_mode}",
        "runner": RUNNER,
        "running_mode": running_mode,
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
        "partitions_reporting": len(stats),
        "files": len(plan.files),
        "total_docs_budget": total_docs,
        "docs_per_partition_cap": per_partition_cap,
        "torch_threads_per_partition": threads,
        "host_cpu_count": __import__("os").cpu_count(),
        "documents_scanned": scanned,
        "scoring_errors": sum(s["errors"] for s in stats),
        "above_threshold": sum(s["above_threshold"] for s in stats),
        # How many workers paid for a model load. Under multi_threading this is 1 however
        # many partitions there are, because the workers are threads sharing one module
        # cache; under multi_processing it rises with the partition count. It is the field
        # that explains the startup column, and the reason Beam's wall-clock number is not
        # directly comparable to Spark's without reading it.
        "model_loads": sum(1 for s in stats if s.get("loaded_model")),
        # PROCESS OR THREAD, as data. Which of the two a runner gives a partition is the
        # fact that most changes how these timings should be read, and it is the fact this
        # code was wrong about twice before it was measured. One distinct pid means the
        # partitions were threads sharing one model; N means separate worker processes.
        "distinct_worker_pids": len({s.get("pid") for s in stats if s.get("pid")}),
        "wall_seconds": round(wall, 2),
        # Two rates, because they answer different questions. Wall includes pipeline
        # construction and the model load, a fixed cost that dominates a short run.
        # Compute excludes it and is what a long scan converges on. Reporting only one of
        # them would be a choice about which story to tell.
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

    # A partition that produced no stats record scanned nothing, and its absence would
    # otherwise be invisible: the remaining partitions still report a clean speedup over a
    # smaller job. This is the Beam-specific failure, because a fused or mis-keyed
    # collection loses whole buckets rather than erroring.
    if len(stats) != len(buckets):
        result["WARNING"] = (
            f"{len(buckets)} buckets were submitted and {len(stats)} reported. The missing "
            "ones scanned nothing, so this run is not comparable to the others."
        )

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

    modes = {runs[p].get("running_mode") for p in runs}
    if len(modes) != 1:
        print(f"REFUSING TO SUMMARISE: the runs used different DirectRunner modes "
              f"{sorted(modes)}. Threads in one process and separate processes pay "
              f"different startup costs, so a speedup across them is not a speedup.")
        return 1

    base = runs[1]["documents_per_second_compute"]
    rows = [{
        "partitions": p,
        "documents_per_second_compute": runs[p]["documents_per_second_compute"],
        "documents_per_second_wall": runs[p]["documents_per_second_wall"],
        "wall_seconds": runs[p]["wall_seconds"],
        "startup_seconds": runs[p]["startup_seconds"],
        "model_loads": runs[p].get("model_loads"),
        "distinct_worker_pids": runs[p].get("distinct_worker_pids"),
        "speedup": round(runs[p]["documents_per_second_compute"] / base, 3) if base else 0.0,
        "efficiency": (
            round(runs[p]["documents_per_second_compute"] / base / p, 3) if base else 0.0
        ),
        "skew": runs[p]["skew"],
    } for p in sorted(runs)]

    out = {
        "mode": f"beam {RUNNER.lower()} {runs[1]['running_mode']}",
        "runner": RUNNER,
        "running_mode": runs[1]["running_mode"],
        "root": runs[1]["root"],
        "is_reserve_pool": runs[1]["is_reserve_pool"],
        "documents_scanned": runs[1]["documents_scanned"],
        "invariant": (
            "Every partition count scans the SAME total number of documents; the "
            "per-partition cap absorbs the difference. The cap, the shard partitioning and "
            "the intra-op thread division come from forge.hard_negative.scan_core, which is "
            "the same code the Spark runner uses, so the two runners are timed over "
            "identical work. Speedup is computed on compute time, which excludes pipeline "
            "construction and the model load."
        ),
        "caveat": (
            "Beam's local portability runner (FnApiRunner) on one machine, pinned by "
            "name because --runner=DirectRunner now resolves to a switching runner that "
            "prefers Prism and discards the worker options this sweep sets. A local "
            "runner performs extra checks a production runner does not, so the absolute "
            "rate is a property of this runner, not of Beam. In multi_threading mode the "
            "workers are threads in one process sharing one model load, which is why "
            "model_loads does not rise with the partition count and why the startup "
            "column is not comparable to Spark's without reading it. No Dataflow or Flink "
            "run exists."
        ),
        "rows": rows,
    }
    path = OUT / "beam_summary.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'parts':>6} {'docs/s cpu':>11} {'docs/s wall':>12} {'startup s':>10} "
          f"{'loads':>6} {'pids':>5} {'speedup':>9} {'eff':>7} {'skew':>7}")
    for r in rows:
        print(f"{r['partitions']:>6} {r['documents_per_second_compute']:>11.2f} "
              f"{r['documents_per_second_wall']:>12.2f} {r['startup_seconds']:>10.1f} "
              f"{str(r['model_loads']):>6} {str(r['distinct_worker_pids']):>5} "
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
                         "same work and the speedup means something. Keep this identical "
                         "to the Spark sweep or the two tables are not comparable.")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="intra-op threads PER PARTITION. Default divides the host's "
                         "cores among the partitions so they do not oversubscribe.")
    ap.add_argument("--running-mode", default="multi_threading",
                    choices=["multi_threading", "multi_processing", "in_memory"],
                    help="local runner worker model. multi_processing is the mode "
                         "comparable to Spark local, because each partition gets its own "
                         "process and therefore its own model load.")
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        raise SystemExit(summarise())
    run(a.root, a.arm, a.threshold, a.partitions, a.max_docs, a.torch_threads,
        a.running_mode)
