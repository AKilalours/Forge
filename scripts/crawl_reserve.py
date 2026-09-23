"""Mine Common Crawl into the reserve pool, on the serial reference, Spark or Beam.

    # measure first: two segments, the real policy, nothing kept
    python scripts/crawl_reserve.py --segments 2 --runner serial --out /tmp/crawl-probe

    # the real pool (the minimal config's target is human_reserve: 500000)
    python scripts/crawl_reserve.py --segments 300 --runner spark --partitions 8 \
        --shards 64 --out data/reserve

Every run writes a report to reports/experiments/crawl/<runner>/ with the plan, the
policy that was enforced, every count, and a `corpus_fingerprint`. Two runs over the same
segments must agree on that fingerprint whichever runner produced them; that is how the
Spark and Beam numbers in the README are allowed to be compared at all.

THREE REFUSALS, EACH FOR A REASON.

A non-empty --out. Shards are named by content hash, so a second run into the same
directory overwrites some files and leaves others, and the result belongs to neither run.

A --out named "reserve" without a training root. The reserve pool's one promise is that
it is disjoint from training; without the training hashes that promise is not checked.

Any --config other than the one the models were trained under, unless stated with
--i-know-the-policy-differs. See crawl_core's module docstring on why the policy matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = REPO / "configs" / "data" / "human_minimal.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--crawl", default="CC-MAIN-2026-34")
    parser.add_argument("--segments", type=int, default=1,
                        help="first N segments of the crawl index")
    parser.add_argument("--paths-file", type=Path,
                        help="one segment path per line; reproduces an earlier plan exactly")
    parser.add_argument("--runner", choices=("serial", "spark", "beam"), default="serial")
    parser.add_argument("--running-mode", default="multi_processing",
                        help="beam only: multi_processing, multi_threading or in_memory")
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--config", type=Path, default=TRAINING_CONFIG)
    parser.add_argument("--i-know-the-policy-differs", action="store_true")
    parser.add_argument("--training-root", type=Path, default=REPO / "data" / "silver")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-version", default="v0.2-reserve")
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports" / "experiments" / "crawl")
    args = parser.parse_args(argv)

    from forge.ingestion.crawl_core import (
        load_policy,
        plan_crawl,
        run_serial,
        training_hashes,
        write_crawl_manifest,
    )

    if args.config.resolve() != TRAINING_CONFIG.resolve() and not args.i_know_the_policy_differs:
        print(f"refusing: {args.config} is not the config the models were trained under "
              f"({TRAINING_CONFIG.name}). Pass --i-know-the-policy-differs to proceed.",
              file=sys.stderr)
        return 2
    if args.out.exists() and any(args.out.rglob("*.parquet")):
        print(f"refusing: {args.out} already holds parquet. Shards are named by content "
              f"hash, so a second run would mix with the first. Use an empty directory.",
              file=sys.stderr)
        return 2

    paths = None
    if args.paths_file:
        paths = [line.strip() for line in args.paths_file.read_text().splitlines() if line.strip()]

    policy = load_policy(args.config)
    excluded = training_hashes(args.training_root)
    plan = plan_crawl(args.crawl, segments=args.segments, partitions=args.partitions,
                      shards=args.shards, paths=paths)
    print(f"plan {plan.fingerprint}: {len(plan.segments)} segments, {plan.partitions} "
          f"partitions, {plan.shards} shards, {len(excluded)} training hashes excluded",
          flush=True)

    if args.runner == "serial":
        report = run_serial(plan, args.out, policy, excluded)
    elif args.runner == "spark":
        from forge.ingestion.spark_crawl import run_spark
        report = run_spark(plan, args.out, policy, excluded)
    else:
        from forge.ingestion.beam_crawl import run_beam
        report = run_beam(plan, args.out, policy, excluded, running_mode=args.running_mode)

    identities = _identities(args.out)
    if len(identities) != report["documents_written"]:
        print(f"refusing to write a manifest: the runner reports "
              f"{report['documents_written']} documents but {len(identities)} are on disk.",
              file=sys.stderr)
        return 1
    write_crawl_manifest(args.out, args.dataset_version, report, identities)

    report["segment_paths"] = list(plan.segments)
    runner_dir = args.report_dir / report["runner"]
    runner_dir.mkdir(parents=True, exist_ok=True)
    report_path = runner_dir / f"{plan.fingerprint}-p{plan.partitions}-s{plan.shards}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: v for k, v in report.items() if k != "segment_paths"}, indent=2))
    print(f"report: {report_path}")
    return 0


def _identities(out: Path) -> list[str]:
    """Re-read what was actually written, rather than trusting what the runner returned.

    The manifest describes the files on disk. Building it from the runner's own account
    would let a shard that failed to write still be listed.
    """
    import pyarrow.parquet as pq

    lines = []
    for path in sorted(out.glob("source=*/split=*/*.parquet")):
        split = path.parent.name.split("=", 1)[1]
        table = pq.ParquetFile(path).read(columns=["doc_id", "content_sha256"])
        for row in table.to_pylist():
            lines.append(f"{row['doc_id']}\t{row['content_sha256']}\t{split}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
