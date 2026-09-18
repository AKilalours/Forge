"""The reserve-pool scan as a Spark job. The scan itself is in scan_core.

WHY SPARK HERE AND NOWHERE ELSE IN THIS REPO. Phase 1 cleaning runs on Polars and PyArrow
because 400k documents fit in memory on one machine and Spark would add operational cost
for no throughput. docs/jd_coverage.md argues that, and it is still right. The mining scan
is the one job with a different shape: it reads a reserve pool sized in millions, runs an
independent forward pass per document, and keeps the small fraction the detector is
confidently wrong about. No shuffle, no join, no cross-document state. Embarrassingly
parallel over shards, and bounded by compute rather than by memory.

forge.hard_negative.reserve.load_reserve is the single-machine version: it globs every
parquet file and materialises every document into one Python list. That is correct at
400k and impossible at 5M. The scan in scan_core is the same scan without that constraint.

WHY THIS MODULE IS NOW ALMOST EMPTY. Everything it used to define is runner-independent and
moved to forge.hard_negative.scan_core when the same scan was ported to Beam. The names are
re-exported rather than removed, because scripts/spark_scan_reserve.py and
tests/unit/test_spark_scan.py import them from here and a rename that breaks a test for no
behavioural reason is churn, not a refactor. The Spark-specific part of the job is entirely
in that script: the SparkSession, local[P], worker reuse, and mapPartitions.

SCOPE, stated because it is the honest limit. This has been run in Spark LOCAL mode over
the parquet in this repository. It has never run on a cluster, and the 5-million-document
pool it is written for does not exist on disk.
"""

from __future__ import annotations

from forge.hard_negative.scan_core import (
    _SCORER_CACHE,
    RESERVE_COLUMNS,
    ScanError,
    ScanPlan,
    ScanStats,
    _rows_from_files,
    _scorer_for,
    balanced_partitions,
    docs_per_partition,
    is_reserve_pool,
    partition_files,
    plan_scan,
    scan_partition,
    threads_per_partition,
)

# The error this module used to define. An ALIAS, not a subclass: scan_core raises
# ScanError, and a subclass here would mean `except SparkScanError` silently stopped
# catching what the core raises.
SparkScanError = ScanError

# The private names are listed deliberately. tests/unit/test_spark_scan.py reaches for
# spark_scan._SCORER_CACHE and spark_scan._scorer_for to pin the per-process model cache,
# so they are part of this module's surface whether or not the underscore suggests it, and
# naming them here is what marks them as re-exports rather than unused imports.
__all__ = [
    "RESERVE_COLUMNS",
    "_SCORER_CACHE",
    "_rows_from_files",
    "_scorer_for",
    "ScanError",
    "ScanPlan",
    "ScanStats",
    "SparkScanError",
    "balanced_partitions",
    "docs_per_partition",
    "is_reserve_pool",
    "partition_files",
    "plan_scan",
    "scan_partition",
    "threads_per_partition",
]
