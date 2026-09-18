"""The same reserve-pool scan as an Apache Beam pipeline.

WHY A SECOND RUNNER AT ALL. Not because Beam is better here. The scan is the one job in
FORGE with a distributed shape, and having written it once for Spark, expressing it again
on a different substrate is what tests whether the job was actually separable from its
runner. It was not: porting it is what forced scan_core to exist. The port is therefore
worth more as evidence about the code's structure than as a benchmark.

WHAT BEAM CHANGES, AND WHAT IT MUST NOT. The scan, the partitioning, the per-partition
document cap and the intra-op thread division all come from scan_core, unchanged and
unwrapped. If any of them were re-implemented here the two runners would be measured over
different work and the comparison would be worthless. What is genuinely Beam's is in this
file and nowhere else: how a bucket of shards becomes an element, how elements are kept
from being fused into one bundle, and which local execution mode the DirectRunner uses.

THE THREE TRAPS. Each one silently produces a meaningless number rather than an error.

1. FUSION. beam.Create(buckets) | beam.FlatMap(scan) looks parallel and is not. The
   DirectRunner fuses the Create and the FlatMap into one stage and may hand the whole
   PCollection to a single bundle, so four buckets are scanned one after another in one
   worker. Wall time comes out identical at every partition count, the speedup reads 1.0,
   and the conclusion "Beam does not parallelise this" would be a conclusion about bundle
   assignment. beam.Reshuffle is the documented fix: it forces a materialisation boundary,
   after which the downstream work is split across workers. It is inserted here on purpose
   and the test suite pins its presence, because it is invisible in the output when absent.

2. `--runner=DirectRunner` IS NOT THE DIRECTRUNNER, and this one was caught by measuring
   rather than by reading. On Beam 2.76, apache_beam.runners.direct.DirectRunner is
   SwitchingDirectRunner: it inspects the pipeline, and unless something is unsupported it
   hands the job to PRISM, a Go binary downloaded on first use and run as a subprocess.
   Prism does not read DirectOptions. So direct_num_workers and direct_running_mode were
   set, ignored, and the sweep still produced a table: 1, 2 and 4 partitions, a documents
   count that matched, and a first run 11x slower than the second because it had
   downloaded the Prism binary mid-measurement. Nothing failed. The independent variable
   simply was not connected to anything.

   The runner is therefore pinned to FnApiRunner, which is Beam's local portability runner
   and the thing `--runner=DirectRunner` resolved to before the Prism switch landed. It is
   the runner that actually honours the two options below, and a sweep whose parallelism
   the runner chooses for itself is not a sweep. The runner name is recorded in every
   artifact so no one later reads this table as a Prism number.

3. EXECUTION MODE. FnApiRunner defaults to multi_threading, where every worker is a THREAD
   in one process. The model cache in scan_core is a module-level dict, so under threading
   the 184M parameter model loads once for the whole job, while Spark local mode gives each
   partition its own worker process and loads it once per worker. That is a real difference
   in what the two numbers mean, not an artefact to hide: it makes Beam's startup cost look
   lower for a reason that has nothing to do with throughput. So the mode is a flag, it is
   recorded in every artifact, and scan_partition reports which call paid for the load, so
   the difference is in the data rather than in a footnote. multi_processing is the mode
   comparable to Spark local.

SCOPE. One machine, one local runner. FnApiRunner is a local runner with extra checks, not
a throughput engine, so an absolute docs/s here is a property of this runner and not of
Beam. Nothing here has run on Dataflow, Flink or Prism at scale, and this repository makes
no claim that it has.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from forge.hard_negative.scan_core import (
    ScanError,
    ScanPlan,
    balanced_partitions,
    partition_files,
    scan_partition,
)

RUNNING_MODES = ("multi_threading", "multi_processing", "in_memory")

# Beam's local portability runner, named explicitly. See trap 2: "DirectRunner" is a
# switching runner that prefers Prism and discards the options this module sets.
RUNNER = "FnApiRunner"


class BeamScanError(ScanError):
    """Beam-specific failures. A subclass, unlike the Spark alias, because these are
    conditions the core scan cannot be in: a bad DirectRunner mode is not a bad scan."""


def beam_pipeline_options(partitions: int, running_mode: str = "multi_threading"):
    """Build the options that actually make the pipeline parallel, on a runner that reads them.

    TWO DEFAULTS, BOTH TRAPS. direct_num_workers defaults to 1, so a pipeline left on the
    defaults runs every bundle in one worker no matter how many buckets it was given; and
    RUNNER is the runner that reads these options at all, which `--runner=DirectRunner` has
    not been since Beam switched it to Prism. The Spark script says local[P] in one place
    and gets P. The Beam equivalent is all three lines below, and getting any of them wrong
    does not fail, it just decouples the measurement from the thing being varied.

    direct_num_workers is set to the partition count rather than to 0 (which means "one per
    core"), because the point of the sweep is to vary the parallelism deliberately and have
    the document cap and thread division track it.
    """
    if running_mode not in RUNNING_MODES:
        raise BeamScanError(
            f"running_mode {running_mode!r} is not one of {RUNNING_MODES}. "
            "multi_processing is the mode comparable to Spark local, because it gives each "
            "partition its own process and therefore its own model load."
        )
    if partitions < 1:
        raise BeamScanError("partitions must be >= 1")

    from apache_beam.options.pipeline_options import DirectOptions, PipelineOptions

    # NOT "DirectRunner". See trap 2 in the module docstring: that name resolves to
    # SwitchingDirectRunner, which prefers Prism, which ignores everything set below.
    options = PipelineOptions([f"--runner={RUNNER}"])
    direct = options.view_as(DirectOptions)
    direct.direct_num_workers = partitions
    direct.direct_running_mode = running_mode
    return options


def keyed_buckets(plan: ScanPlan) -> list[tuple[int, list[str]]]:
    """One element per partition, keyed by index.

    The key is not decoration. Reshuffle is a key-based operation, and an unkeyed
    PCollection of lists gives it nothing to spread the elements over, so the buckets can
    land in one bundle again. Keying by bucket index is the smallest thing that makes the
    redistribution meaningful, and it also means a partition's stats can be attributed to
    the bucket that produced them rather than to whichever order they came back in.
    """
    return list(enumerate(balanced_partitions(plan)))


def scan_bucket(element: tuple[int, Any], arm: str, threshold: float,
                limit: int | None = None,
                torch_threads: int | None = None) -> Iterator[dict[str, Any]]:
    """Run one bucket through the shared scan and tag every record with its partition.

    partition_files is the same unwrapper the Spark path uses, called for the same reason:
    the element's shape is decided by the runner, a list that should have been a path is
    what broke the first Spark run, and guessing at the shape inside the worker is how that
    failure arrives 14 seconds in as a pyarrow TypeError instead of here as an error about
    partition elements.
    """
    index, payload = element
    files = partition_files([payload])
    if not files:
        return
    for record in scan_partition(files, arm, threshold, limit=limit,
                                 torch_threads=torch_threads):
        record["partition_index"] = index
        yield record


def build_scan(pcoll, arm: str, threshold: float, limit: int | None = None,
               torch_threads: int | None = None):
    """Attach the scan to a PCollection of keyed buckets.

    Separated from the pipeline so a test can assert the transform graph contains the
    Reshuffle without executing a scan that needs the model on disk. The composition is
    the part that goes wrong, and it is the part that is cheap to check.
    """
    import apache_beam as beam

    return (
        pcoll
        # THE FUSION BREAK. Without this the next FlatMap is fused into whatever produced
        # the collection and can run every bucket in one bundle. See trap 1 above.
        | "RedistributeBuckets" >> beam.Reshuffle()
        | "ScanBucket" >> beam.FlatMap(scan_bucket, arm=arm, threshold=threshold,
                                       limit=limit, torch_threads=torch_threads)
    )
