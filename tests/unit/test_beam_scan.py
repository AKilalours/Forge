"""The Beam pipeline's decisions, and the two ways it fails without saying anything.

The Beam port shares the entire scan with the Spark one, so there is nothing to test twice:
partitioning, the document cap, the thread division and the model cache are covered by
test_scan_core.py and test_spark_scan.py against the same functions. What is Beam's own is
small and almost entirely a matter of pipeline construction, which is also where both of
its real failure modes live.

Neither failure raises. A fused pipeline runs every bucket in one worker and reports a
speedup of 1.0; a DirectRunner left on direct_num_workers=1 does the same. In both cases
the output is correct and the measurement is worthless, which is the hardest kind of bug to
notice in a benchmark. So the tests here pin the transform graph and the options, not the
results.

The tests that need apache-beam skip cleanly rather than fail, for the reason the Spark
suite skips on a missing JVM: a machine without the optional extra should say so.
"""

from __future__ import annotations

import pytest

from forge.hard_negative.beam_scan import (
    BeamScanError,
    ScanPlan,
    keyed_buckets,
    scan_bucket,
)

# importorskip is per test rather than at module level on purpose. Most of what is worth
# pinning here needs no Beam: the bucket keying is pure, and beam_pipeline_options validates
# its arguments BEFORE importing apache_beam so a typo fails on the driver either way. Only
# the three tests that build a graph or read DirectOptions need the extra installed, and
# skipping the whole file for them would hide six checks on a machine without it.
_NEEDS_BEAM = "apache-beam is in the orchestration extra"


def _plan(n_files: int, partitions: int) -> ScanPlan:
    return ScanPlan(files=tuple(f"s{i}.parquet" for i in range(n_files)),
                    partitions=partitions, threshold=0.99)


def test_buckets_are_keyed_so_reshuffle_has_something_to_spread() -> None:
    """Reshuffle is key-based. An unkeyed PCollection of lists gives it nothing to
    redistribute over, so the buckets can land in one bundle again and the fusion break
    becomes decorative."""
    keyed = keyed_buckets(_plan(8, 4))
    assert [k for k, _ in keyed] == [0, 1, 2, 3]
    assert [v for _, v in keyed] == [
        ["s0.parquet", "s4.parquet"],
        ["s1.parquet", "s5.parquet"],
        ["s2.parquet", "s6.parquet"],
        ["s3.parquet", "s7.parquet"],
    ]


def test_the_beam_buckets_are_the_same_buckets_spark_gets() -> None:
    """The comparison in the README is only legitimate if both runners scan the same shards
    in the same grouping. This is that claim, as an assertion."""
    from forge.hard_negative.scan_core import balanced_partitions

    plan = _plan(17, 5)
    assert [v for _, v in keyed_buckets(plan)] == balanced_partitions(plan)


def test_no_empty_bucket_is_submitted() -> None:
    """An empty bucket still costs a worker and, under multi_processing, a model load."""
    keyed = keyed_buckets(_plan(3, 8))
    assert len(keyed) == 3
    assert all(v for _, v in keyed)


def test_the_scan_graph_contains_the_fusion_break() -> None:
    """THE TRAP WITH NO ERROR MESSAGE.

    Create | FlatMap is a pipeline the DirectRunner may fuse into one stage and hand to a
    single bundle. Four buckets then run one after another, wall time is identical at every
    partition count, and the honest-looking conclusion "Beam does not parallelise this" is
    a statement about bundle assignment. Reshuffle forces the materialisation boundary. It
    is invisible in the output when absent, so it is pinned in the graph.
    """
    beam = pytest.importorskip("apache_beam", reason=_NEEDS_BEAM)

    from forge.hard_negative.beam_scan import build_scan

    # Constructed and never run. `with beam.Pipeline()` executes on exit, which would load
    # a 184M parameter model to assert something about a graph.
    pipeline = beam.Pipeline()
    build_scan(pipeline | beam.Create(keyed_buckets(_plan(4, 2)), reshuffle=False),
               arm="baseline", threshold=0.99, limit=1)
    names = [t.unique_name for t in pipeline.to_runner_api().components.transforms.values()]

    assert any("RedistributeBuckets" in n for n in names), f"no Reshuffle in the graph: {names}"
    assert any("ScanBucket" in n for n in names), f"no scan in the graph: {names}"


def test_the_runner_is_told_how_many_workers_to_use() -> None:
    """THE TRAP WITH NO ERROR MESSAGE. direct_num_workers defaults to 1, so a pipeline on
    the defaults runs every bucket in one worker however many it was given. This is the
    Beam equivalent of Spark's local[P], and forgetting it flattens the curve rather than
    failing."""
    pytest.importorskip("apache_beam", reason=_NEEDS_BEAM)
    from apache_beam.options.pipeline_options import DirectOptions

    from forge.hard_negative.beam_scan import beam_pipeline_options

    for partitions in (1, 2, 4):
        direct = beam_pipeline_options(partitions).view_as(DirectOptions)
        assert direct.direct_num_workers == partitions
        assert direct.direct_running_mode == "multi_threading"


def test_multi_processing_is_available_because_it_is_the_mode_spark_is_comparable_to() -> None:
    """Under multi_threading the workers are threads in one process and share scan_core's
    model cache, so the model loads once for the whole job. Spark local gives each
    partition a process. Comparing the two startup costs without the mode recorded would be
    comparing different things."""
    pytest.importorskip("apache_beam", reason=_NEEDS_BEAM)
    from apache_beam.options.pipeline_options import DirectOptions

    from forge.hard_negative.beam_scan import beam_pipeline_options

    direct = beam_pipeline_options(4, "multi_processing").view_as(DirectOptions)
    assert direct.direct_running_mode == "multi_processing"


def test_an_unknown_running_mode_is_refused_on_the_driver() -> None:
    """Beam accepts the string and fails later, inside the runner, at a point where the
    error is about workers rather than about the typo."""
    from forge.hard_negative.beam_scan import beam_pipeline_options

    with pytest.raises(BeamScanError, match="not one of"):
        beam_pipeline_options(2, "multithreading")


def test_zero_partitions_is_refused() -> None:
    from forge.hard_negative.beam_scan import beam_pipeline_options

    with pytest.raises(BeamScanError, match=">= 1"):
        beam_pipeline_options(0)


def test_scan_bucket_unwraps_the_element_with_the_same_function_spark_uses() -> None:
    """The bug that broke the first Spark run was a list where a path belonged, arriving as
    a pyarrow TypeError 14 seconds into the job. Beam's element shape is decided by Beam,
    so the same unwrapper is used here, and an empty bucket yields nothing rather than
    handing pyarrow an empty list."""
    assert list(scan_bucket((0, []), arm="baseline", threshold=0.99)) == []


def test_every_record_carries_the_bucket_that_produced_it(monkeypatch) -> None:
    """Without the partition index, a stats record cannot be attributed to a bucket, and a
    bucket that silently scanned nothing is indistinguishable from one with no hits. That
    is the specific way a fused or mis-keyed Beam pipeline loses work."""
    from forge.hard_negative import beam_scan

    def fake_scan_partition(files, arm, threshold, limit=None, torch_threads=None):
        yield {"doc_id": "d1", "score_mean": 0.999}
        yield {"doc_id": None, "_stats": True, "scanned": 1, "errors": 0,
               "above_threshold": 1, "seconds": 0.5, "loaded_model": True}

    monkeypatch.setattr(beam_scan, "scan_partition", fake_scan_partition)
    records = list(beam_scan.scan_bucket((3, ["a.parquet"]), arm="baseline", threshold=0.99))
    assert [r["partition_index"] for r in records] == [3, 3]


def test_the_runner_is_pinned_by_name_and_is_not_directrunner() -> None:
    """THE TRAP THAT SURVIVED A CLEAN SWEEP.

    On Beam 2.76 apache_beam.runners.direct.DirectRunner is SwitchingDirectRunner, which
    hands the pipeline to Prism unless something is unsupported. Prism is a Go binary in a
    subprocess and does not read DirectOptions, so direct_num_workers and
    direct_running_mode were set and discarded. The sweep still produced a table with the
    right document count at 1, 2 and 4 partitions; the parallelism was simply not the
    variable being changed, and the first run was 11x slower than the second because it
    downloaded the Prism binary mid-measurement.

    This pins the runner name, because the failure is a plausible table rather than an
    error, and the next person to "simplify" this back to DirectRunner will get one too.
    """
    pytest.importorskip("apache_beam", reason=_NEEDS_BEAM)
    from apache_beam.options.pipeline_options import StandardOptions

    from forge.hard_negative.beam_scan import RUNNER, beam_pipeline_options

    assert RUNNER == "FnApiRunner"
    runner = beam_pipeline_options(2).view_as(StandardOptions).runner
    assert runner == "FnApiRunner", (
        f"runner is {runner!r}. DirectRunner resolves to a switching runner that prefers "
        "Prism and ignores the worker count this sweep varies."
    )


def test_the_switching_runner_really_does_prefer_prism() -> None:
    """The claim above is about Beam's behaviour, not this repository's, so it is asserted
    against the installed Beam rather than trusted. If a future version stops preferring
    Prism this fails and the docstring gets corrected instead of quietly rotting."""
    pytest.importorskip("apache_beam", reason=_NEEDS_BEAM)
    import inspect

    from apache_beam.runners.direct.direct_runner import SwitchingDirectRunner

    body = inspect.getsource(SwitchingDirectRunner.run_pipeline)
    assert "PrismRunner" in body, (
        "SwitchingDirectRunner no longer mentions Prism. Re-check whether "
        "--runner=DirectRunner now honours direct_num_workers, and update beam_scan's "
        "trap 2 to match what this version actually does."
    )
