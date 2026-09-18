"""The arithmetic that makes two runners comparable, which until now lived in a script.

docs_per_partition, threads_per_partition and is_reserve_pool were defined inside
scripts/spark_scan_reserve.py. Nothing imported them, so nothing tested them, and they are
the three functions that decide whether a published speedup means anything: the first holds
the work constant, the second stops the machine oversubscribing itself, and the third is
the only thing standing between a timing run and a mining round over training data.

They are library code now because the Beam runner needs the identical behaviour. These are
the tests that were missing the whole time.
"""

from __future__ import annotations

import pytest

from forge.hard_negative.scan_core import (
    docs_per_partition,
    is_reserve_pool,
    threads_per_partition,
)


def test_the_total_document_budget_is_held_constant_across_partition_counts() -> None:
    """THE INVARIANT THE FIRST VERSION GOT BACKWARDS.

    The cap is per partition, so it must shrink as partitions grow. Taking a
    --max-docs-per-partition instead meant four partitions scanned four times the work,
    wall time stayed flat, and the speedup read 1.0 from a run that was not comparable to
    its own baseline.
    """
    for partitions in (1, 2, 4, 8):
        cap = docs_per_partition(40, partitions)
        assert cap is not None
        assert cap * partitions >= 40, "the budget must be reachable"
        assert (cap - 1) * partitions < 40, "the cap must not be larger than it needs"


def test_the_cap_rounds_up_so_no_document_is_dropped() -> None:
    """Floor division would cap three partitions at 13 for a budget of 40 and scan 39. A
    speedup over 39 documents compared against a baseline over 40 is a speedup with an
    invisible 2.5% error in it."""
    assert docs_per_partition(40, 3) == 14
    assert docs_per_partition(40, 7) == 6


def test_no_budget_means_no_cap_rather_than_a_cap_of_zero() -> None:
    """A full pool scan passes None. Returning 0 here would scan nothing and report it as
    success."""
    assert docs_per_partition(None, 4) is None


def test_threads_are_divided_among_partitions_not_handed_to_each() -> None:
    """THE OVERSUBSCRIPTION FIX, and the reason the first sweep was wrong.

    Leaving torch at its default per partition means P partitions times T threads asking
    for P*T cores. The measured symptom was a skew of 1.08 with efficiency at 43%:
    balanced partitions on a saturated machine. Division is what makes the total constant.
    """
    import os

    cores = os.cpu_count() or 1
    for partitions in (1, 2, 4):
        assert threads_per_partition(partitions) * partitions <= max(cores, partitions)


def test_thread_division_never_returns_zero() -> None:
    """More partitions than cores is the ordinary case on a laptop, and torch refuses a
    thread count of zero."""
    assert threads_per_partition(1024) == 1


def test_an_explicit_thread_override_is_honoured_and_still_floored_at_one() -> None:
    assert threads_per_partition(4, override=3) == 3
    assert threads_per_partition(4, override=0) == 1


def test_only_a_directory_named_reserve_is_a_reserve_pool(tmp_path) -> None:
    """The check is path-based and strict on purpose. A cleverer check is one that
    eventually says yes to the training pool, and a "false positive" mined out of training
    data is a document the model memorised, which says nothing about production."""
    reserve = tmp_path / "reserve"
    reserve.mkdir()
    assert is_reserve_pool(str(reserve))


@pytest.mark.parametrize("name", ["silver", "reserve_pool", "Reserve", "data", "reserved"])
def test_anything_else_is_refused_including_names_that_nearly_match(tmp_path, name) -> None:
    d = tmp_path / name
    d.mkdir()
    assert not is_reserve_pool(str(d)), f"{name!r} must not be treated as the reserve pool"


def test_a_trailing_slash_or_dot_segment_does_not_defeat_the_check(tmp_path) -> None:
    """resolve() is load bearing. Without it "data/reserve/." and "data/reserve/" have
    names "." and "", neither of which is "reserve", so a legitimate reserve path would be
    silently downgraded to timing-only and a mining round would quietly emit nothing."""
    reserve = tmp_path / "reserve"
    reserve.mkdir()
    assert is_reserve_pool(f"{reserve}/")
    assert is_reserve_pool(f"{reserve}/.")


def test_the_spark_module_still_exports_what_its_callers_import() -> None:
    """The move to scan_core must not be a rename in disguise. spark_scan is a shim now,
    and both the script and the existing test file import from it."""
    from forge.hard_negative import scan_core, spark_scan

    assert spark_scan.SparkScanError is scan_core.ScanError, (
        "an alias, not a subclass: a subclass would mean `except SparkScanError` stopped "
        "catching what the core actually raises"
    )
    for name in ("plan_scan", "balanced_partitions", "partition_files", "scan_partition",
                 "ScanPlan", "ScanStats", "_SCORER_CACHE", "_scorer_for"):
        assert getattr(spark_scan, name) is getattr(scan_core, name), name


def test_the_model_is_loaded_once_even_when_partitions_are_threads() -> None:
    """THE RACE A PROCESS-BASED RUNNER CANNOT EXPOSE, and the reason to port at all.

    `if arm not in cache: cache[arm] = load_arm(arm)` is safe under Spark local mode,
    because each partition is its own worker process and there is nothing to race with.
    Under Beam's local runner in multi_threading mode each partition is a thread in one
    process, and every thread evaluates the cache check before any of them has finished
    loading. Measured on the stub: four partitions, four model loads, cache present and
    decorative. With the real 184M parameter arm that is four copies of the weights
    resident at once.

    No error, no warning. The only reason it was seen is that scan_partition reports
    whether its own call paid for the load, which had been added to explain a startup
    column. This test is what stops it coming back, and it fails without the lock.
    """
    import sys
    import threading
    import types

    from forge.hard_negative import scan_core

    calls = {"n": 0}
    started = threading.Barrier(4)

    def slow_load_arm(arm: str) -> object:
        calls["n"] += 1
        # Wide enough that every thread is inside the unlocked check at the same time.
        # Without the lock this sleep is what makes all four of them load.
        threading.Event().wait(0.05)
        return object()

    stub = types.ModuleType("forge.inference.scorer")
    stub.load_arm = slow_load_arm
    real = sys.modules.get("forge.inference.scorer")
    sys.modules["forge.inference.scorer"] = stub
    scan_core._SCORER_CACHE.clear()

    seen: list[object] = []

    def worker() -> None:
        started.wait()
        seen.append(scan_core._scorer_for("baseline"))

    try:
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        scan_core._SCORER_CACHE.clear()
        if real is not None:
            sys.modules["forge.inference.scorer"] = real
        else:
            del sys.modules["forge.inference.scorer"]

    assert calls["n"] == 1, (
        f"the model was loaded {calls['n']} times by 4 concurrent partitions. The cache "
        "check is not atomic; _SCORER_LOCK is what makes it one load."
    )
    assert len({id(s) for s in seen}) == 1, "every thread must get the same scorer"


def test_exactly_one_of_four_concurrent_partitions_claims_the_model_load() -> None:
    """THE REPORTING BUG, which is the race's second-order version and was measured first.

    The first attempt at this field counted loads in a module-level integer and had each
    partition compare it before and after acquiring its scorer. A thread that blocked on
    the lock while another loaded the model sees the counter move and reports that it paid
    for the load, so four partitions reported four loads on the run where the lock had
    just made it one: the artifact said the cache was useless at the moment it started
    working. A before-and-after reading of shared state cannot say which caller acted.
    """
    import sys
    import threading
    import types

    from forge.hard_negative import scan_core

    started = threading.Barrier(4)

    def slow_load_arm(arm: str) -> object:
        threading.Event().wait(0.05)
        return object()

    stub = types.ModuleType("forge.inference.scorer")
    stub.load_arm = slow_load_arm
    real = sys.modules.get("forge.inference.scorer")
    sys.modules["forge.inference.scorer"] = stub
    scan_core._SCORER_CACHE.clear()

    claimed: list[bool] = []

    def worker() -> None:
        started.wait()
        claimed.append(scan_core._acquire_scorer("baseline")[1])

    try:
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        scan_core._SCORER_CACHE.clear()
        if real is not None:
            sys.modules["forge.inference.scorer"] = real
        else:
            del sys.modules["forge.inference.scorer"]

    assert sum(claimed) == 1, (
        f"{sum(claimed)} of 4 partitions claimed the load. Exactly one did it; the others "
        "waited on the lock, and a partition that waited must not report the load as its "
        "own or the startup column becomes fiction."
    )


def test_the_second_call_in_a_process_does_not_claim_the_load() -> None:
    """The ordinary sequential case, which is what Spark local mode does: one worker
    process takes several partitions in turn, the first pays for the model and the rest
    must not be counted again."""
    import sys
    import types

    from forge.hard_negative import scan_core

    stub = types.ModuleType("forge.inference.scorer")
    stub.load_arm = lambda arm: object()
    real = sys.modules.get("forge.inference.scorer")
    sys.modules["forge.inference.scorer"] = stub
    scan_core._SCORER_CACHE.clear()
    try:
        first, first_loaded = scan_core._acquire_scorer("baseline")
        second, second_loaded = scan_core._acquire_scorer("baseline")
    finally:
        scan_core._SCORER_CACHE.clear()
        if real is not None:
            sys.modules["forge.inference.scorer"] = real
        else:
            del sys.modules["forge.inference.scorer"]

    assert first is second
    assert first_loaded is True
    assert second_loaded is False


# ---------------------------------------------------------------------------
# THE POOL AUDIT, and the run that produced two hundred lines of traceback.
#
# The column check lived inside the worker, at read time. Pointing the Beam sweep at
# data/silver after the v0.2-min regeneration gave four partitions each dying separately on
# a generated shard, a Beam traceback long enough to bury its own one-line cause, and no
# artifact. ScanPlan already refused an empty file list and an impossible partition count on
# the driver, on the stated principle that an obviously wrong plan should be visible before
# the compute is spent. The schema was the part of "obviously wrong" left to the workers.
#
# The deeper problem was not the schema. data/silver holds the human corpus under source=*/
# and the generated arms under mirrors/ and random/, and the generated shards carry
# `generator`, `label` and `sample_id` instead of `doc_id`. Had they happened to carry a
# doc_id the scan would have run happily and mined AI text as human hard negatives, which
# is the opposite of what the mining loop is for.
# ---------------------------------------------------------------------------


def _write_shard(path, columns: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _human(n: int = 3) -> dict:
    return {"doc_id": [f"d{i}" for i in range(n)],
            "source_group_id": ["g"] * n,
            "text": ["human text"] * n}


def _generated(n: int = 3) -> dict:
    return {"sample_id": [f"s{i}" for i in range(n)],
            "source_group_id": ["g"] * n,
            "generator": ["qwen"] * n,
            "label": ["ai"] * n,
            "text": ["generated text"] * n}


def test_a_generated_shard_is_refused_not_scanned(tmp_path) -> None:
    """THE CORRECTNESS BOUNDARY, and it is not about columns.

    The scan keeps documents the detector calls AI when they are human, so every hit is a
    false positive by construction. On a generated shard every hit is a TRUE positive, and
    mining those as hard negatives would teach the detector to call its own synthetic text
    human. A generated shard in the pool is not a degraded measurement, it is an inverted
    one, so it fails rather than being skipped.
    """
    from forge.hard_negative.scan_core import ScanError, audit_pool

    _write_shard(tmp_path / "source=fw" / "part-000.parquet", _human())
    _write_shard(tmp_path / "mirrors" / "part-qwen.parquet", _generated())

    with pytest.raises(ScanError, match="GENERATED text"):
        audit_pool(sorted(str(p) for p in tmp_path.rglob("*.parquet")))


def test_the_refusal_names_every_offending_shard_not_just_the_first(tmp_path) -> None:
    """Four partitions each reported their own first bad shard and nothing about the other
    three. One driver-side error listing all of them is the difference between one fix and
    four runs."""
    from forge.hard_negative.scan_core import ScanError, audit_pool

    for family in ("falcon", "phi", "qwen", "smollm"):
        _write_shard(tmp_path / "mirrors" / f"part-{family}.parquet", _generated())

    with pytest.raises(ScanError) as caught:
        audit_pool(sorted(str(p) for p in tmp_path.rglob("*.parquet")))
    message = str(caught.value)
    for family in ("falcon", "phi", "qwen", "smollm"):
        assert family in message, f"{family} is not named in the refusal"


def test_the_refusal_says_how_to_narrow_the_scan(tmp_path) -> None:
    """An error that states a rule without stating the fix costs a round trip. The pattern
    that selects the human pool is the actionable half."""
    from forge.hard_negative.scan_core import ScanError, audit_pool

    _write_shard(tmp_path / "mirrors" / "part-qwen.parquet", _generated())
    with pytest.raises(ScanError, match=r"source=\*/split=\*/\*\.parquet"):
        audit_pool([str(next(tmp_path.rglob("*.parquet")))])


def test_a_shard_missing_doc_id_is_refused_on_the_driver(tmp_path) -> None:
    from forge.hard_negative.scan_core import ScanError, audit_pool

    _write_shard(tmp_path / "part-000.parquet",
                 {"source_group_id": ["g"], "text": ["t"]})
    with pytest.raises(ScanError, match="cannot be scanned"):
        audit_pool([str(tmp_path / "part-000.parquet")])


def test_a_file_that_is_not_parquet_is_reported_rather_than_crashing(tmp_path) -> None:
    """data/reserve/ is gitignored, so a fresh checkout's likeliest contents are a README
    or a stray download. The audit must say which file and why."""
    from forge.hard_negative.scan_core import ScanError, audit_pool

    bad = tmp_path / "notes.parquet"
    bad.write_bytes(b"not parquet at all")
    with pytest.raises(ScanError, match="could not be opened"):
        audit_pool([str(bad)])


def test_a_clean_human_pool_audits_and_reports_its_rows(tmp_path) -> None:
    from forge.hard_negative.scan_core import audit_pool

    _write_shard(tmp_path / "a.parquet", _human(4))
    _write_shard(tmp_path / "b.parquet", _human(6))
    pool = audit_pool(sorted(str(p) for p in tmp_path.rglob("*.parquet")))
    assert len(pool.files) == 2
    assert pool.total_rows == 10


def test_the_fingerprint_changes_when_a_shard_is_rewritten_in_place(tmp_path) -> None:
    """THE HOLE IN MY OWN GATE.

    The first cross-runner check compared `documents_scanned` between the two artifacts,
    which is a property of the document cap and not of the pool. The Spark sweep ran over 6
    human shards; the v0.2-min regeneration then added 24 generated shards to the same
    tree. Both sweeps would have reported 40 documents from different corpora and the gate
    would have passed. Row counts are in the hash for the same reason: a regeneration
    rewrites a shard under its own name.
    """
    from forge.hard_negative.scan_core import audit_pool

    shard = tmp_path / "a.parquet"
    _write_shard(shard, _human(4))
    before = audit_pool([str(shard)]).fingerprint

    _write_shard(shard, _human(9))
    after = audit_pool([str(shard)]).fingerprint

    assert before != after, "a rewritten shard must not keep its fingerprint"

    _write_shard(tmp_path / "b.parquet", _human(4))
    with_extra = audit_pool(sorted(str(p) for p in tmp_path.rglob("*.parquet"))).fingerprint
    assert with_extra != after, "an added shard must not keep the fingerprint"


def test_the_fingerprint_is_stable_across_two_audits_of_one_pool(tmp_path) -> None:
    """It has to be, or the gate it feeds rejects every honest re-run."""
    from forge.hard_negative.scan_core import audit_pool

    _write_shard(tmp_path / "a.parquet", _human(4))
    files = sorted(str(p) for p in tmp_path.rglob("*.parquet"))
    assert audit_pool(files).fingerprint == audit_pool(files).fingerprint


def test_plan_scan_selects_only_the_human_shards_when_given_the_pattern(tmp_path) -> None:
    """The fix for the failed sweep, as an assertion, in both directions: the human layout
    selects only human shards, and a recursive glob over the same tree is still refused."""
    from forge.hard_negative.scan_core import ScanError, plan_scan

    _write_shard(tmp_path / "source=fw" / "split=test" / "part-000.parquet", _human())
    _write_shard(tmp_path / "mirrors" / "split=test" / "part-qwen.parquet", _generated())

    plan = plan_scan(str(tmp_path), partitions=1, threshold=0.99,
                     pattern="source=*/split=*/*.parquet")
    assert len(plan.files) == 1
    assert plan.pool is not None and plan.pool.total_rows == 3

    # The refusal now needs the recursive glob passed EXPLICITLY, because the default is
    # the writer's layout rather than "everything under here". That is the point of the
    # change: the dangerous pattern is no longer what you get by not thinking about it.
    # The audit still stands behind it for anyone who does pass it.
    with pytest.raises(ScanError, match="GENERATED text"):
        plan_scan(str(tmp_path), partitions=1, threshold=0.99, pattern="**/*.parquet")


def test_the_default_pattern_comes_from_the_writer_not_from_a_second_opinion() -> None:
    """THE FLAG THAT SHOULD NOT HAVE BEEN A FLAG.

    The scan defaulted to a recursive **/*.parquet glob, which under data/silver takes the
    human corpus and both generated arms, and the Beam sweep died four partitions deep as a
    result. The fix was a --pattern argument: correct, and it put the burden on whoever runs
    the command to know something the repository already knew. forge.ingestion.writer
    defines the layout, forge.training.data reads it back with exactly that glob, and this
    scan was the third reader written as though the layout were unknown.

    If the writer ever changes its layout, this fails rather than silently scanning nothing.
    """
    from forge.hard_negative.scan_core import human_pool_pattern
    from forge.ingestion.writer import PARTITION_GLOB

    assert human_pool_pattern() == PARTITION_GLOB


def test_a_default_plan_over_a_mixed_root_no_longer_needs_to_be_told_the_pattern(
    tmp_path,
) -> None:
    """The regression, stated as the absence of an argument. This is the exact tree that
    produced two hundred lines of Beam traceback."""
    from forge.hard_negative.scan_core import plan_scan

    _write_shard(tmp_path / "source=fw" / "split=test" / "part-000.parquet", _human())
    _write_shard(tmp_path / "mirrors" / "split=test" / "part-qwen.parquet", _generated())

    plan = plan_scan(str(tmp_path), partitions=1, threshold=0.99)
    assert len(plan.files) == 1
    assert plan.files[0].endswith("source=fw/split=test/part-000.parquet")
