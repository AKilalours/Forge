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
