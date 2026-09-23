"""Spark and Beam must write the corpus the serial reference writes. Through real workers.

This is the test the whole `crawl_core` arrangement exists to make possible. Each runner
is run on the same two local WET segments, with 2 partitions so that work genuinely
crosses process boundaries, and must produce the same `corpus_fingerprint` and the same
counts as `run_serial`. A runner that sharded by worker, kept the first duplicate it saw,
or lost a partition would still write a plausible corpus; it would not write this one.

Segments are local files rather than a monkeypatched network, because a monkeypatch lives
in the test process and never reaches a Spark Python worker. A runner test built on one
would only ever test the driver.

Skips, rather than fails, without PySpark and a JVM, or without Beam, for the reason the
reserve scan's suites do: CI installs neither, and a machine without the optional extra
should say so rather than go red.
"""

from __future__ import annotations

import shutil

import pytest

from forge.ingestion.crawl_core import plan_crawl, run_serial, training_hashes

from test_crawl_core import CRAWL, policy, segments, training_root  # noqa: F401

COMPARED = (
    "documents_seen", "documents_kept_before_dedup", "duplicates_removed",
    "overlaps_training", "documents_written", "by_split", "corpus_fingerprint",
)


@pytest.fixture
def reference(segments, policy, training_root, tmp_path):  # noqa: F811
    plan = plan_crawl(CRAWL, paths=segments, partitions=2, shards=3)
    return run_serial(plan, tmp_path / "serial", policy, training_hashes(training_root))


def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*.parquet"))


@pytest.mark.skipif(shutil.which("java") is None, reason="Spark needs a JVM")
def test_spark_writes_the_reference_corpus(segments, policy, training_root, reference,  # noqa: F811
                                           tmp_path):
    pytest.importorskip("pyspark", reason="pyspark is in the [distributed] extra")
    from forge.ingestion.spark_crawl import run_spark

    plan = plan_crawl(CRAWL, paths=segments, partitions=2, shards=3)
    report = run_spark(plan, tmp_path / "spark", policy, training_hashes(training_root))

    assert {k: report[k] for k in COMPARED} == {k: reference[k] for k in COMPARED}
    assert _files(tmp_path / "spark") == _files(tmp_path / "serial")
    assert report["segments_read"] == 2


@pytest.mark.parametrize("mode", ["multi_processing", "in_memory"])
def test_beam_writes_the_reference_corpus(segments, policy, training_root, reference,  # noqa: F811
                                          tmp_path, mode):
    pytest.importorskip("apache_beam", reason="apache-beam is in the [distributed] extra")
    from forge.ingestion.beam_crawl import run_beam

    plan = plan_crawl(CRAWL, paths=segments, partitions=2, shards=3)
    report = run_beam(plan, tmp_path / mode, policy, training_hashes(training_root),
                      running_mode=mode)

    assert {k: report[k] for k in COMPARED} == {k: reference[k] for k in COMPARED}
    assert _files(tmp_path / mode) == _files(tmp_path / "serial")
    assert report["segments_read"] == 2
