"""Hard negative mining, the Failure Atlas, and selection.

None of this has met a real failure yet. These tests use synthetic score distributions
with KNOWN failure structure, which is the only way to check that mining finds the
structure that is actually there rather than the structure the model happens to like.
"""

import numpy as np
import pytest

from forge.common.schemas import FailureRecord, Label
from forge.failure_atlas.atlas import build_atlas
from forge.failure_atlas.clustering import NOISE, kmeans, silhouette
from forge.failure_atlas.embedding import HashingEmbedder
from forge.hard_negative.mining import MinedLedger, ReserveDoc, scan, scan_false_negatives
from forge.hard_negative.selection import (
    SelectionPolicy,
    proportional_quotas,
    select_proportional,
    select_top_k,
    split_clusters,
)

# Three genuinely different human registers. The detector below is built to fail on
# exactly one of them, which is the structure the atlas must recover.
REGISTERS = {
    "academic": "Citation-heavy academic prose with long paragraphs and a formal register, item {i}, discussing the methodology at length.",
    "casual": "hey so basically the thing is you gotta just try it out and see what happens, number {i}, no big deal really",
    "legal": "SECTION {i}. PURSUANT TO THE FOREGOING PROVISIONS THE COMMITTEE SHALL DETERMINE THE APPLICABLE THRESHOLD.",
}


def _reserve(n_per=80):
    docs = []
    for reg, tpl in REGISTERS.items():
        for i in range(n_per):
            docs.append(ReserveDoc(f"{reg}_{i}", f"grp_{reg}_{i}", tpl.format(i=i), "fineweb", "web", reg))
    return docs


class BiasedScorer:
    """Confidently wrong on academic prose, correct on everything else.

    This is a realistic failure: formal, complex, citation-heavy human writing is the
    register production detectors most often mistake for AI.
    """

    model_version = "forge-0.2-test"

    def __init__(self, bad_register="academic", bad_score=0.97, good_score=0.03):
        self.bad, self.bad_score, self.good_score = bad_register, bad_score, good_score

    def score(self, texts):
        return [self.bad_score if "Citation-heavy" in t else self.good_score for t in texts]


# ------------------------------------------------------------------ mining scan

def test_mining_finds_the_failure_mode_that_is_actually_there():
    recs, stats = scan(_reserve(), BiasedScorer(), operating_threshold=0.5, min_confidence=0.90)
    assert len(recs) == 80
    assert all(r.text_register == "academic" for r in recs)
    assert stats.scanned == 240 and stats.errors == 80


def test_low_confidence_errors_are_not_mined():
    """A document called AI at 0.51 is a coin flip near the threshold, not a failure mode."""
    recs, stats = scan(_reserve(), BiasedScorer(bad_score=0.55), 0.5, min_confidence=0.90)
    assert recs == []
    assert stats.errors == 80, "they are still production errors"
    assert stats.above_threshold == 0, "but none clear the mining confidence gate"


def test_confidence_gate_below_the_operating_threshold_is_rejected():
    """Otherwise mining selects documents the model did not actually get wrong."""
    with pytest.raises(ValueError):
        scan(_reserve(), BiasedScorer(), operating_threshold=0.9, min_confidence=0.5)


def test_the_ledger_stops_round_two_re_mining_round_one():
    """Without this, the training set fills with duplicates of the easiest-to-find mode
    and the measured improvement is an artifact of oversampling."""
    ledger = MinedLedger()
    r1, _ = scan(_reserve(), BiasedScorer(), 0.5, 0.90, ledger=ledger)
    ledger.add(r.sample_id for r in r1)
    r2, s2 = scan(_reserve(), BiasedScorer(), 0.5, 0.90, ledger=ledger)
    assert r2 == []
    assert s2.already_mined == 80


def test_ledger_round_trips_through_disk(tmp_path):
    p = tmp_path / "mined.json"
    a = MinedLedger(p)
    a.add(["x", "y"])
    a.save("round_1")
    assert set(MinedLedger(p)._ids) == {"x", "y"}


def test_false_negatives_are_mined_too():
    """A loop that only mines false positives drives FPR down while FNR quietly climbs."""
    ai_docs = [ReserveDoc(f"ai_{i}", f"grp_ai_{i}", "some machine text here", "mirror", "web") for i in range(30)]

    class MissesEverything:
        model_version = "m"

        def score(self, texts):
            return [0.02] * len(texts)

    recs, stats = scan_false_negatives(ai_docs, MissesEverything(), 0.5, max_confidence=0.10)
    assert len(recs) == 30
    assert all(r.failure_type == "ai_false_negative" for r in recs)
    assert all(r.true_label is Label.AI and r.prediction is Label.HUMAN for r in recs)


# ------------------------------------------------------------------ atlas

def _mixed_failures(per_mode=40):
    """Failures spread across three modes with different sizes and confidences."""
    recs, texts = [], []
    spec = [("academic", per_mode * 3, 0.97), ("legal", per_mode, 0.93), ("casual", per_mode // 2, 0.91)]
    for reg, n, conf in spec:
        for i in range(n):
            recs.append(
                FailureRecord(
                    sample_id=f"{reg}_{i}", true_label=Label.HUMAN, prediction=Label.AI,
                    confidence=conf - (i % 10) * 0.001, domain="web", source="fineweb",
                    text_register=reg, model_version="m", failure_type="human_false_positive",
                    discovered_at="2026-08-31T00:00:00Z", discovered_by="r1",
                )
            )
            texts.append(REGISTERS[reg].format(i=i))
    return recs, texts


def test_atlas_recovers_the_real_failure_modes():
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    # each cluster should be dominated by one register
    for s in atlas.summaries:
        top_reg, count = s.top_registers[0]
        assert count / s.size > 0.9, f"cluster {s.cluster_id} is mixed: {s.top_registers}"


def test_atlas_labels_name_the_mode_from_metadata_only():
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    labels = {s.label() for s in atlas.summaries}
    assert any("academic" in x for x in labels)


def test_cluster_shares_sum_to_one():
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    assert abs(sum(s.share for s in atlas.summaries) - 1.0) < 1e-9


def test_atlas_is_deterministic():
    recs, texts = _mixed_failures()
    a = build_atlas(recs, texts, k=3)
    b = build_atlas(recs, texts, k=3)
    assert a.labels.tolist() == b.labels.tolist()


def test_atlas_records_which_clustering_method_ran():
    """A report must not present k-means output as if it were density-based."""
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    assert atlas.clustering.method in ("kmeans", "hdbscan")
    assert atlas.as_dict()["clustering"]["method"] == atlas.clustering.method


def test_empty_atlas_is_an_error_not_an_empty_report():
    with pytest.raises(ValueError):
        build_atlas([], [])


# ------------------------------------------------------------------ selection

def test_proportional_selection_covers_every_mode_and_top_k_does_not():
    """The central claim of Phase 4.

    Global top-k spends the whole budget on whichever mode produces the most confident
    errors. Proportional sampling guarantees every mode gets a share, so the retrained
    model sees all of them.
    """
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    budget = 60

    prop = select_proportional(
        recs, atlas.labels, SelectionPolicy(max_selected=budget, holdout_cluster_fraction=0.0)
    )
    topk = select_top_k(recs, budget)

    prop_modes = {r.text_register for r in prop.train}
    topk_modes = {r.text_register for r in topk}

    assert len(prop_modes) == 3, f"proportional missed a mode: {prop_modes}"
    assert len(topk_modes) == 1, f"top-k unexpectedly spread out: {topk_modes}"


def test_small_modes_are_not_rounded_out_of_existence():
    sizes = {0: 1000, 1: 20, 2: 5}
    q = proportional_quotas(sizes, budget=100, min_per_cluster=10)
    assert q[2] >= 5, "the smallest mode must still be represented"
    assert sum(q.values()) == 100


def test_quotas_never_exceed_availability():
    sizes = {0: 3, 1: 4}
    q = proportional_quotas(sizes, budget=100, min_per_cluster=10)
    assert q[0] <= 3 and q[1] <= 4


def test_quotas_respect_a_budget_smaller_than_the_floors():
    sizes = {0: 100, 1: 100, 2: 100}
    q = proportional_quotas(sizes, budget=5, min_per_cluster=10)
    assert sum(q.values()) == 5


# --------------------------------------------- cluster-level holdout

def test_holdout_is_by_cluster_not_by_document():
    """A document-level split puts near-duplicates of one failure on both sides, so the
    held-out score measures memorization rather than generalization."""
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    sel = select_proportional(recs, atlas.labels, SelectionPolicy(max_selected=1000))
    assert sel.holdout_clusters, "nothing was held out"
    assert not (set(sel.train_clusters) & set(sel.holdout_clusters))
    train_ids = {r.sample_id for r in sel.train}
    hold_ids = {r.sample_id for r in sel.holdout}
    assert not (train_ids & hold_ids)


def test_held_out_modes_are_entirely_absent_from_training():
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    sel = select_proportional(recs, atlas.labels, SelectionPolicy(max_selected=1000))
    by_id = {r.sample_id: c for r, c in zip(recs, atlas.labels.tolist(), strict=True)}
    for r in sel.train:
        assert by_id[r.sample_id] not in sel.holdout_clusters


def test_cluster_split_is_deterministic_so_rounds_are_comparable():
    ids = list(range(12))
    a = split_clusters(ids, SelectionPolicy())
    b = split_clusters(ids, SelectionPolicy())
    assert a == b


def test_split_never_holds_out_everything():
    train, hold = split_clusters([0], SelectionPolicy(holdout_cluster_fraction=1.0))
    assert train and not hold


def test_noise_points_are_dropped_by_default():
    """HDBSCAN outliers are one-off oddities, not failure modes worth generating against."""
    recs, texts = _mixed_failures(per_mode=10)
    labels = np.array([NOISE] * 5 + [0] * (len(recs) - 5))
    sel = select_proportional(recs, labels, SelectionPolicy(max_selected=100, holdout_cluster_fraction=0.0))
    assert sel.dropped_noise == 5
    noise_ids = {r.sample_id for r in recs[:5]}
    assert not ({r.sample_id for r in sel.train} & noise_ids)


def test_kmeans_rejects_more_clusters_than_points():
    with pytest.raises(ValueError):
        kmeans(np.zeros((3, 4)), k=5)


def test_silhouette_is_nan_for_a_single_cluster():
    x = HashingEmbedder().embed(["a b c d", "a b c e"])
    assert np.isnan(silhouette(x, np.array([0, 0])))


def test_atlas_reports_its_own_weakness():
    """Selection guarantees coverage of CLUSTERS, not of true failure modes. If
    clustering merged two modes, the smaller one is starved no matter how fair the
    selection policy is, and nothing downstream can notice. So the atlas says so."""
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=3)
    warnings = atlas.quality_warnings()
    assert any("hashing_v1" in w for w in warnings), "test embedder should be flagged"
    assert "quality_warnings" in atlas.as_dict()


def test_a_dominant_cluster_is_flagged_as_probably_merged():
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=2)   # too few clusters on purpose
    assert any("merged together" in w or "poorly separated" in w for w in atlas.quality_warnings())


def test_duplicate_cluster_labels_are_flagged_as_a_split_mode():
    """One true mode spread over several clusters over-weights it in proportional
    selection, because each of its clusters draws its own share of the budget."""
    recs, texts = _mixed_failures()
    atlas = build_atlas(recs, texts, k=6)   # more clusters than true modes
    assert any("split across several clusters" in w for w in atlas.quality_warnings())


def test_an_explicit_k_always_means_kmeans_regardless_of_environment():
    """Regression guard for a bug that only appeared on the GPU pod.

    cluster(method="auto") preferred HDBSCAN when it was importable, and silently ignored
    an explicit `k`. On a laptop without the mining extra, k-means ran; on the pod, where
    the extra provides HDBSCAN, the same call with the same k ran a different algorithm.
    Two tests passed in one environment and failed in the other, from identical code.

    `k` is a k-means parameter. Supplying one is a request for k-means.
    """
    recs, texts = _mixed_failures(per_mode=20)
    atlas = build_atlas(recs, texts, k=3)
    assert atlas.clustering.method == "kmeans"
    assert atlas.clustering.n_clusters == 3


def test_without_k_the_method_is_whatever_is_available_and_is_recorded():
    """A report must never present k-means output as if it were density-based."""
    recs, texts = _mixed_failures(per_mode=20)
    atlas = build_atlas(recs, texts, min_cluster_size=10)
    assert atlas.clustering.method in ("kmeans", "hdbscan")
    assert atlas.as_dict()["clustering"]["method"] == atlas.clustering.method
