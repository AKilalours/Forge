"""Selection: which mined failures actually enter the next training set.

Two decisions live here, and both change what the experiment measures.

--------------------------------------------------------------------------------
1. Sample proportionally ACROSS clusters, not globally by confidence.
--------------------------------------------------------------------------------
Global top-k spends the entire budget on whichever single failure mode happens to
produce the most confident errors. The model then overfits that one mode while every
other failure mode goes untouched, and the headline "hard negatives improved robustness"
becomes "we fixed one thing very thoroughly". `select_proportional` guarantees coverage
by giving every cluster a share of the budget floor-weighted by its size, with a minimum
per cluster so small modes are not rounded out of existence.

--------------------------------------------------------------------------------
2. Split mined negatives at the CLUSTER level, not the document level.
--------------------------------------------------------------------------------
This is not in the original plan and it matters.

Mined documents come from the reserve pool, which is outside train/val/test. If all of
them go into training, there is no held-out hard-negative set and no way to measure
whether mining generalized. The obvious fix, splitting the mined documents randomly, is
worse than useless: documents within one cluster are near-identical by construction, so a
random split puts near-duplicates of the same failure on both sides and the held-out
score measures memorization.

So mined failures are split by cluster: some entire failure MODES go to training, others
are held out untouched. A model that improves on held-out clusters has generalized to
failure modes it never saw. A model that only improves on trained clusters has memorized,
and the two are indistinguishable under a document-level split.

This costs coverage, which is a real trade: modes held out for measurement do not get
fixed this round. They can be released into training in a later round, and the ledger
tracks that.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from forge.common.schemas import FailureRecord
from forge.failure_atlas.clustering import NOISE


@dataclass(frozen=True)
class SelectionPolicy:
    max_selected: int = 100_000
    min_per_cluster: int = 10
    include_noise: bool = False           # HDBSCAN outliers: one-offs, not modes
    holdout_cluster_fraction: float = 0.25
    seed: str = "forge-v1"


@dataclass
class Selection:
    train: list[FailureRecord] = field(default_factory=list)
    holdout: list[FailureRecord] = field(default_factory=list)
    train_clusters: list[int] = field(default_factory=list)
    holdout_clusters: list[int] = field(default_factory=list)
    per_cluster_quota: dict[int, int] = field(default_factory=dict)
    dropped_noise: int = 0

    def as_dict(self) -> dict:
        return {
            "n_train": len(self.train), "n_holdout": len(self.holdout),
            "train_clusters": self.train_clusters, "holdout_clusters": self.holdout_clusters,
            "per_cluster_quota": self.per_cluster_quota, "dropped_noise": self.dropped_noise,
        }


def _bucket(cluster_id: int, salt: str) -> float:
    h = hashlib.sha256(f"cluster_{cluster_id}|{salt}".encode()).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


def split_clusters(cluster_ids: list[int], policy: SelectionPolicy) -> tuple[list[int], list[int]]:
    """Deterministic cluster-level holdout. Same clusters held out on every rebuild, so
    round two is comparable to round one."""
    if not cluster_ids:
        return [], []
    holdout = [c for c in cluster_ids if _bucket(c, policy.seed) < policy.holdout_cluster_fraction]
    train = [c for c in cluster_ids if c not in set(holdout)]
    # Never hold out everything, and never hold out nothing when a holdout was asked for.
    if not train:
        train, holdout = holdout, []
    elif not holdout and policy.holdout_cluster_fraction > 0 and len(cluster_ids) > 1:
        smallest = cluster_ids[-1]
        holdout = [smallest]
        train = [c for c in cluster_ids if c != smallest]
    return sorted(train), sorted(holdout)


def proportional_quotas(sizes: dict[int, int], budget: int, min_per_cluster: int) -> dict[int, int]:
    """Largest-remainder apportionment with a floor, so small modes survive rounding."""
    if not sizes:
        return {}
    clusters = sorted(sizes)
    floors = {c: min(min_per_cluster, sizes[c]) for c in clusters}
    used = sum(floors.values())
    if used >= budget:
        # Budget cannot even cover the floors: distribute it round-robin by size.
        quotas = {c: 0 for c in clusters}
        order = sorted(clusters, key=lambda c: -sizes[c])
        i = 0
        while sum(quotas.values()) < budget:
            c = order[i % len(order)]
            if quotas[c] < sizes[c]:
                quotas[c] += 1
            elif all(quotas[x] >= sizes[x] for x in clusters):
                break
            i += 1
        return quotas

    remaining = budget - used
    total = sum(sizes.values())
    exact = {c: remaining * sizes[c] / total for c in clusters}
    quotas = {c: floors[c] + int(exact[c]) for c in clusters}
    # cap at availability, then hand out the remainder by largest fractional part
    for c in clusters:
        quotas[c] = min(quotas[c], sizes[c])
    leftover = budget - sum(quotas.values())
    for c in sorted(clusters, key=lambda c: -(exact[c] - int(exact[c]))):
        if leftover <= 0:
            break
        if quotas[c] < sizes[c]:
            quotas[c] += 1
            leftover -= 1
    return quotas


def select_proportional(
    records: list[FailureRecord],
    labels: np.ndarray,
    policy: SelectionPolicy = SelectionPolicy(),
) -> Selection:
    by_cluster: dict[int, list[FailureRecord]] = {}
    dropped_noise = 0
    for r, c in zip(records, labels.tolist(), strict=True):
        c = int(c)
        if c == NOISE and not policy.include_noise:
            dropped_noise += 1
            continue
        by_cluster.setdefault(c, []).append(r)

    cluster_ids = sorted(by_cluster, key=lambda c: -len(by_cluster[c]))
    train_ids, holdout_ids = split_clusters(cluster_ids, policy)

    train_sizes = {c: len(by_cluster[c]) for c in train_ids}
    quotas = proportional_quotas(train_sizes, policy.max_selected, policy.min_per_cluster)

    sel = Selection(
        train_clusters=sorted(train_ids), holdout_clusters=sorted(holdout_ids),
        per_cluster_quota=quotas, dropped_noise=dropped_noise,
    )
    for c in train_ids:
        # Within a cluster, take the most confident errors: the clearest examples of the mode.
        ranked = sorted(by_cluster[c], key=lambda r: -r.confidence)
        sel.train.extend(ranked[: quotas.get(c, 0)])
    for c in holdout_ids:
        sel.holdout.extend(by_cluster[c])
    return sel


def select_top_k(records: list[FailureRecord], budget: int) -> list[FailureRecord]:
    """The naive alternative, kept so the ablation can measure what proportional buys.

    Not the default. See the module docstring.
    """
    return sorted(records, key=lambda r: -r.confidence)[:budget]
