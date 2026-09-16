"""The Failure Atlas.

Every failure, from evaluation or from production, becomes a FailureRecord. Records are
embedded and clustered so failures are described as MODES rather than as a list.

"3,000 false positives" is not actionable. "Cluster 17: formal register, government
source, long paragraphs, 32 percent of all false positives, mean confidence 0.96" tells
you what data to generate next. Clusters are the input to targeted generation, which is
what closes the loop, so this file sits directly on the critical path of the project's
central claim.

Cluster summaries are built from record METADATA (domain, source, register, confidence)
rather than from an LLM reading the text. That is a deliberate limitation: the summaries
are less vivid, but they are reproducible and they cannot hallucinate a failure mode that
is not in the data.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from forge.common.schemas import FailureRecord
from forge.failure_atlas.clustering import NOISE, ClusterResult, cluster, silhouette
from forge.failure_atlas.embedding import Embedder, HashingEmbedder


@dataclass
class ClusterSummary:
    cluster_id: int
    size: int
    share: float
    mean_confidence: float
    top_domains: list[tuple[str, int]]
    top_sources: list[tuple[str, int]]
    top_registers: list[tuple[str, int]]
    failure_types: dict[str, int]
    exemplar_ids: list[str]

    def label(self) -> str:
        """A short human-readable name, from metadata only. No model, no invention."""
        parts = []
        if self.top_registers:
            parts.append(self.top_registers[0][0])
        if self.top_domains:
            parts.append(self.top_domains[0][0])
        if self.top_sources:
            parts.append(f"via {self.top_sources[0][0]}")
        return " / ".join(parts) if parts else f"cluster {self.cluster_id}"

    def as_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id, "label": self.label(), "size": self.size,
            "share": round(self.share, 4), "mean_confidence": round(self.mean_confidence, 4),
            "top_domains": self.top_domains, "top_sources": self.top_sources,
            "top_registers": self.top_registers, "failure_types": self.failure_types,
            "exemplar_ids": self.exemplar_ids,
        }


@dataclass
class Atlas:
    records: list[FailureRecord]
    labels: np.ndarray
    clustering: ClusterResult
    embedder_name: str
    silhouette: float
    summaries: list[ClusterSummary] = field(default_factory=list)

    def quality_warnings(self, min_silhouette: float = 0.4, max_share: float = 0.5) -> list[str]:
        """Reasons not to trust this atlas.

        Selection downstream guarantees coverage of CLUSTERS, not of true failure modes.
        If clustering merged two modes, the smaller one is starved no matter how fair the
        selection policy is, and nothing later in the pipeline can notice. So the atlas
        reports its own weakness rather than presenting clean-looking cluster shares.
        """
        w: list[str] = []
        if not np.isnan(self.silhouette) and self.silhouette < min_silhouette:
            w.append(
                f"silhouette {self.silhouette:.3f} is below {min_silhouette}: clusters are "
                "poorly separated, so distinct failure modes may be merged and small ones "
                "starved by selection. Consider a semantic embedder or density-based clustering."
            )
        if self.summaries and self.summaries[0].share > max_share:
            w.append(
                f"cluster {self.summaries[0].cluster_id} holds {self.summaries[0].share:.0%} "
                "of all failures; it is probably several modes merged together."
            )
        if self.embedder_name == "hashing_v1":
            w.append(
                "embedder is hashing_v1, which captures surface form but not semantics. "
                "Fine for tests, not for a real mining run."
            )
        dupes = [
            lbl for lbl, n in Counter(s.label() for s in self.summaries).items() if n > 1
        ]
        if dupes:
            w.append(
                f"clusters share a metadata label {dupes}: one true mode was likely split "
                "across several clusters, which over-weights it in proportional selection."
            )
        return w

    def cluster_of(self, sample_id: str) -> int:
        for r, c in zip(self.records, self.labels.tolist(), strict=True):
            if r.sample_id == sample_id:
                return int(c)
        raise KeyError(sample_id)

    def as_dict(self) -> dict:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_failures": len(self.records),
            "embedder": self.embedder_name,
            "clustering": self.clustering.as_dict(),
            "silhouette": None if np.isnan(self.silhouette) else round(self.silhouette, 4),
            "quality_warnings": self.quality_warnings(),
            "clusters": [s.as_dict() for s in self.summaries],
        }

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return p


def _top(counter: Counter, n: int = 3) -> list[tuple[str, int]]:
    return [(k, v) for k, v in counter.most_common(n)]


def summarize(records: list[FailureRecord], labels: np.ndarray, top_n: int = 3) -> list[ClusterSummary]:
    total = int((labels != NOISE).sum())
    out: list[ClusterSummary] = []
    for c in sorted({int(x) for x in labels.tolist()} - {NOISE}):
        idx = [i for i, x in enumerate(labels.tolist()) if x == c]
        rs = [records[i] for i in idx]
        out.append(
            ClusterSummary(
                cluster_id=c,
                size=len(rs),
                share=len(rs) / total if total else 0.0,
                mean_confidence=float(np.mean([r.confidence for r in rs])),
                top_domains=_top(Counter(r.domain for r in rs), top_n),
                top_sources=_top(Counter(r.source for r in rs), top_n),
                top_registers=_top(Counter(r.text_register or "unknown" for r in rs), top_n),
                failure_types=dict(Counter(r.failure_type for r in rs)),
                # Highest-confidence errors: the most informative examples in the mode.
                exemplar_ids=[r.sample_id for r in sorted(rs, key=lambda r: -r.confidence)[:5]],
            )
        )
    return sorted(out, key=lambda s: -s.size)


def build_atlas(
    records: list[FailureRecord],
    texts: list[str],
    embedder: Embedder | None = None,
    method: str = "auto",
    min_cluster_size: int = 25,
    k: int | None = None,
    seed: int = 42,
) -> Atlas:
    if len(records) != len(texts):
        raise ValueError("records and texts must align")
    if not records:
        raise ValueError("cannot build an atlas from zero failures")
    emb = embedder or HashingEmbedder()
    x = emb.embed(texts)
    res = cluster(x, method=method, min_cluster_size=min_cluster_size, k=k, seed=seed)
    return Atlas(
        records=records, labels=res.labels, clustering=res, embedder_name=emb.name,
        silhouette=silhouette(x, res.labels), summaries=summarize(records, res.labels),
    )
