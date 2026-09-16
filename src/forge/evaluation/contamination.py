"""Contamination checking between training data and external benchmarks.

A single training document that also appears in RAID or MAGE invalidates the external
claim, and overlap raises no error on its own: the model simply scores better and the
number looks like a win.

Two checks, because exact-hash alone is not enough.

**Exact.** Content hash of the normalised text. The normalisation MUST be the identical
function ingestion used, or two byte-different-but-identical documents hash differently
and the overlap is missed. That is why this module imports `normalize` from the cleaning
pipeline rather than reimplementing it. Anyone tempted to write a quick local
`text.lower().strip()` here would produce a check that passes while contamination is
present, which is worse than no check at all.

**Near-duplicate.** MinHash over word shingles. FineWeb and RAID both draw on public web
text, so a training document and a benchmark document can be the same article with
different boilerplate, a different crawl date, or a light rewrite. Those contaminate just
as badly as an exact match and exact hashing misses every one of them.

**The near-duplicate threshold here is deliberately looser than deduplication's.**
Deduplication uses 0.8; contamination defaults to 0.5. The error costs run in opposite
directions:

    dedup false positive        -> two distinct documents merged, training data lost
    dedup false negative        -> a near-duplicate pair straddles a split

    contamination false positive -> a pair gets flagged for manual review, cheap
    contamination false negative -> the external result is invalid and nobody knows

Reusing 0.8 here would be inheriting a threshold tuned for the opposite trade. A short
benchmark document with a sentence of site boilerplate appended sits at a true Jaccard of
about 0.75, which 0.8 misses and 0.5 catches. Over-flagging costs a look; under-flagging
costs the headline claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from forge.cleaning.normalize import normalize
from forge.common.hashing import content_sha256
from forge.dedup.minhash import LshParams, MinHashLSH

# Looser than dedup's 0.8, on purpose. See the module docstring.
CONTAMINATION_NEAR_THRESHOLD = 0.5
DEDUP_NEAR_THRESHOLD = 0.8


def canonical_hash(text: str) -> str:
    """Hash of the text after the SAME normalisation ingestion applied."""
    return content_sha256(normalize(text))


@dataclass
class ContaminationReport:
    benchmark: str
    n_train: int
    n_eval: int
    exact_overlap: list[str] = field(default_factory=list)
    near_overlap: list[tuple[str, str, float]] = field(default_factory=list)
    near_threshold: float = 0.5

    @property
    def clean(self) -> bool:
        return not self.exact_overlap and not self.near_overlap

    def as_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "clean": self.clean,
            "exact_overlap": len(self.exact_overlap),
            "near_overlap": len(self.near_overlap),
            "near_threshold": self.near_threshold,
            "examples_exact": self.exact_overlap[:5],
            "examples_near": [
                {"eval_id": e, "train_id": t, "jaccard": round(j, 3)}
                for e, t, j in self.near_overlap[:5]
            ],
        }

    def raise_if_contaminated(self) -> None:
        if not self.clean:
            raise RuntimeError(
                f"{self.benchmark} is contaminated: {len(self.exact_overlap)} exact and "
                f"{len(self.near_overlap)} near-duplicate overlaps with training data. "
                "Any external number computed on it is invalid."
            )


def check(
    benchmark: str,
    train: Iterable[tuple[str, str]],
    evaluation: Iterable[tuple[str, str]],
    near_threshold: float = CONTAMINATION_NEAR_THRESHOLD,
    check_near: bool = True,
) -> ContaminationReport:
    """train and evaluation are iterables of (doc_id, text)."""
    train = list(train)
    evaluation = list(evaluation)

    by_hash: dict[str, str] = {}
    lsh = MinHashLSH(LshParams(threshold=near_threshold)) if check_near else None
    for doc_id, text in train:
        by_hash.setdefault(canonical_hash(text), doc_id)
        if lsh is not None:
            lsh.add(doc_id, normalize(text))

    rep = ContaminationReport(benchmark, len(train), len(evaluation), near_threshold=near_threshold)
    for doc_id, text in evaluation:
        norm = normalize(text)
        h = canonical_hash(text)
        if h in by_hash:
            rep.exact_overlap.append(doc_id)
            continue
        if lsh is not None:
            hits = lsh.query(norm)
            if hits:
                rep.near_overlap.append((doc_id, hits[0][0], hits[0][1]))
    return rep
