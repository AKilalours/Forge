"""Hard negative mining. The heart of FORGE.

    human reserve pool -> current detector -> high-confidence false positives
      -> embed -> cluster (Failure Atlas) -> sample across clusters
      -> targeted mirrors -> retrain

The claim under test in this whole project is that choosing WHICH synthetic data to
generate, based on where the current model actually fails, buys more robustness per
example than generating more data at random. Everything here exists to make that
comparison fair.

Three properties this module is built to guarantee.

**The reserve pool is never trained on wholesale.** Only documents mining selects, and
the mirrors generated from them, enter training. Training on the whole pool would be
"more data", which is the baseline this is supposed to beat.

**Confidence gating, not just error counting.** A document the model called AI at 0.51
is a coin flip near the threshold and tells you nothing about a failure mode. One it
called AI at 0.97 is the model being confidently wrong, which is both the expensive
error in production and the informative one for training.

**Mined ids are remembered across rounds.** The flywheel turns repeatedly. Without a
ledger, round two re-mines the same easy-to-find failures, the training set fills with
duplicates of one mode, and the measured improvement is an artifact of oversampling.

Nothing in this file has met a real failure yet. It is tested against synthetic score
distributions, and it stays untested against real ones until Phase 3 trains a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from forge.common.schemas import FailureRecord, Label


@runtime_checkable
class Scorer(Protocol):
    """Anything that turns text into an AI probability.

    A protocol rather than a concrete detector so mining is testable with a fake scorer
    that has known failure structure. Testing mining against a real model would only tell
    you about that model.
    """

    model_version: str

    def score(self, texts: list[str]) -> list[float]: ...


@dataclass
class ReserveDoc:
    doc_id: str
    source_group_id: str
    text: str
    source: str
    domain: str
    register: str = "informational"


@dataclass
class MiningStats:
    scanned: int = 0
    errors: int = 0
    above_threshold: int = 0
    already_mined: int = 0
    selected: int = 0

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "errors_at_operating_threshold": self.errors,
            "above_confidence_gate": self.above_threshold,
            "skipped_already_mined": self.already_mined,
            "selected": self.selected,
            "error_rate": round(self.errors / self.scanned, 6) if self.scanned else 0.0,
        }


class MinedLedger:
    """Remembers which reserve documents previous rounds already took."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._ids: set[str] = set()
        if self.path and self.path.exists():
            self._ids = set(json.loads(self.path.read_text()).get("mined", []))

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._ids

    def add(self, doc_ids: Iterable[str]) -> None:
        self._ids.update(doc_ids)

    def save(self, round_name: str) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"updated_at": datetime.now(timezone.utc).isoformat(),
                 "last_round": round_name, "mined": sorted(self._ids)},
                indent=2,
            )
            + "\n"
        )

    def __len__(self) -> int:
        return len(self._ids)


def scan(
    reserve: Iterable[ReserveDoc],
    scorer: Scorer,
    operating_threshold: float,
    min_confidence: float = 0.90,
    ledger: MinedLedger | None = None,
    batch_size: int = 64,
    round_name: str = "mining_run_001",
) -> tuple[list[FailureRecord], MiningStats]:
    """Find high-confidence false positives: human documents the model calls AI.

    `operating_threshold` is the production threshold from the evaluation lab, so
    "error" here means the same thing it means in production. `min_confidence` is the
    separate, stricter gate for what is worth mining.
    """
    if min_confidence < operating_threshold:
        raise ValueError(
            f"min_confidence {min_confidence} is below the operating threshold "
            f"{operating_threshold}; the mining gate must be at least as strict as "
            "the production decision, or it selects documents that were not errors"
        )

    stats = MiningStats()
    out: list[FailureRecord] = []
    now = datetime.now(timezone.utc)

    for batch in _batched(reserve, batch_size):
        scores = scorer.score([d.text for d in batch])
        for doc, s in zip(batch, scores, strict=True):
            stats.scanned += 1
            if s < operating_threshold:
                continue
            stats.errors += 1          # a false positive in production terms
            if s < min_confidence:
                continue
            stats.above_threshold += 1
            if ledger is not None and doc.doc_id in ledger:
                stats.already_mined += 1
                continue
            out.append(
                FailureRecord(
                    sample_id=doc.doc_id,
                    true_label=Label.HUMAN,
                    prediction=Label.AI,
                    confidence=float(s),
                    domain=doc.domain,
                    source=doc.source,
                    text_register=doc.register,
                    model_version=scorer.model_version,
                    failure_type="human_false_positive",
                    discovered_at=now,
                    discovered_by=round_name,
                )
            )
    stats.selected = len(out)
    return out, stats


def _batched(it: Iterable[ReserveDoc], n: int) -> Iterator[list[ReserveDoc]]:
    buf: list[ReserveDoc] = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def scan_false_negatives(
    ai_pool: Iterable[ReserveDoc],
    scorer: Scorer,
    operating_threshold: float,
    max_confidence: float = 0.10,
    round_name: str = "mining_run_001",
) -> tuple[list[FailureRecord], MiningStats]:
    """The other half of the atlas: AI documents the model confidently calls human.

    Symmetric to `scan`, and easy to forget. A loop that only mines false positives
    drives FPR down while FNR quietly climbs, and the release gate catches that only
    after the fact.
    """
    stats = MiningStats()
    out: list[FailureRecord] = []
    now = datetime.now(timezone.utc)
    for batch in _batched(ai_pool, 64):
        for doc, s in zip(batch, scorer.score([d.text for d in batch]), strict=True):
            stats.scanned += 1
            if s >= operating_threshold:
                continue
            stats.errors += 1
            if s > max_confidence:
                continue
            stats.above_threshold += 1
            out.append(
                FailureRecord(
                    sample_id=doc.doc_id, true_label=Label.AI, prediction=Label.HUMAN,
                    confidence=float(1.0 - s), domain=doc.domain, source=doc.source,
                    text_register=doc.register, model_version=scorer.model_version,
                    failure_type="ai_false_negative", discovered_at=now, discovered_by=round_name,
                )
            )
    stats.selected = len(out)
    return out, stats
