"""The fixed-order cleaning pipeline. data_spec_v1 section 7.

Order is a correctness property here, not a style choice. Two orderings in particular
are load-bearing:

  markup removal BEFORE length filtering
      A 3,000-character page that is 2,900 characters of HTML is a 100-character
      document. Filtering first keeps junk and drops good short documents.

  deduplication BEFORE split assignment
      Deduplicating afterwards leaves near-duplicate pairs straddling train and test.
      That is the same leak as a row-level split, wearing a different hat, and it does
      not raise an error. It just makes every number better than it should be.

Every rejected document is counted by reason. A pipeline that silently drops 60 percent
of its input is a bug you want to see in the report, not discover later.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from forge.cleaning import langid, pii
from forge.cleaning.filters import (
    LengthPolicy,
    QualityPolicy,
    approx_token_count,
    check_length,
    check_quality,
)
from forge.cleaning.normalize import normalize
from forge.common.hashing import content_sha256
from forge.common.schemas import HumanDocument, Quality
from forge.common.splits import assign_split, group_id_for
from forge.dedup.exact import ExactDeduper
from forge.dedup.minhash import LshParams, MinHashLSH
from forge.ingestion.sources import RawRecord

PROCESSING_VERSION = "clean_v1"

STAGES = [
    "schema_validation",
    "unicode_normalization",
    "markup_removal",
    "length_filter",
    "language_id",
    "quality_scoring",
    "pii_filter",
    "length_recheck_after_redaction",
    "exact_dedup",
    "near_duplicate_dedup",
    "domain_classification",
    "split_assignment",
    "parquet_write",
]


@dataclass
class CleaningPolicy:
    length: LengthPolicy = field(default_factory=LengthPolicy)
    quality: QualityPolicy = field(default_factory=QualityPolicy)
    lsh: LshParams = field(default_factory=LshParams)
    language: str = "en"
    language_score_min: float = 0.85
    redistributable_licenses: frozenset[str] = frozenset(
        {"public-domain-us", "public-domain-us-gov", "synthetic"}
    )


@dataclass
class CleaningStats:
    seen: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    # Carried into the report and the MANIFEST so a corpus cleaned WITHOUT the global
    # dedup stages can never be mistaken for one cleaned with them. See Cleaner.
    dedup_applied: bool = True

    @property
    def keep_rate(self) -> float:
        return self.kept / self.seen if self.seen else 0.0

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "keep_rate": round(self.keep_rate, 4),
            "rejected": dict(self.rejected),
            "dedup_applied": self.dedup_applied,
        }


class Cleaner:
    """Stateful because dedup needs memory of everything seen so far in this run.

    THAT STATEFULNESS IS WHY THIS CLASS CANNOT SIMPLY BE HANDED TO A DISTRIBUTED RUNNER.
    Every other stage here is a pure function of one document. Exact and near-duplicate
    dedup are not: they are functions of one document AND everything seen before it. Give
    each Spark or Beam worker its own Cleaner and each one deduplicates only against its
    own partition, so a page that appears in two segments survives twice. That is not a
    thrown error, it is a corpus with duplicates straddling train and test, which is the
    exact leak the fixed stage order in this module exists to prevent, and it would make
    every reported number better than it should be.

    So `dedup=False` exists: it runs the per-document stages and stops before the two
    global ones, which is what a mining worker should do. Deduplication then happens once,
    over everything, as a shuffle in the pipeline that owns the whole corpus. The ordering
    still lives in exactly one place, which is the point; a second copy of this sequence
    written for the distributed path is how the two would drift.

    A corpus cleaned with `dedup=False` is NOT finished, and `stats.dedup_applied` records
    that so a MANIFEST cannot claim otherwise.
    """

    def __init__(self, policy: CleaningPolicy | None = None, *, dedup: bool = True) -> None:
        self.policy = policy or CleaningPolicy()
        self.dedup = dedup
        self.stats = CleaningStats(dedup_applied=dedup)
        self._exact = ExactDeduper()
        self._lsh = MinHashLSH(self.policy.lsh)

    def process(self, records: Iterable[RawRecord]) -> Iterator[HumanDocument]:
        for rec in records:
            self.stats.seen += 1
            doc = self._process_one(rec)
            if doc is not None:
                self.stats.kept += 1
                yield doc

    def _reject(self, reason: str) -> None:
        self.stats.rejected[reason] += 1

    def _process_one(self, rec: RawRecord) -> HumanDocument | None:
        # 1. schema validation
        if not rec.text or not rec.source_record_id:
            self._reject("empty_or_unidentified")
            return None

        # 3 and 4. normalization and markup removal (before any length judgement)
        text = normalize(rec.text)
        if not text:
            self._reject("empty_after_normalization")
            return None

        # 5. length, BEFORE language id. See the spec changelog for v1.1.
        # Language detection on a two-word string is meaningless, and attributing
        # "Too short." to a language failure hides the real reason in the report.
        # Length is also the cheapest gate, so at 5M documents it belongs first.
        if reasons := check_length(text, self.policy.length):
            self._reject(reasons[0])
            return None

        # 2. language id (upstream score preferred, see langid.py)
        lang = langid.detect(text, upstream=rec.upstream_language)
        if lang.language != self.policy.language or lang.score < self.policy.language_score_min:
            self._reject("language")
            return None

        # 6. quality
        if reasons := check_quality(text, self.policy.quality):
            self._reject(reasons[0])
            return None

        # 7. PII redaction (before hashing, so the hash matches what we store)
        text, pii_flags = pii.redact(text)

        # Re-check length AFTER redaction. Redaction can shrink the word count: a phone
        # number written "(555) 123-4567" is two whitespace tokens that collapse to one
        # "[PHONE]". A document sitting at exactly the floor therefore lands below it,
        # and the stored token_count (computed post-redaction) then disagrees with the
        # gate that admitted it. Found by assert_corpus_matches_policy on a real run:
        # 1 of 60,000 documents was written at 147 tokens against a floor of 150.
        #
        # The cheap pre-filter stays where it is so expensive stages are skipped early;
        # this is the final check that makes the gate and the stored value agree.
        if reasons := check_length(text, self.policy.length):
            self._reject(f"{reasons[0]}_after_redaction")
            return None

        doc_id = f"{rec.source_id}_{rec.source_record_id}"
        sha = content_sha256(text)

        # 8 and 9. exact then near-duplicate dedup. The only two stages that depend on
        # documents other than this one, and therefore the only two a per-partition worker
        # must not run. See the class docstring.
        if self.dedup:
            if self._exact.is_duplicate(doc_id, text) is not None:
                self._reject("exact_duplicate")
                return None

            if self._lsh.add_if_new(doc_id, text) is not None:
                self._reject("near_duplicate")
                return None

        # 10, 11. domain/register carried from the source, then split assignment
        group = group_id_for(doc_id)
        return HumanDocument(
            doc_id=doc_id,
            source_group_id=group,
            text=text,
            source=rec.source_id,
            source_config=rec.source_config,
            source_record_id=rec.source_record_id,
            license=rec.license,
            domain=rec.domain,
            text_register=rec.register,
            language=lang.language,
            language_score=lang.score,
            date=rec.date,
            acquired_at=rec.acquired_at,
            processing_version=PROCESSING_VERSION,
            content_sha256=sha,
            token_count=approx_token_count(text),
            quality=Quality(
                edu_score=rec.extra.get("edu_score"),
                length_ok=True,
                pii_flags=pii_flags,
            ),
            split=assign_split(group),
            redistributable=rec.license in self.policy.redistributable_licenses,
        )


def run(records: Iterable[RawRecord], policy: CleaningPolicy | None = None):
    cleaner = Cleaner(policy)
    docs = list(cleaner.process(records))
    return docs, cleaner.stats
