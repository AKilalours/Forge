"""Phase 1 orchestration: config in, FORGE-HUMAN out.

One design point worth stating. Deduplication is shared across all sources in a run,
not per source. FineWeb and FineWeb-Edu are both derived from Common Crawl, so the same
page can appear in both. Deduping per source would let that pair through, and the pair
would land in different splits, which is the leak this whole pipeline is built to
prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge.cleaning.filters import LengthPolicy, QualityPolicy
from forge.cleaning.pipeline import Cleaner, CleaningPolicy
from forge.common.config import REPO_ROOT, load
from forge.common.schemas import HumanDocument
from forge.common.splits import check_no_group_leakage
from forge.ingestion import sources as src
from forge.ingestion.writer import write_manifest, write_metadata_only, write_parquet


@dataclass
class IngestResult:
    docs: list[HumanDocument]
    stats_by_source: dict[str, dict]
    partitions: dict[str, int]
    manifest_path: Path | None
    reserve_docs: int = 0
    reserve_partitions: dict[str, int] = field(default_factory=dict)
    shortfalls: dict[str, dict] = field(default_factory=dict)


def _build_source(entry: dict, local_root: Path | None):
    sid = entry["id"]
    if sid in ("fw", "fwe"):
        cls = src.FineWebSource if sid == "fw" else src.FineWebEduSource
        return cls(config=entry.get("config", "sample-10BT"), filters=entry.get("filters"))
    if sid == "gut":
        return src.GutenbergSource(entry.get("path") or (local_root or Path(".")) / "gutenberg")
    if sid == "gov":
        return src.GovInfoSource(entry.get("path") or (local_root or Path(".")) / "govinfo")
    if sid == "cc":
        from forge.ingestion.commoncrawl import CommonCrawlSource

        return CommonCrawlSource(
            entry["crawl"],
            fmt=entry.get("format", "wet"),
            segments=int(entry.get("segments", 1)),
            segment_paths_override=entry.get("paths"),
        )
    if sid == "local":
        return src.LocalJSONLSource(
            entry["path"],
            source_id=entry.get("source_id", "local"),
            license=entry.get("license", "unknown"),
            domain=entry.get("domain", "web"),
            register=entry.get("register", "informational"),
        )
    raise ValueError(f"unknown source id {sid!r}")


def policy_from_config(cfg: dict) -> CleaningPolicy:
    """Build the cleaning policy FROM the config.

    This function exists because of a real bug. The runner previously contained the line

        policy.length.__dict__   # frozen dataclasses; defaults come from the spec

    which is a no-op expression statement. It reads an attribute and discards it. Every
    value under `cleaning:` in the config was silently ignored and the LengthPolicy
    defaults were used instead: min_chars 200, min_tokens 50, max_tokens 20000, rather
    than the configured 800 / 150 / 400.

    The consequence was invisible in the run output. Rejection counts looked healthy, the
    quota was met, and nothing errored. Only the token-count distribution of the WRITTEN
    corpus revealed it: 65 percent of documents fell outside the configured range and one
    document was 19,639 tokens, which is 51 training windows all carrying one label.

    LengthPolicy and QualityPolicy are frozen, so they must be constructed, not mutated.
    That is exactly what the original line failed to do.
    """
    c = cfg.get("cleaning", {})
    base_len, base_q = LengthPolicy(), QualityPolicy()
    return CleaningPolicy(
        length=LengthPolicy(
            min_chars=c.get("min_chars", base_len.min_chars),
            min_tokens=c.get("min_tokens", base_len.min_tokens),
            max_tokens=c.get("max_tokens", base_len.max_tokens),
        ),
        quality=QualityPolicy(
            max_symbol_ratio=c.get("max_symbol_ratio", base_q.max_symbol_ratio),
            max_repeated_line_ratio=c.get("max_repeated_line_ratio", base_q.max_repeated_line_ratio),
            min_mean_word_length=c.get("min_mean_word_length", base_q.min_mean_word_length),
            max_mean_word_length=c.get("max_mean_word_length", base_q.max_mean_word_length),
            min_unique_word_ratio=c.get("min_unique_word_ratio", base_q.min_unique_word_ratio),
        ),
        language=c.get("language", "en"),
        language_score_min=c.get("language_score_min", 0.85),
    )


def assert_corpus_matches_policy(docs: list[HumanDocument], policy: CleaningPolicy) -> None:
    """Post-condition: every written document must satisfy the filter that was asked for.

    Rejection accounting alone could not catch the bug above, because the counts were
    accurate for the policy that actually ran. What was missing was a check that the
    policy which ran was the policy that was requested. This closes that: it validates the
    OUTPUT against the CONFIG rather than trusting the plumbing between them.
    """
    bad = [
        d for d in docs
        if not (policy.length.min_tokens <= d.token_count <= policy.length.max_tokens)
    ]
    if bad:
        lo = min(d.token_count for d in bad)
        hi = max(d.token_count for d in bad)
        raise RuntimeError(
            f"{len(bad)} of {len(docs)} written documents violate the configured length "
            f"policy ({policy.length.min_tokens}-{policy.length.max_tokens} tokens); "
            f"observed {lo} to {hi}. The configured filter did not reach the cleaner."
        )


def _quota(entry: dict, total: int | None) -> int | None:
    """Per-source document budget from its train_share."""
    if total is None:
        return None
    share = entry.get("train_share")
    return int(round(total * share)) if share else None


def ingest(
    config_path: str,
    out_root: str | Path | None = None,
    total: int | None = None,
    with_reserve: bool = False,
    reserve_root: str | Path | None = None,
    read_multiplier: int = 20,
) -> IngestResult:
    """Build the training pool, and optionally the reserve pool in the SAME pass.

    The reserve pool must be disjoint from training data: mining scans it for false
    positives, and a reserve document that is also a training document is one the model
    has already memorised, so it will not be a false positive and the mining pass finds
    nothing. Building both pools in one pass with ONE Cleaner is what guarantees that,
    because deduplication is global across the run. Two separate commands over the same
    stream would produce two overlapping pools.

    Reserve documents get a group id but NO split, because they sit outside train, val
    and test entirely.
    """
    cfg = load(config_path)
    out_root = Path(out_root) if out_root else REPO_ROOT / "data" / "silver"
    total = total if total is not None else cfg.get("targets", {}).get("human_train")

    policy = policy_from_config(cfg)

    # One cleaner for the whole run: dedup must be global across sources.
    cleaner = Cleaner(policy)
    all_docs: list[HumanDocument] = []
    stats_by_source: dict[str, dict] = {}

    reserve_target = cfg.get("targets", {}).get("human_reserve", 0) if with_reserve else 0
    reserve_docs: list[HumanDocument] = []
    shortfalls: dict[str, dict] = {}

    for entry in cfg.get("sources", []):
        before_seen, before_kept = cleaner.stats.seen, cleaner.stats.kept
        before_rejected = cleaner.stats.rejected.copy()

        source = _build_source(entry, REPO_ROOT / "data" / "raw")
        quota = _quota(entry, total)
        res_quota = _quota(entry, reserve_target) or 0
        want = (quota or 0) + res_quota
        # Read more than the quota because most documents are rejected. With an
        # aggressive length filter the keep rate can be a few percent, so this multiplier
        # is the difference between hitting the target and quietly under-delivering.
        read_limit = want * read_multiplier if want else None

        produced = 0
        for doc in cleaner.process(source.stream(limit=read_limit)):
            if quota is not None and produced < quota:
                all_docs.append(doc)
            elif res_quota and len(reserve_docs) < reserve_target:
                reserve_docs.append(doc)
            else:
                break
            produced += 1
            if produced >= want:
                break

        sid = entry["id"]
        stats_by_source[sid] = {
            "seen": cleaner.stats.seen - before_seen,
            "kept": cleaner.stats.kept - before_kept,
            "rejected": dict(cleaner.stats.rejected - before_rejected),
        }
        seen_here = stats_by_source[sid]["seen"]
        kept_here = stats_by_source[sid]["kept"]
        stats_by_source[sid]["keep_rate"] = round(kept_here / seen_here, 4) if seen_here else 0.0
        if quota and produced < want:
            shortfalls[sid] = {
                "wanted": want, "got": produced,
                "keep_rate": stats_by_source[sid]["keep_rate"],
                "hint": (
                    "the source stream was exhausted before the quota was met. Raise "
                    "--read-multiplier, or relax the length filter in the config: with "
                    "max_tokens set low, most web documents are rejected as too long."
                ),
            }

    # Hard invariants. If either raises, nothing downstream is trustworthy.
    check_no_group_leakage([(d.source_group_id, d.split) for d in all_docs])
    assert_corpus_matches_policy(all_docs + reserve_docs, policy)

    version = cfg.get("dataset_version", "v0.1")
    partitions = write_parquet(all_docs, out_root)

    reserve_partitions: dict[str, int] = {}
    if reserve_docs:
        reserve_partitions = write_parquet(
            reserve_docs, Path(reserve_root) if reserve_root else REPO_ROOT / "data" / "reserve"
        )
    # Publishable artifact lives outside the partition tree: different schema, and a
    # glob over the data lake must never pick it up.
    write_metadata_only(all_docs, REPO_ROOT / "data" / "publishable" / version / "metadata_only.parquet")
    manifest = write_manifest(out_root, version, all_docs, stats_by_source)
    return IngestResult(all_docs, stats_by_source, partitions, manifest,
                        reserve_docs=len(reserve_docs),
                        reserve_partitions=reserve_partitions, shortfalls=shortfalls)
