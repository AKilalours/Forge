"""External benchmark loaders. EVAL-ONLY, always.

Schemas below were read from the live dataset cards on 2026-08-31 and are recorded in
data_spec_v1 section 1.1. They are pinned here as constants so a silent upstream schema
change fails loudly at load time instead of producing a plausible wrong number.

Nothing in this file has been run against the real datasets. Neither the development
sandbox nor the cloud container can reach huggingface.co, so every loader is verified
against synthetic fixtures matching the recorded schema exactly. The parsing logic is
tested; the download is not.

Three per-dataset traps, each of which produces a believable but wrong result:

  MAGE polarity. MEASURED, not assumed: label 0 is MACHINE, label 1 is HUMAN. This file
  originally guessed the opposite. scripts/mage_polarity_probe.py resolved it from the
  data, tabulating label against the `src` field over 6,000 rows: every label-0 source
  names a generator (eli5_machine_continuation_flan_t5_base, ..._text-davinci-003,
  ..._gpt-3.5-turbo) and every label-1 source names a human corpus (hswag_human,
  xsum_human, yelp_human). 100% against 0%, no overlap.
  An inverted label turns an AUROC of 0.05 into a reported 0.95 and nothing crashes, which
  is why the polarity is asserted at load rather than trusted.

  HC3 shape. `human_answers` and `chatgpt_answers` are LISTS. One row expands to several
  documents, so row counts are not document counts, and every document from one question
  must share a group id or the same question appears on both sides of any split.

  RAID adversarial grouping. Twelve attacks are applied to the same base generations, and
  they share `source_id`. Scoring them as independent documents counts one base text up
  to thirteen times and lets a single easy or hard base text dominate the number.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

# Pinned schemas. A mismatch is a hard error, not a warning.
RAID_COLUMNS = {
    "id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty",
    "attack", "domain", "title", "prompt", "generation",
}
MAGE_COLUMNS = {"text", "label", "src"}
# MEASURED from the src field over 6,000 rows, not taken from the dataset card.
# See the module docstring and scripts/mage_polarity_probe.py.
MAGE_MACHINE_LABEL = 0
HC3_COLUMNS = {"id", "question", "human_answers", "chatgpt_answers", "source"}

# English natural language only. FORGE v1 is not a code or multilingual detector, and
# scoring it on those inflates or deflates the number for reasons unrelated to the
# research question. Reported separately, never in the headline.
RAID_EXCLUDED_DOMAINS = frozenset({"code", "czech", "german"})

# RAID's human rows carry model == "human".
RAID_HUMAN_MODEL = "human"


class SchemaMismatch(RuntimeError):
    pass


class LabelPolarityError(AssertionError):
    pass


@dataclass
class BenchmarkDoc:
    """One evaluation document, normalised across the three benchmarks."""

    doc_id: str
    group_id: str          # shared by variants of the same base text
    text: str
    label: int             # 1 = machine-generated, FORGE's convention throughout
    benchmark: str
    domain: str = "unknown"
    model: str = "unknown"
    attack: str = "none"
    meta: dict = field(default_factory=dict)


def _require_columns(present: Iterable[str], expected: set[str], name: str) -> None:
    present = set(present)
    missing = expected - present
    if missing:
        raise SchemaMismatch(
            f"{name}: expected columns {sorted(missing)} are absent. The upstream schema "
            f"changed since data_spec_v1 section 1.1 was written (saw {sorted(present)}). "
            "Update the spec and this loader deliberately rather than coercing."
        )


# --------------------------------------------------------------------------- RAID

def parse_raid(
    rows: Iterable[dict],
    excluded_domains: frozenset[str] = RAID_EXCLUDED_DOMAINS,
    include_attacks: bool = True,
) -> list[BenchmarkDoc]:
    rows = list(rows)
    if rows:
        _require_columns(rows[0].keys(), RAID_COLUMNS, "RAID")
    out: list[BenchmarkDoc] = []
    for r in rows:
        domain = str(r.get("domain", "")).lower()
        if domain in excluded_domains:
            continue
        attack = str(r.get("attack") or "none")
        if not include_attacks and attack != "none":
            continue
        model = str(r.get("model", ""))
        out.append(
            BenchmarkDoc(
                doc_id=str(r["id"]),
                # Attacked variants share the base generation's source_id. Grouping by it
                # stops one base text being counted up to thirteen times.
                group_id=str(r.get("source_id") or r["id"]),
                text=str(r.get("generation", "")),
                label=0 if model == RAID_HUMAN_MODEL else 1,
                benchmark="raid",
                domain=domain or "unknown",
                model=model or "unknown",
                attack=attack,
                meta={
                    "decoding": r.get("decoding"),
                    "repetition_penalty": r.get("repetition_penalty"),
                    "adv_source_id": r.get("adv_source_id"),
                },
            )
        )
    return out


# --------------------------------------------------------------------------- MAGE

def parse_mage(rows: Iterable[dict], machine_label: int = MAGE_MACHINE_LABEL) -> list[BenchmarkDoc]:
    """MAGE rows to documents. `machine_label` defaults to the MEASURED convention.

    It was 1, which is backwards, and the run that caught it was the first time any score
    was computed on this benchmark. Nothing before that point could have noticed: the
    parser was self-consistent, the schema check passed, and the counts looked healthy.
    Only comparing scores against labels exposed it.
    """
    rows = list(rows)
    if rows:
        _require_columns(rows[0].keys(), MAGE_COLUMNS, "MAGE")
    return [
        BenchmarkDoc(
            doc_id=f"mage_{i}",
            group_id=f"mage_{i}",
            text=str(r["text"]),
            label=1 if int(r["label"]) == machine_label else 0,
            benchmark="mage",
            model=str(r.get("src", "unknown")),
            meta={"src": r.get("src")},
        )
        for i, r in enumerate(rows)
    ]


def assert_mage_polarity(docs: list[BenchmarkDoc], scores: list[float], margin: float = 0.05) -> None:
    """Fail loudly if the benchmark's labels are inverted relative to FORGE's convention.

    Uses the detector's own scores: documents labelled machine must score higher on
    average than documents labelled human. If they do not, either the labels are inverted
    or the detector is worse than random, and both must stop the run rather than produce
    a number.
    """
    y = np.asarray([d.label for d in docs], int)
    s = np.asarray(scores, float)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        raise LabelPolarityError("cannot check polarity without both classes present")
    m, h = float(s[y == 1].mean()), float(s[y == 0].mean())
    if m <= h + margin:
        raise LabelPolarityError(
            f"MAGE polarity check failed: mean score on machine-labelled documents "
            f"({m:.4f}) is not above human-labelled ({h:.4f}) by {margin}. Either the "
            "label convention is inverted or the detector is near-random. Both invalidate "
            "the reported number."
        )


# --------------------------------------------------------------------------- HC3

def parse_hc3(rows: Iterable[dict]) -> list[BenchmarkDoc]:
    """One row expands to several documents. Every document from one question shares a
    group id, so a question never lands on both sides of any split or resample."""
    rows = list(rows)
    if rows:
        _require_columns(rows[0].keys(), HC3_COLUMNS, "HC3")
    out: list[BenchmarkDoc] = []
    for r in rows:
        qid = str(r.get("id", len(out)))
        group = f"hc3_q{qid}"
        for kind, answers, label in (
            ("human", r.get("human_answers") or [], 0),
            ("chatgpt", r.get("chatgpt_answers") or [], 1),
        ):
            for j, ans in enumerate(answers):
                text = str(ans).strip()
                if not text:
                    continue
                out.append(
                    BenchmarkDoc(
                        doc_id=f"{group}_{kind}_{j}",
                        group_id=group,
                        text=text,
                        label=label,
                        benchmark="hc3",
                        domain=str(r.get("source", "unknown")),
                        model="chatgpt" if label else "human",
                        meta={"question": r.get("question")},
                    )
                )
    return out


# --------------------------------------------------------------------------- loading

# datasets v4 removed support for loading-script datasets. HC3 is still published as a
# script (HC3.py), so the ordinary path now raises:
#
#     RuntimeError: Dataset scripts are no longer supported, but found HC3.py
#
# The Hub keeps an auto-generated parquet conversion of every dataset on the branch
# refs/convert/parquet, which needs no script. The fallback below reads that instead.
#
# Pinning datasets<4 would be the other fix and is worse: it freezes a core dependency
# across the whole project to work around one benchmark's packaging, and the pin would
# outlive the reason for it.
_SCRIPTS_GONE = "Dataset scripts are no longer supported"
PARQUET_BRANCH = "refs/convert/parquet"


def _parquet_conversion(repo: str, config: str | None, split: str):  # pragma: no cover
    """Read a dataset's auto-generated parquet conversion.

    Files on that branch are laid out <config>/<split>/*.parquet. The revision is passed in
    the path because it contains slashes, which have to survive as part of the ref rather
    than being read as directories.
    """
    from datasets import load_dataset
    from huggingface_hub import list_repo_files

    files = [
        f
        for f in list_repo_files(repo, repo_type="dataset", revision=PARQUET_BRANCH)
        if f.endswith(".parquet")
    ]
    if config:
        scoped = [f for f in files if f.split("/")[0] == config]
        files = scoped or files
    scoped = [f for f in files if f"/{split}/" in f or f"-{split}-" in f]
    files = scoped or files
    if not files:
        raise RuntimeError(
            f"no parquet conversion found for {repo} (config={config}, split={split})"
        )
    revision = PARQUET_BRANCH.replace("/", "%2F")
    urls = [f"hf://datasets/{repo}@{revision}/{f}" for f in sorted(files)]
    return load_dataset("parquet", data_files=urls, split="train", streaming=True)


def load_hf(repo: str, config: str | None, split: str) -> Iterator[dict]:  # pragma: no cover
    """Streaming load, with a fallback for datasets still published as scripts."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("needs the `data` extra: pip install -e '.[data]'") from e
    try:
        yield from load_dataset(repo, name=config, split=split, streaming=True)
    except RuntimeError as error:
        if _SCRIPTS_GONE not in str(error):
            raise
        print(
            f"[benchmarks] {repo} is a script dataset; reading the parquet conversion "
            f"on {PARQUET_BRANCH}",
            flush=True,
        )
        yield from _parquet_conversion(repo, config, split)


def load_raid(split: str = "train", max_rows: int = 200_000) -> list[BenchmarkDoc]:  # pragma: no cover
    """RAID, English prose domains.

    THE BUG THIS FIXES. The default was "extra", chosen because it is labelled and disjoint
    from train. RAID-extra is the EXTRA-DOMAINS split: code, czech and german. FORGE
    excludes exactly those three as out of scope, so parse_raid correctly discarded 100% of
    it and the run failed with "raid returned no documents". The loader was asking for the
    one split whose entire contents this project's own policy rejects.

    Confirmed by reading the data (scripts/raid_probe.py):

        split 'extra' first row: model='human' domain='code'
        split 'train' first row: model='human' domain='abstracts'
        split 'test'           : ValueError, available splits are ['train', 'extra']

    RAID-train is labelled, so self-reporting on it is possible. The held-out labels are in
    the separate `raid_test` config, which needs a leaderboard submission.

    `max_rows` stops the stream early. parse_raid materialises what it is given, and RAID's
    train split is millions of rows; without a cap, streaming is defeated and a 4,000
    document sample costs a full download. The cap is applied BEFORE domain filtering, so
    it bounds work rather than changing which documents are eligible.
    """
    if split == "test":
        raise ValueError(
            "RAID has no test split in the `raid` config; the held-out labels live in the "
            "`raid_test` config and need a leaderboard submission. "
            "Available: train, extra. See data_spec_v1 section 1.1."
        )
    return parse_raid(_take(load_hf("liamdugan/raid", "raid", split), max_rows))


def _take(rows: Iterator[dict], limit: int) -> Iterator[dict]:  # pragma: no cover
    """Stop a stream after `limit` rows.

    Without this, `rows = list(rows)` inside every parser reads the entire split before a
    single document is sampled, which turns a streaming loader into a full download.
    """
    for i, row in enumerate(rows):
        if i >= limit:
            return
        yield row


def load_mage(split: str = "test") -> list[BenchmarkDoc]:  # pragma: no cover
    return parse_mage(load_hf("yaful/MAGE", None, split))


def load_hc3() -> list[BenchmarkDoc]:  # pragma: no cover
    return parse_hc3(load_hf("Hello-SimpleAI/HC3", "all", "train"))


def group_aware_scores(docs: list[BenchmarkDoc], scores: list[float]) -> tuple[list[int], list[float]]:
    """Collapse variants of the same base text to one point, averaging their scores.

    Without this, RAID's twelve attacks make one base generation count thirteen times, so
    a single unusually easy or hard base text moves the headline number.
    """
    agg: dict[str, list[float]] = {}
    lab: dict[str, int] = {}
    for d, s in zip(docs, scores, strict=True):
        agg.setdefault(d.group_id, []).append(float(s))
        lab[d.group_id] = d.label
    keys = sorted(agg)
    return [lab[k] for k in keys], [float(np.mean(agg[k])) for k in keys]
