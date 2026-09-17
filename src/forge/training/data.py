"""Parquet to windowed, tokenized training examples.

This is the join between the data lake and the model. It reads human documents, their
mirrors, and any mixed documents produced by token_labels.py, and emits fixed-width
windows with both a document label and per-token labels.

The one rule enforced here that nothing else can enforce: **a window's document label is
computed from its own content**, via modeling/dataset.py. A mostly-human document with one
AI paragraph must not stamp "ai" on its mostly-human windows.

torch is imported lazily so the rest of the package still imports without it.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from forge.common.schemas import TokenLabel
from forge.modeling.alignment import IGNORE_INDEX, align_spans_to_tokens
from forge.modeling.windowing import windows


@dataclass
class RawExample:
    doc_id: str
    source_group_id: str
    split: str
    text: str
    label: int                                  # document level, 0 human 1 ai
    spans: list[tuple[int, int, str]] | None    # character spans, None = single-label doc
    domain: str = "unknown"
    generator_family: str = "human"
    generator_released: str = ""


def _single_span(text: str, label: int) -> list[tuple[int, int, str]]:
    tl = TokenLabel.AI_GENERATED.value if label else TokenLabel.HUMAN.value
    return [(0, len(text), tl)]


# WHICH PROMPT FAMILY EACH ARM IS ALLOWED TO TRAIN ON. Arm C trains on mirrors, so it
# shares the mirror arm's prompt family.
ARM_PROMPT_FAMILY = {"random": "random", "mirror": "mirror", "hard_negative": "mirror"}

# Kept for callers that still import it; derived, not a second source of truth.
ARM_PROMPT_VERSION = {arm: f"{fam}_v1" for arm, fam in ARM_PROMPT_FAMILY.items()}


class ArmMismatch(RuntimeError):
    pass


def check_arm_versions(expect_arm: str, seen_versions: set[str], ai_root) -> None:
    """Refuse to train an arm on another arm's data, or on a mixture of prompt versions.

    THE CHECK WAS PINNED TO EXACT VERSION STRINGS AND THE STRINGS MOVED. This was a map
    from arm to the literal "random_v1" / "mirror_v1". Correcting the generation prompts
    to ask for words rather than tokens made them v2, per data_spec_v1's rule that a
    wording change is a new version, and the first training run after that died with
    "config declares arm 'random' but data/silver/random contains ['random_v2']". The
    guard was right that something had changed and wrong about what: the data was correct
    and the guard was stale.

    Bumping the literals to v2 would have fixed tonight and rotted again at v3. What the
    guard is actually for is that two arms must not read the same directory, since that
    produces identical results and a false finding. So it now checks the prompt FAMILY,
    which is the arm's identity and does not change, and separately requires that every
    document in a directory share ONE version, which is the stronger half: a directory
    holding both v1 and v2 is a corpus half-regenerated, and averaging across a prompt
    change is its own silent confound.
    """
    family = ARM_PROMPT_FAMILY.get(expect_arm)
    if not family or not seen_versions:
        return
    wrong = {v for v in seen_versions if not re.fullmatch(rf"{family}_v\d+", v or "")}
    if wrong:
        raise ArmMismatch(
            f"config declares arm {expect_arm!r} (expects prompt_version "
            f"{family}_v<N>) but {ai_root} contains {sorted(seen_versions)}. The arm is "
            "training on the wrong data. Two arms reading the same directory produce "
            "identical results and a false finding."
        )
    if len(seen_versions) > 1:
        raise ArmMismatch(
            f"{ai_root} mixes prompt versions {sorted(seen_versions)}. A half-regenerated "
            "corpus trains across a prompt change, which is a confound nothing downstream "
            "can see. Regenerate the whole arm or delete the older part."
        )


def load_examples(
    human_root: str | Path | None = None,
    ai_root: str | Path | None = None,
    mixed_root: str | Path | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    limit: int | None = None,
    expect_arm: str | None = None,
    ai_cap: int | None = None,
    ai_reference: list[RawExample] | None = None,
    human_cap: int | None = None,
    mirror_root: str | Path | None = None,   # deprecated alias
) -> list[RawExample]:
    """Load one arm's training data.

    `expect_arm` exists because of a bug that would have produced a FALSE NEGATIVE at
    real cost. All three arm configs pointed their AI source at the same directory, and
    `data.include` was decorative, so Arm A and Arm B would have trained on identical
    data, produced identical numbers, and supported the conclusion "matched mirrors make
    no difference".

    Nothing would have errored. Two successful runs, two plausible rows in the results
    table, one false finding. So the arm is now declared in the config and verified
    against the `prompt_version` recorded on every generated document.

    `ai_cap` holds the AI budget equal across arms. Generation produces a different
    NUMBER of accepted documents per arm, because each arm's validator rejects at its own
    rate: the mirror arm accepts a length ratio in [0.6, 1.6], the random arm [0.5, 2.0],
    and the mirror arm's first-pass rejection ran near 40%. Without a cap, one arm trains
    on thousands more documents than the other and the comparison stops being about
    matching and starts being about volume, which is the one confound the experiment was
    designed to exclude. Set it to the smaller arm's count for both arms.
    """
    ai_root = ai_root or mirror_root
    human_rows: list[RawExample] = []
    ai_rows: list[RawExample] = []
    mixed_rows: list[RawExample] = []

    if human_root:
        from forge.ingestion.writer import PARTITION_GLOB

        for f in sorted(glob.glob(str(Path(human_root) / PARTITION_GLOB))):
            for r in pq.read_table(
                f, columns=["doc_id", "source_group_id", "text", "split", "domain"]
            ).to_pylist():
                if r["split"] not in splits:
                    continue
                human_rows.append(RawExample(r["doc_id"], r["source_group_id"], r["split"],
                                             r["text"], 0, _single_span(r["text"], 0), r["domain"]))

    if ai_root:
        seen_versions: set[str] = set()
        for f in sorted(glob.glob(str(Path(ai_root) / "split=*/*.parquet"))):
            for r in pq.read_table(
                f, columns=["sample_id", "source_group_id", "text", "split", "domain",
                            "generator", "mirror"]
            ).to_pylist():
                if r["split"] not in splits:
                    continue
                g = r.get("generator") or {}
                seen_versions.add(((r.get("mirror") or {}).get("prompt_version")) or "unknown")
                ai_rows.append(RawExample(r["sample_id"], r["source_group_id"], r["split"],
                                          r["text"], 1, _single_span(r["text"], 1), r["domain"],
                                          g.get("family", "unknown"), str(g.get("released", ""))))

        if expect_arm:
            check_arm_versions(expect_arm, seen_versions, ai_root)

    if mixed_root:
        for f in sorted(glob.glob(str(Path(mixed_root) / "split=*/*.parquet"))):
            for r in pq.read_table(f).to_pylist():
                if r["split"] not in splits:
                    continue
                spans = [(s["start_char"], s["end_char"], s["label"]) for s in r["spans"]]
                ai = sum(e - s for s, e, lbl in spans if lbl != TokenLabel.HUMAN.value)
                mixed_rows.append(RawExample(r["doc_id"], r["source_group_id"], r["split"], r["text"],
                                             int(ai / max(len(r["text"]), 1) >= 0.5), spans,
                                             r.get("domain", "unknown")))

    # human_cap exists to control TRAINING TIME, not the experiment. Both arms read the
    # same human corpus, so capping it identically in both cannot favour either one. It is
    # a separate knob from ai_cap because the two answer different questions: ai_cap makes
    # the arms comparable, human_cap makes the run finish tonight.
    #
    # NOT the `limit` parameter, which round-robins across (source, split) buckets and
    # therefore returns EQUAL numbers of train, val and test rows. That is correct for a
    # smoke run, where the point is to touch every split, and wrong for a real run, where
    # it would spend a third of the budget on the test split. cap_documents caps per split
    # and preserves the proportions the corpus actually has.
    if human_cap is not None and len(human_rows) > human_cap:
        human_rows = cap_documents(human_rows, human_cap)

    if ai_cap is not None and len(ai_rows) > ai_cap:
        if ai_reference is not None:
            # Match the reference arm's LENGTH distribution, not just its count. See
            # cap_documents_matching for why an equal count alone leaves a confound.
            ai_rows = cap_documents_matching(ai_rows, ai_reference, ai_cap)
        else:
            ai_rows = cap_documents(ai_rows, ai_cap)

    if limit is None:
        return human_rows + ai_rows + mixed_rows
    return interleave(*_by_source_and_split(human_rows, ai_rows, mixed_rows), limit=limit)


# Word-count bins for length matching. Fixed rather than derived from the data, so two
# arms capped on different days land in the same bins and a cap is reproducible.
LENGTH_BINS = (0, 150, 250, 350, 500, 750, 1_100)


def length_bin(text: str) -> int:
    words = len(text.split())
    for index in range(len(LENGTH_BINS) - 1, -1, -1):
        if words >= LENGTH_BINS[index]:
            return index
    return 0


def _rank(row: RawExample) -> str:
    import hashlib

    return hashlib.sha256(f"cap:{row.doc_id}".encode()).hexdigest()


def cap_documents_matching(
    rows: list[RawExample], reference: list[RawExample], cap: int
) -> list[RawExample]:
    """Cap to `cap` documents whose LENGTH distribution tracks `reference`.

    WHY THIS EXISTS, and it is not a nicety.

    The mirror arm's validator rejects a generation whose length ratio to its source falls
    outside [0.6, 1.6]. Generation used one max_new_tokens for every document, so a short
    source could never be matched: the model wrote to the cap and overshot. Length
    overshoot was 85% of all rejections, and the documents that survived skew long.

    The control arm draws its target lengths from the whole human corpus, so it does not
    skew. Capping both arms to the same COUNT therefore leaves them differing in length
    distribution as well as in matching, and a detector can learn length. A win for either
    arm would then have an obvious alternative explanation.

    Matching the distribution at cap time removes that confound without regenerating
    anything. Selection stays deterministic: within each (split, length bin) cell, documents
    are ranked by a salted hash of their id.

    Where a cell in `rows` cannot supply its full share, the shortfall is redistributed
    across the remaining cells rather than silently shrinking the result, so the returned
    count is exact and the arms stay equal in size as well as in shape.
    """
    if cap >= len(rows):
        return list(rows)

    def cell(row: RawExample) -> tuple[str, int]:
        return (row.split, length_bin(row.text))

    reference_counts: dict[tuple[str, int], int] = {}
    for row in reference:
        reference_counts[cell(row)] = reference_counts.get(cell(row), 0) + 1
    total_reference = sum(reference_counts.values()) or 1

    available: dict[tuple[str, int], list[RawExample]] = {}
    for row in rows:
        available.setdefault(cell(row), []).append(row)
    for key in available:
        available[key].sort(key=_rank)

    # First pass: each cell's proportional share, bounded by what it can supply.
    taken: dict[tuple[str, int], int] = {}
    for key, count in reference_counts.items():
        want = int(cap * count / total_reference)
        taken[key] = min(want, len(available.get(key, [])))

    # Second pass: hand the shortfall to whichever cells still have documents, largest
    # first, so the total is exact. Without this the arms would differ in size, which is
    # the very thing the cap exists to prevent.
    remaining = cap - sum(taken.values())
    while remaining > 0:
        candidates = [
            key for key in available if len(available[key]) > taken.get(key, 0)
        ]
        if not candidates:
            break
        candidates.sort(key=lambda k: len(available[k]) - taken.get(k, 0), reverse=True)
        for key in candidates:
            if remaining == 0:
                break
            taken[key] = taken.get(key, 0) + 1
            remaining -= 1

    keep = {
        row.doc_id
        for key, count in taken.items()
        for row in available.get(key, [])[:count]
    }
    return [row for row in rows if row.doc_id in keep]


def cap_documents(rows: list[RawExample], cap: int) -> list[RawExample]:
    """Keep `cap` documents, chosen deterministically and evenly across splits.

    Ranking by a salted hash of the document id is stable across machines and runs, and
    independent of read order, so two arms capped to the same number are capped the same
    way rather than by whatever the filesystem returned first. Capping per split keeps the
    train/val/test proportions the generation produced; a global cap would let one split
    absorb the whole reduction.
    """
    import hashlib

    def rank(row: RawExample) -> str:
        return hashlib.sha256(f"cap:{row.doc_id}".encode()).hexdigest()

    by_split: dict[str, list[RawExample]] = {}
    for row in rows:
        by_split.setdefault(row.split, []).append(row)

    keep: set[str] = set()
    remaining = cap
    for i, (_split, split_rows) in enumerate(sorted(by_split.items())):
        share = round(cap * len(split_rows) / len(rows))
        # Last split takes the rounding remainder so the total is exact.
        if i == len(by_split) - 1:
            share = remaining
        share = max(0, min(share, len(split_rows), remaining))
        keep |= {r.doc_id for r in sorted(split_rows, key=rank)[:share]}
        remaining -= share
    return [r for r in rows if r.doc_id in keep]


def _by_source_and_split(*sources: list[RawExample]) -> list[list[RawExample]]:
    """One bucket per (source, split) pair, so a limit spans both.

    Round-robining over sources alone is not enough. The parquet files are partitioned by
    split and read in sorted order, so the front of each source's list is a single split:
    a 200-row limit drew 100 humans and 100 AI documents that were all `test`, leaving no
    training windows at all. Splitting the buckets further means a limit always contains
    train, val and test.
    """
    buckets: list[list[RawExample]] = []
    for rows in sources:
        grouped: dict[str, list[RawExample]] = {}
        for row in rows:
            grouped.setdefault(row.split, []).append(row)
        buckets.extend(grouped[key] for key in sorted(grouped))
    return buckets


def interleave(*buckets: list[RawExample], limit: int | None = None) -> list[RawExample]:
    """Concatenate the sources, or take a limit that draws from ALL of them.

    WHY. Each loader used to append to one list and return early once it had `limit`
    rows. Humans are read first, so any limited load returned humans only: `--smoke`,
    which is --limit 200, trained on 200 human documents and zero AI documents. The run
    completed, reported a loss, and told you nothing, because a classifier fed one class
    cannot fail in a way a smoke test would notice.

    That is the same defect as the corpus loader's prefix truncation, in a different
    file. The lesson worth keeping: an early return inside a source-specific loop is
    never a subsample, it is a filter on source.

    Round-robin across the buckets so a limit is a real sample of what a full run would
    see. Callers pass one bucket per (source, split) pair; see _by_source_and_split.
    """
    if limit is None:
        return [row for bucket in buckets for row in bucket]
    picked: list[RawExample] = []
    index = 0
    while len(picked) < limit and any(index < len(b) for b in buckets):
        for bucket in buckets:
            if index < len(bucket):
                picked.append(bucket[index])
                if len(picked) >= limit:
                    break
        index += 1
    return picked


def build_dataset(examples: list[RawExample], tokenizer, max_length: int = 512,
                  stride: int = 384, ai_window_threshold: float = 0.5):
    """Tokenize once per document, then slice into overlapping windows.

    Tokenizing per window instead would retokenize overlapping text and, worse, would put
    window boundaries at token positions that differ from the offsets used for span
    alignment.
    """

    feats = []
    for ex in examples:
        enc = tokenizer(ex.text, return_offsets_mapping=True, add_special_tokens=False,
                        truncation=False)
        offsets = [tuple(o) for o in enc["offset_mapping"]]
        ids = enc["input_ids"]
        if not ids:
            continue
        spans = [(s, e, TokenLabel(lbl)) for s, e, lbl in (ex.spans or _single_span(ex.text, ex.label))]
        token_labels = align_spans_to_tokens(offsets, spans)

        # reserve two positions for the special tokens the collator adds
        body = max_length - 2
        for wi, (a, b) in enumerate(windows(len(ids), size=body, stride=min(stride, body))):
            wl = token_labels[a:b]
            real = [x for x in wl if x != IGNORE_INDEX]
            frac = (sum(1 for x in real if x != 0) / len(real)) if real else 0.0
            feats.append({
                "input_ids": ids[a:b],
                "token_labels": wl,
                "doc_label": int(frac >= ai_window_threshold),
                "doc_id": ex.doc_id,
                "source_group_id": ex.source_group_id,
                "split": ex.split,
                "domain": ex.domain,
                "generator_family": ex.generator_family,
                "window_index": wi,
                "ai_char_fraction": frac,
            })
    return feats


class Collator:
    """Pads a batch and adds special tokens, keeping token labels aligned.

    Special tokens get IGNORE_INDEX so the token head is never trained to predict
    authorship for a [CLS] that carries no text.
    """

    def __init__(self, tokenizer, max_length: int = 512) -> None:
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        import torch

        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        if pad is None:
            pad = 0
        ids, labs = [], []
        for b in batch:
            seq = ([cls] if cls is not None else []) + list(b["input_ids"]) + ([sep] if sep is not None else [])
            lab = ([IGNORE_INDEX] if cls is not None else []) + list(b["token_labels"]) + ([IGNORE_INDEX] if sep is not None else [])
            ids.append(seq[: self.max_length])
            labs.append(lab[: self.max_length])
        width = max(len(x) for x in ids)
        attn = [[1] * len(x) + [0] * (width - len(x)) for x in ids]
        ids = [x + [pad] * (width - len(x)) for x in ids]
        labs = [x + [IGNORE_INDEX] * (width - len(x)) for x in labs]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "token_labels": torch.tensor(labs, dtype=torch.long),
            "doc_labels": torch.tensor([b["doc_label"] for b in batch], dtype=torch.long),
        }
