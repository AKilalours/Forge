"""Character-level ground truth for the token head. data_spec_v1 v1.2.

The problem this solves: mirrors give clean document-level labels, but the token head
has to say WHERE a document turns from human to AI, and nothing in the mirror corpus
carries that information.

Three constructions, and the third is the one that keeps the other two honest.

  splice          human paragraphs interleaved with paragraphs from that same document's
                  own mirror. Boundaries are exact and free. Weakness: the seam is an
                  abrupt discontinuity, and a token head can learn "something changed
                  here" instead of "the author changed here".

  edit_diff       a human document partially rewritten by a model; the character diff
                  gives the ai_assisted spans. This is what real AI-assisted writing
                  looks like, and there is no artificial seam. Costs model time.

  splice_control  paragraphs from two DIFFERENT human documents, every span labelled
                  human. Not optional. Without it a token head can score well on the
                  splice set purely by detecting discontinuity, and no metric in the
                  evaluation lab would reveal that. The control is the experiment that
                  distinguishes "detects authorship" from "detects seams".

All three inherit source_group_id and split from their human source, exactly like
mirrors. A control built from two human documents is only ever built from two documents
already in the same split, and inherits the first one's group.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Literal

from forge.common.schemas import Split, TokenLabel

Construction = Literal["splice", "edit_diff", "splice_control"]

_PARA = re.compile(r"\n\s*\n")
MIN_PARAGRAPHS = 4
MIN_PARAGRAPH_CHARS = 80
MIN_EDIT_FRACTION = 0.05
MAX_EDIT_FRACTION = 0.95


@dataclass
class Span:
    start_char: int
    end_char: int
    label: TokenLabel

    def as_dict(self) -> dict:
        return {"start_char": self.start_char, "end_char": self.end_char, "label": self.label.value}


@dataclass
class MixedDocument:
    doc_id: str
    source_group_id: str
    split: Split
    text: str
    spans: list[Span]
    construction: Construction
    human_source_id: str
    ai_source_id: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ai_char_fraction(self) -> float:
        n = len(self.text)
        if not n:
            return 0.0
        ai = sum(s.end_char - s.start_char for s in self.spans if s.label is not TokenLabel.HUMAN)
        return ai / n


class SpanError(ValueError):
    pass


def validate_spans(text: str, spans: list[Span]) -> None:
    """Spans must tile the document exactly: no gaps, no overlaps, no drift.

    A gap means some characters have no label and the loss silently ignores them. An
    overlap means a character has two labels and whichever comes last in iteration wins,
    which is a bug that produces plausible-looking training data.
    """
    if not spans:
        raise SpanError("document has no spans")
    ordered = sorted(spans, key=lambda s: s.start_char)
    if ordered[0].start_char != 0:
        raise SpanError(f"first span starts at {ordered[0].start_char}, expected 0")
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.start_char != a.end_char:
            raise SpanError(f"gap or overlap between {a.end_char} and {b.start_char}")
    if ordered[-1].end_char != len(text):
        raise SpanError(f"last span ends at {ordered[-1].end_char}, text is {len(text)} chars")


def merge_adjacent(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for s in sorted(spans, key=lambda x: x.start_char):
        if out and out[-1].label is s.label and out[-1].end_char == s.start_char:
            out[-1] = Span(out[-1].start_char, s.end_char, s.label)
        else:
            out.append(Span(s.start_char, s.end_char, s.label))
    return out


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA.split(text) if len(p.strip()) >= MIN_PARAGRAPH_CHARS]


def _rng_bits(seed_key: str) -> int:
    return int(hashlib.sha256(seed_key.encode()).hexdigest(), 16)


def _interleave_plan(n_human: int, n_ai: int, seed_key: str) -> list[bool]:
    """Deterministic True/False plan, True meaning 'take an AI paragraph'.

    Deterministic so a rebuild produces the identical dataset. Both sources must appear,
    otherwise a 'mixed' document is really a pure one and its label is a lie.
    """
    total = min(n_human + n_ai, n_human + n_ai)
    bits = _rng_bits(seed_key)
    plan = [bool((bits >> i) & 1) for i in range(total)]
    if not any(plan):
        plan[total // 2] = True
    if all(plan):
        plan[0] = False
    return plan


def build_splice(
    human_id: str,
    human_text: str,
    ai_id: str,
    ai_text: str,
    source_group_id: str,
    split: Split,
    ai_label: TokenLabel = TokenLabel.AI_GENERATED,
    construction: Construction = "splice",
) -> MixedDocument | None:
    """Interleave paragraphs. Returns None when either side is too short to splice."""
    hp, ap = paragraphs(human_text), paragraphs(ai_text)
    if len(hp) < 2 or len(ap) < 2 or len(hp) + len(ap) < MIN_PARAGRAPHS:
        return None

    plan = _interleave_plan(len(hp), len(ap), f"{human_id}|{ai_id}")
    parts: list[tuple[str, TokenLabel]] = []
    hi = ai = 0
    for take_ai in plan:
        # Fall through when the preferred source is exhausted. Without this, a slot
        # whose source has run dry is silently dropped and paragraphs go missing, which
        # shortens the document and skews the human/AI character ratio away from what
        # the plan intended.
        if take_ai and ai >= len(ap):
            take_ai = False
        elif not take_ai and hi >= len(hp):
            take_ai = True
        if take_ai and ai < len(ap):
            parts.append((ap[ai], ai_label))
            ai += 1
        elif not take_ai and hi < len(hp):
            parts.append((hp[hi], TokenLabel.HUMAN))
            hi += 1
    if len({lbl for _, lbl in parts}) < 2 and construction != "splice_control":
        return None

    text_parts, spans, cursor = [], [], 0
    sep = "\n\n"
    for i, (para, label) in enumerate(parts):
        chunk = para if i == len(parts) - 1 else para + sep
        text_parts.append(chunk)
        # The separator is attributed to the paragraph it follows, so spans tile exactly.
        spans.append(Span(cursor, cursor + len(chunk), label))
        cursor += len(chunk)

    text = "".join(text_parts)
    spans = merge_adjacent(spans)
    validate_spans(text, spans)
    return MixedDocument(
        doc_id=f"mix_{construction}_{human_id}",
        source_group_id=source_group_id,
        split=split,
        text=text,
        spans=spans,
        construction=construction,
        human_source_id=human_id,
        ai_source_id=ai_id,
        meta={"n_human_paragraphs": hi, "n_ai_paragraphs": ai},
    )


def build_splice_control(
    human_a_id: str,
    human_a_text: str,
    human_b_id: str,
    human_b_text: str,
    source_group_id: str,
    split: Split,
) -> MixedDocument | None:
    """Two different human documents, every span labelled human.

    This is the control that tells you whether the token head learned authorship or
    learned discontinuity. If it flags these, it learned discontinuity.
    """
    doc = build_splice(
        human_a_id, human_a_text, human_b_id, human_b_text,
        source_group_id, split,
        ai_label=TokenLabel.HUMAN, construction="splice_control",
    )
    if doc is None:
        return None
    doc.spans = merge_adjacent(doc.spans)
    validate_spans(doc.text, doc.spans)
    return doc


MIN_EQUAL_RUN_CHARS = 24  # shorter "unchanged" islands are absorbed into the edit

_TOKEN_SPLIT = re.compile(r"\S+\s*")


def _word_offsets(text: str) -> list[tuple[int, int, str]]:
    """(start, end, word-with-trailing-space) covering the text exactly."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for m in _TOKEN_SPLIT.finditer(text):
        if m.start() > pos:  # leading whitespace before the first token
            out.append((pos, m.start(), text[pos : m.start()]))
        out.append((m.start(), m.end(), m.group()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text), text[pos:]))
    return out


def spans_from_diff(
    original: str,
    edited: str,
    changed_label: TokenLabel = TokenLabel.AI_ASSISTED,
    min_equal_run_chars: int = MIN_EQUAL_RUN_CHARS,
) -> list[Span]:
    """Character spans of the EDITED text, marking what the model rewrote.

    Diffing is done at WORD granularity, not character granularity, and short unchanged
    runs are absorbed into the surrounding edit. Both rules exist because of a real
    failure found in testing.

    A character-level SequenceMatcher run over two completely unrelated paragraphs still
    reports matches: fragments like "n ", "a", "ing ", "en" appear in both texts by
    coincidence. Labelled naively, a total rewrite scores an ai fraction of 0.63 rather
    than ~1.0, and the resulting training document is AI text shot through with
    one-character islands labelled human. A token head trained on that is asked to flip
    its prediction every few characters on noise.

    Word granularity removes the sub-word coincidences. The min_equal_run_chars floor
    removes the remaining short ones: a single shared word such as "the" between two
    rewritten sentences is a coincidence, not preserved human authorship.
    """
    o_words = [w for _, _, w in _word_offsets(original)]
    e_offsets = _word_offsets(edited)
    e_words = [w for _, _, w in e_offsets]

    sm = SequenceMatcher(None, o_words, e_words, autojunk=False)
    spans: list[Span] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if j2 <= j1:
            continue
        start = e_offsets[j1][0]
        end = e_offsets[j2 - 1][1]
        spans.append(Span(start, end, TokenLabel.HUMAN if tag == "equal" else changed_label))

    spans = _absorb_short_equal_runs(spans, changed_label, min_equal_run_chars)
    spans = merge_adjacent(spans)
    validate_spans(edited, spans)
    return spans


def _absorb_short_equal_runs(spans: list[Span], changed_label: TokenLabel, floor: int) -> list[Span]:
    """Relabel short human runs that sit between two edits. A shared 'the' is chance."""
    out = [Span(s.start_char, s.end_char, s.label) for s in spans]
    for i, s in enumerate(out):
        if s.label is not TokenLabel.HUMAN:
            continue
        if (s.end_char - s.start_char) >= floor:
            continue
        prev_changed = i > 0 and out[i - 1].label is changed_label
        next_changed = i + 1 < len(out) and out[i + 1].label is changed_label
        if prev_changed and next_changed:
            out[i] = Span(s.start_char, s.end_char, changed_label)
    return out


def build_edit_diff(
    human_id: str,
    human_text: str,
    edited_text: str,
    source_group_id: str,
    split: Split,
    editor: str,
) -> MixedDocument | None:
    """A human document partially rewritten by a model. No artificial seam."""
    if edited_text.strip() == human_text.strip():
        return None
    spans = spans_from_diff(human_text, edited_text)
    doc = MixedDocument(
        doc_id=f"mix_edit_{human_id}",
        source_group_id=source_group_id,
        split=split,
        text=edited_text,
        spans=spans,
        construction="edit_diff",
        human_source_id=human_id,
        ai_source_id=editor,
        meta={"editor": editor},
    )
    # A rewrite that touched almost nothing, or almost everything, is not "assisted".
    #
    # Known limitation, recorded rather than hidden. ai_char_fraction measures how many
    # CHARACTERS changed, not how much of the document an AI influenced. A model that
    # fixes one word per sentence across an entire document touches every paragraph but
    # produces a diff of maybe 3 percent, and is rejected here. That is a real
    # AI-assistance case the product cares about and this construction cannot label it,
    # because at character level the untouched words genuinely are human.
    #
    # The 0.05 floor is kept anyway: below it the positive class is a handful of tokens
    # per document, and a token head trained on that learns noise. Light-touch editing
    # needs a different construction (sentence-level provenance from the editor itself
    # rather than a post-hoc diff) and is recorded as future work.
    frac = doc.ai_char_fraction
    if frac < MIN_EDIT_FRACTION or frac > MAX_EDIT_FRACTION:
        return None
    return doc


class Editor:
    """Phase 3 deployment: partially rewrite a human document.

    The prompt must ask for a rewrite of a SUBSET of sentences, leaving the rest byte
    identical. A model that paraphrases the whole document produces a diff of 100
    percent, which is a mirror, not an assisted document.
    """

    def __init__(self, model_id: str, revision: str, target_edit_fraction: float = 0.35) -> None:
        self.model_id, self.revision = model_id, revision
        self.target_edit_fraction = target_edit_fraction

    def edit(self, text: str) -> str:
        raise NotImplementedError("Phase 3 deployment: wire a local instruct model or the API family")


@dataclass
class BuildStats:
    built: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {"built": self.built, "skipped": dict(self.skipped)}


def why_splice_failed(human_text: str, ai_text: str) -> str | None:
    """Return the reason a splice cannot be built, or None if it can.

    Exists so a run that produces zero mixed documents says WHY. A generator that
    returns single-block text yields no splices at all, and without this the pipeline
    reports an empty result with no explanation.
    """
    hp, ap = paragraphs(human_text), paragraphs(ai_text)
    if len(hp) < 2:
        return "human_too_few_paragraphs"
    if len(ap) < 2:
        return "ai_too_few_paragraphs"
    if len(hp) + len(ap) < MIN_PARAGRAPHS:
        return "too_few_paragraphs_total"
    return None
