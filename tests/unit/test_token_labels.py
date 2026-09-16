"""Token-label construction. data_spec_v1 v1.2."""

import pytest

from forge.common.schemas import Split, TokenLabel
from forge.generation.token_labels import (
    Span,
    SpanError,
    build_edit_diff,
    build_splice,
    build_splice_control,
    spans_from_diff,
    validate_spans,
)

H = "\n\n".join(
    f"Human paragraph number {i} carrying enough characters to clear the minimum length filter." for i in range(5)
)
A = "\n\n".join(
    f"Machine paragraph number {i} carrying enough characters to clear the minimum length filter." for i in range(5)
)


def test_spans_tile_the_document_exactly():
    d = build_splice("h1", H, "a1", A, "grp_h1", Split.TRAIN)
    validate_spans(d.text, d.spans)  # must not raise


def test_a_gap_between_spans_is_rejected():
    """A gap means some characters carry no label and the loss silently ignores them."""
    with pytest.raises(SpanError):
        validate_spans("abcdef", [Span(0, 2, TokenLabel.HUMAN), Span(3, 6, TokenLabel.AI_GENERATED)])


def test_spans_must_reach_the_end_of_the_text():
    with pytest.raises(SpanError):
        validate_spans("abcdef", [Span(0, 4, TokenLabel.HUMAN)])


def test_no_paragraphs_are_dropped_when_one_source_runs_out():
    """Regression guard. The interleave plan can ask for a source that is exhausted; if
    the slot is dropped instead of falling through, paragraphs vanish and the human/AI
    character ratio drifts away from what the plan intended."""
    d = build_splice("h1", H, "a1", A, "grp_h1", Split.TRAIN)
    assert d.meta["n_human_paragraphs"] == 5
    assert d.meta["n_ai_paragraphs"] == 5


def test_splice_contains_both_authors():
    d = build_splice("h1", H, "a1", A, "grp_h1", Split.TRAIN)
    labels = {s.label for s in d.spans}
    assert TokenLabel.HUMAN in labels and TokenLabel.AI_GENERATED in labels
    assert 0.0 < d.ai_char_fraction < 1.0


def test_splice_inherits_group_and_split():
    d = build_splice("h1", H, "a1", A, "grp_h1", Split.TEST)
    assert d.source_group_id == "grp_h1" and d.split is Split.TEST


def test_construction_is_deterministic():
    a = build_splice("h1", H, "a1", A, "grp_h1", Split.TRAIN)
    b = build_splice("h1", H, "a1", A, "grp_h1", Split.TRAIN)
    assert a.text == b.text and [s.as_dict() for s in a.spans] == [s.as_dict() for s in b.spans]


def test_control_is_entirely_human():
    """The control is what separates 'detects authorship' from 'detects discontinuity'.
    If it carried any AI label it would not be a control."""
    c = build_splice_control("h1", H, "h2", A, "grp_h1", Split.TRAIN)
    assert {s.label for s in c.spans} == {TokenLabel.HUMAN}
    assert c.ai_char_fraction == 0.0
    assert c.construction == "splice_control"


def test_control_still_has_a_discontinuity():
    """It must look structurally like a splice, otherwise it controls for nothing."""
    c = build_splice_control("h1", H, "h2", A, "grp_h1", Split.TRAIN)
    assert "Machine paragraph" in c.text and "Human paragraph" in c.text


def test_too_short_input_produces_nothing_rather_than_a_bad_label():
    assert build_splice("h", "one short line", "a", A, "grp_h", Split.TRAIN) is None


# ------------------------------------------------------------------ edit diff

def test_diff_spans_mark_only_what_changed():
    spans = spans_from_diff("the cat sat on the mat", "the cat perched on the mat")
    labels = [s.label for s in spans]
    assert TokenLabel.AI_ASSISTED in labels and TokenLabel.HUMAN in labels
    changed = "".join("the cat perched on the mat"[s.start_char:s.end_char]
                      for s in spans if s.label is TokenLabel.AI_ASSISTED)
    assert "perched" in changed and "the cat" not in changed


def test_diff_spans_tile_the_edited_text():
    edited = "the cat perched on the mat"
    validate_spans(edited, spans_from_diff("the cat sat on the mat", edited))


def test_a_full_rewrite_is_not_an_assisted_document():
    """A diff of ~100 percent is a mirror wearing the wrong label.

    Note the fixture: H and A differ by a single word per paragraph, so A is NOT a full
    rewrite of H, it is a 5 percent diff. Using it here would test nothing. The rewrite
    has to actually share no text.
    """
    rewritten = "\n\n".join(
        f"An entirely different formulation, numbered {i}, expressing unrelated content at length."
        for i in range(5)
    )
    assert build_edit_diff("h1", H, rewritten, "grp_h1", Split.TRAIN, editor="m") is None


def test_light_touch_editing_on_a_long_document_is_a_documented_gap():
    """Recorded limitation, not a bug.

    ai_char_fraction measures how many CHARACTERS changed, not how much of a document an
    AI influenced. In a long document a few small edits fall below the 0.05 floor and the
    document is rejected, even though it is genuinely AI-assisted. The floor is kept
    because below it the positive class is a handful of tokens per document and the token
    head would learn noise. Labelling light-touch editing needs provenance from the
    editor itself rather than a post-hoc diff, and is recorded as future work.

    Note the dependence on length: the SAME edit density in a short document lands above
    the floor and is accepted, which is why this test uses a long one.
    """
    long_doc = "\n\n".join([H] * 12)
    lightly = long_doc.replace("Human paragraph number 2 carrying", "Human paragraph number 2 bearing", 1)
    assert build_edit_diff("h1", long_doc, lightly, "grp_h1", Split.TRAIN, editor="m") is None

    # same kind of edit, short document: above the floor, accepted
    short_edit = H.replace("carrying", "bearing")
    assert build_edit_diff("h2", H, short_edit, "grp_h2", Split.TRAIN, editor="m") is not None


def test_an_untouched_document_is_not_an_assisted_document():
    assert build_edit_diff("h1", H, H, "grp_h1", Split.TRAIN, editor="m") is None


def test_a_partial_rewrite_is_accepted():
    edited = H.replace(
        "Human paragraph number 2 carrying enough characters to clear the minimum length filter.",
        "The second section was rewritten entirely by a language model in different words here.",
    )
    d = build_edit_diff("h1", H, edited, "grp_h1", Split.VAL, editor="m")
    assert d is not None
    assert 0.05 < d.ai_char_fraction < 0.95
    assert d.split is Split.VAL and d.source_group_id == "grp_h1"


def test_word_granularity_prevents_spurious_human_islands():
    """Regression guard for a real bug.

    A character-level diff between two unrelated texts still matches fragments like
    "n ", "a", "ing " by coincidence. Labelled naively, a total rewrite scored an ai
    fraction of 0.63 instead of ~1.0, and produced AI documents shot through with
    one-character spans labelled human, which asks the token head to flip its prediction
    every few characters on noise.
    """
    rewritten = "\n\n".join(
        f"An entirely different formulation, numbered {i}, expressing unrelated content at length."
        for i in range(5)
    )
    spans = spans_from_diff(H, rewritten)
    ai = sum(s.end_char - s.start_char for s in spans if s.label is not TokenLabel.HUMAN)
    assert ai / len(rewritten) > 0.95
    assert all(
        (s.end_char - s.start_char) >= 8 for s in spans
    ), "no sub-word islands should survive"


def test_one_rewritten_paragraph_gives_a_proportionate_fraction():
    edited = H.replace(
        "Human paragraph number 2 carrying enough characters to clear the minimum length filter.",
        "The second section was rewritten entirely by a language model using different words.",
    )
    spans = spans_from_diff(H, edited)
    ai = sum(s.end_char - s.start_char for s in spans if s.label is not TokenLabel.HUMAN)
    assert 0.1 < ai / len(edited) < 0.35
