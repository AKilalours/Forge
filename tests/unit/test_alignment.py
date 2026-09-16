"""Char-span to token-label alignment: the highest-risk pure function in Phase 3."""

from forge.common.schemas import Split
from forge.common.schemas import TokenLabel as T
from forge.modeling.alignment import (
    IGNORE_INDEX,
    LABEL_TO_ID,
    align_spans_to_tokens,
    document_label_from_tokens,
    spans_from_token_labels,
)
from forge.modeling.dataset import windows_from_spans

HUMAN, ASSIST, GEN = LABEL_TO_ID[T.HUMAN], LABEL_TO_ID[T.AI_ASSISTED], LABEL_TO_ID[T.AI_GENERATED]


def test_special_tokens_are_ignored_not_labelled():
    """[CLS] and [SEP] have offset (0,0) and appear in every window. Labelling them
    trains the model to predict authorship for tokens carrying no text."""
    offsets = [(0, 0), (0, 4), (4, 9), (0, 0)]
    ids = align_spans_to_tokens(offsets, [(0, 9, T.HUMAN)])
    assert ids[0] == IGNORE_INDEX and ids[-1] == IGNORE_INDEX
    assert ids[1] == ids[2] == HUMAN


def test_a_token_takes_the_label_of_the_span_covering_most_of_it():
    # token spans chars 3..7; span A covers 3..5 (2 chars), span B covers 5..7 (2 chars)
    # tie goes to the earlier span, by the rule stated in the module docstring
    offsets = [(3, 7)]
    ids = align_spans_to_tokens(offsets, [(0, 5, T.HUMAN), (5, 10, T.AI_GENERATED)])
    assert ids[0] == HUMAN
    # shift the boundary so B clearly wins
    ids = align_spans_to_tokens(offsets, [(0, 4, T.HUMAN), (4, 10, T.AI_GENERATED)])
    assert ids[0] == GEN


def test_boundary_is_not_off_by_one():
    """A systematic off-by-one shifts every boundary in the corpus by one token and
    boundary F1 reports a real-looking number that is wrong by a constant."""
    offsets = [(0, 5), (5, 10), (10, 15), (15, 20)]
    ids = align_spans_to_tokens(offsets, [(0, 10, T.HUMAN), (10, 20, T.AI_GENERATED)])
    assert ids == [HUMAN, HUMAN, GEN, GEN]


def test_round_trip_recovers_the_original_spans():
    offsets = [(0, 0), (0, 4), (4, 10), (10, 16), (0, 0)]
    spans = [(0, 10, T.HUMAN), (10, 16, T.AI_ASSISTED)]
    ids = align_spans_to_tokens(offsets, spans)
    assert spans_from_token_labels(offsets, ids) == spans


def test_ai_fraction_from_tokens_ignores_special_tokens():
    ids = [IGNORE_INDEX, HUMAN, HUMAN, GEN, GEN, IGNORE_INDEX]
    assert document_label_from_tokens(ids) == 0.5


def test_a_window_is_labelled_by_its_own_content_not_the_documents():
    """A mostly-human document with one AI paragraph must not stamp 'ai' on its
    mostly-human windows."""
    offsets = [(i * 4, i * 4 + 4) for i in range(1000)]
    spans = [(0, 3800, T.HUMAN), (3800, 4000, T.AI_GENERATED)]
    ex = windows_from_spans("d1", "grp_d1", Split.TRAIN, offsets, spans, size=512, stride=384)
    assert ex[0].document_label == 0
    assert ex[-1].ai_char_fraction > ex[0].ai_char_fraction
    assert all(e.source_group_id == "grp_d1" and e.split is Split.TRAIN for e in ex)
