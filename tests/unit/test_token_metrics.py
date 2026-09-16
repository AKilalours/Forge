"""Token, segment and boundary F1. Plan section 22, gap G1."""


from forge.evaluation.token_metrics import (
    boundary_f1,
    extract_boundaries,
    extract_segments,
    score_tokens,
    segment_f1,
    token_f1,
)
from forge.modeling.alignment import IGNORE_INDEX

H, A = 0, 2   # human, ai_generated


def test_perfect_prediction_scores_one_everywhere():
    y = [H, H, A, A, A, H, H]
    s = score_tokens(y, y)
    assert s.token_f1_macro == 1.0 and s.segment_f1 == 1.0 and s.boundary_f1 == 1.0


def test_token_f1_alone_hides_a_model_that_finds_nothing():
    """This is the reason segment and boundary F1 exist.

    With AI regions at 5 percent of tokens, predicting all-human scores well on the
    majority class while finding zero regions.
    """
    y_true = [H] * 95 + [A] * 5
    y_pred = [H] * 100
    macro, per, _ = token_f1(y_true, y_pred)
    assert per[H] > 0.97, "the human class alone looks excellent"
    sf1, _, rec, _, _ = segment_f1(y_true, y_pred)
    assert sf1 == 0.0 and rec == 0.0, "but not a single segment was found"


def test_ignored_positions_are_excluded():
    y_true = [IGNORE_INDEX, H, A, IGNORE_INDEX]
    y_pred = [A, H, A, H]
    macro, per, support = token_f1(y_true, y_pred)
    assert support == {H: 1, A: 1}


def test_segments_do_not_join_across_ignored_positions():
    """Joining would merge two genuinely separate AI regions split by a special token,
    inflating recall."""
    labels = [A, A, IGNORE_INDEX, A, A]
    assert len(extract_segments(labels)) == 2


def test_segment_matching_is_one_to_one():
    """Without this, one huge predicted segment matches every true segment and scores
    perfect recall while being useless to a reader."""
    y_true = [A, A, H, H, A, A, H, H, A, A]
    y_pred = [A] * 10
    _, _, rec, n_true, n_pred = segment_f1(y_true, y_pred)
    assert n_true == 3 and n_pred == 1
    assert rec <= 1 / 3


def test_segment_iou_threshold_rejects_a_poor_overlap():
    y_true = [H, H, H, H, A, A, A, A]
    y_pred = [H, H, H, H, H, H, H, A]      # only 1 of 4 AI tokens found, IoU 0.25
    assert segment_f1(y_true, y_pred, iou_threshold=0.5)[0] == 0.0
    assert segment_f1(y_true, y_pred, iou_threshold=0.2)[0] > 0.0


def test_boundary_tolerance_distinguishes_late_from_missing():
    """Being one token late on a seam is a different failure from missing the seam."""
    y_true = [H, H, H, H, A, A, A, A]
    late = [H, H, H, H, H, A, A, A]        # boundary one token late
    missing = [H] * 8                       # no boundary at all
    assert boundary_f1(y_true, late, tolerance=2)[0] == 1.0
    assert boundary_f1(y_true, late, tolerance=0)[0] == 0.0
    assert boundary_f1(y_true, missing, tolerance=2)[0] == 0.0


def test_boundaries_are_label_changes():
    assert extract_boundaries([H, H, A, A, H]) == [2, 4]


def test_a_model_with_right_regions_and_sloppy_edges_is_visible():
    """Exactly the case token F1 obscures: high token score, perfect segments, poor
    boundaries."""
    y_true = [H] * 10 + [A] * 20 + [H] * 10
    y_pred = [H] * 13 + [A] * 14 + [H] * 13
    s = score_tokens(y_true, y_pred, tolerance=1)
    assert s.token_f1_macro > 0.8
    assert s.segment_f1 == 1.0
    assert s.boundary_f1 == 0.0


def test_scores_serialise():
    d = score_tokens([H, A, A, H], [H, A, A, H]).as_dict()
    for k in ("token_f1_macro", "segment_f1", "boundary_f1", "support"):
        assert k in d
