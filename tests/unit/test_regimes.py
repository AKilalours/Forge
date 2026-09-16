"""Regime constructors. Plan section 13, gap G3."""

from dataclasses import dataclass

import pytest

from forge.evaluation.regimes import (
    RegimeError,
    summarize,
    temporal,
    unseen_domain,
    unseen_generator,
)


@dataclass
class Rec:
    doc_id: str
    source_group_id: str
    domain: str = "web"
    generator_family: str = "human"
    generator_released: str = ""


def _corpus():
    out = []
    for i, dom in enumerate(["news", "books", "reviews", "academic", "poetry"] * 6):
        g = f"grp_{i}"
        out.append(Rec(f"h{i}", g, dom, "human"))
        fam = ["qwen", "phi", "gemma"][i % 3]
        rel = {"qwen": "2024-09", "phi": "2024-08", "gemma": "2025-03"}[fam]
        out.append(Rec(f"a{i}", g, dom, fam, rel))     # mirror shares the human's group
    return out


def test_unseen_domain_holds_out_whole_domains():
    s = unseen_domain(_corpus(), ["academic", "poetry"])
    assert set(s.test_keys) == {"academic", "poetry"}
    assert not (set(s.train_keys) & set(s.test_keys))


def test_a_mirror_never_lands_opposite_its_human():
    """Partitioning by row instead of by group puts a human in train and its mirror in
    the OOD test set, so 'unseen domain' measures memorisation."""
    corpus = _corpus()
    s = unseen_domain(corpus, ["academic"])
    by_id = {r.doc_id: r.source_group_id for r in corpus}
    train_groups = {by_id[i] for i in s.train_ids}
    test_groups = {by_id[i] for i in s.test_ids}
    assert not (train_groups & test_groups)


def test_requesting_a_domain_that_does_not_exist_fails_loudly():
    with pytest.raises(RegimeError, match="no records matched"):
        unseen_domain(_corpus(), ["nonexistent_domain"])


def test_holding_out_everything_fails_loudly():
    with pytest.raises(RegimeError, match="entire dataset"):
        unseen_domain(_corpus(), ["news", "books", "reviews", "academic", "poetry"])


def test_unseen_generator_holds_out_families():
    s = unseen_generator(_corpus(), ["gemma"])
    assert s.test_keys == ["gemma"]
    assert "gemma" not in s.train_keys


def test_human_cannot_be_a_held_out_family():
    """An unseen-generator test set with no human examples has an undefined FPR, and FPR
    is the metric that matters most."""
    with pytest.raises(RegimeError, match="cannot be a held-out"):
        unseen_generator(_corpus(), ["human"])


def test_temporal_splits_on_model_release_date():
    s = temporal(_corpus(), cutoff="2024-12")
    assert "release date, not generation date" in s.note
    assert s.test_ids and s.train_ids


def test_temporal_holds_out_only_the_newer_models():
    corpus = _corpus()
    s = temporal(corpus, cutoff="2024-12")
    by_id = {r.doc_id: r for r in corpus}
    newer = {by_id[i].generator_family for i in s.test_ids if by_id[i].generator_family != "human"}
    assert newer == {"gemma"}


def test_summarize_shows_what_is_available():
    d = summarize(_corpus())
    assert set(d["domains"]) == {"news", "books", "reviews", "academic", "poetry"}
    assert "gemma" in d["generator_families"]


def test_group_key_comes_from_the_ai_member_not_the_human():
    """Regression guard for a real bug.

    _partition originally took each group's key from its FIRST record. The human is
    usually first and carries family 'human' with no release date, so the mirror's family
    and release date were never seen and R3 and R4 held out nothing at all.
    """
    from forge.evaluation.regimes import resolve_group_key

    recs = [Rec("h1", "g1", "human"), Rec("a1", "g1", generator_family="gemma")]
    keys = resolve_group_key(recs, lambda r: r.generator_family, lambda r: r.source_group_id,
                             neutral="human")
    assert keys["g1"] == "gemma"


def test_a_human_with_no_mirror_stays_neutral_and_trains():
    from forge.evaluation.regimes import resolve_group_key

    recs = [Rec("h9", "g9", generator_family="human")]
    keys = resolve_group_key(recs, lambda r: r.generator_family, lambda r: r.source_group_id,
                             neutral="human")
    assert keys["g9"] == "human"


def test_r3_test_set_contains_both_classes():
    """Otherwise FPR on the unseen-generator regime is undefined."""
    corpus = _corpus()
    s = unseen_generator(corpus, ["gemma"])
    by_id = {r.doc_id: r for r in corpus}
    fams = {by_id[i].generator_family for i in s.test_ids}
    assert "human" in fams and "gemma" in fams
