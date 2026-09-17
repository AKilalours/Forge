"""Arm A, the control arm. These tests exist to stop the comparison being rigged."""

import pytest

from forge.common.config import load
from forge.generation.assignment import held_out_families, parse_roster
from forge.generation.random_synthetic import (
    TOPICS,
    generate_random,
    length_pool_from_corpus,
    spec_for,
)

POOL = [150, 200, 250, 300, 350, 400]


def _cfg():
    return load("configs/generation/generators_minimal.yaml")


def test_random_documents_are_not_matched_to_anything():
    """The whole point of the control. topic_match, length_match and style_match must all
    be False, or Arm A is a weak mirror rather than a random baseline."""
    res = generate_random(20, POOL, _cfg(), backend="fake")
    assert res.docs
    for d in res.docs:
        assert d.mirror.topic_match is False
        assert d.mirror.length_match is False
        assert d.mirror.style_match is False
        assert d.mirror.attributes["arm"] == "random"


def test_topics_come_from_the_fixed_inventory_not_the_corpus():
    """Sampling topics from the human corpus would make this a mirror, not a control."""
    res = generate_random(30, POOL, _cfg(), backend="fake")
    assert all(d.mirror.attributes["topic"] in TOPICS for d in res.docs)


def test_length_distribution_matches_the_corpus_range_without_matching_documents():
    """Arm A must live in the same length RANGE, or the detector learns length and Arm A
    collapses for a reason unrelated to topic matching."""
    res = generate_random(40, POOL, _cfg(), backend="fake")
    targets = {d.mirror.target_tokens for d in res.docs}
    assert targets <= set(POOL)
    assert len(targets) > 1, "a single target length would be its own artifact"


def test_generation_is_deterministic_so_a_partial_run_resumes():
    assert spec_for(7, POOL) == spec_for(7, POOL)
    a = generate_random(10, POOL, _cfg(), backend="fake")
    b = generate_random(10, POOL, _cfg(), backend="fake")
    assert [d.text for d in a.docs] == [d.text for d in b.docs]


def test_only_held_in_families_generate_the_control_arm():
    """Same rule as mirrors: a held-out family in ANY training arm invalidates R3."""
    res = generate_random(40, POOL, _cfg(), backend="fake")
    held_out = {f.family for f in held_out_families(parse_roster(_cfg()))}
    assert not ({d.generator.family for d in res.docs} & held_out)


def test_the_control_arm_uses_the_same_generator_families_as_mirrors():
    """If mirrors win on generator diversity the result says nothing about matching."""
    from forge.generation.assignment import held_in_families

    res = generate_random(60, POOL, _cfg(), backend="fake")
    expected = {f.family for f in held_in_families(parse_roster(_cfg()))}
    assert set(res.stats["families_used"]) == expected


def test_random_documents_get_their_own_groups():
    """They have no human source, so they cannot inherit a group. Sharing one would put
    unrelated documents in the same split bucket."""
    res = generate_random(20, POOL, _cfg(), backend="fake")
    assert len({d.source_group_id for d in res.docs}) == len(res.docs)


def test_assistant_preamble_is_rejected_here_too():
    """Arm A must be held to the same quality bar as Arm B. A control polluted with chat
    formatting would lose for the wrong reason and flatter the mirror arm."""
    from forge.generation.generators.base import Decoding

    class Preambler:
        family, model_id, revision = "bad", "forge/bad", "v1"

        def generate(self, prompts, decoding: Decoding):
            return ["Certainly! Here is your text:\n\n" + " ".join(["word"] * 250)] * len(prompts)

    import forge.generation.run as run_mod

    orig = run_mod.build_generator
    run_mod.build_generator = lambda spec, backend: Preambler()
    try:
        res = generate_random(5, POOL, _cfg(), backend="fake")
        assert res.docs == []
        assert res.stats["rejected"]["assistant_preamble"] > 0
    finally:
        run_mod.build_generator = orig


def test_length_pool_is_read_from_the_real_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        length_pool_from_corpus(tmp_path)


def test_empty_length_pool_is_refused():
    with pytest.raises(ValueError):
        generate_random(5, [], _cfg(), backend="fake")


# ------------------------------------------- generating one family at a time, on a small disk

def _held_in_names():
    from forge.generation.assignment import held_in_families
    return sorted(f.family for f in held_in_families(parse_roster(_cfg())))


def test_four_one_family_runs_reconstruct_the_single_run_exactly():
    """The invariant that makes splitting the run safe rather than merely convenient.

    A pod with 30 GB of disk cannot hold four generator families at once, so the corpus
    has to be built one model at a time. That is only sound if the pieces add up to what
    one run would have produced. They do, because the family filter is applied AFTER
    assignment: every document is still assigned against the full roster, and each
    invocation generates its family's share rather than being handed the whole corpus.

    Filtering the roster before assignment would also have "worked", in the sense that it
    produced documents and no error, and it would have given the first model every
    document and called the result a four-family corpus.
    """
    whole = generate_random(60, POOL, _cfg(), backend="fake")
    pieces = [
        d
        for fam in _held_in_names()
        for d in generate_random(60, POOL, _cfg(), backend="fake", only_family=fam).docs
    ]
    by_id = {d.sample_id: d for d in pieces}
    assert len(by_id) == len(pieces), "two families claimed the same document"
    assert sorted(by_id) == sorted(d.sample_id for d in whole.docs)
    for d in whole.docs:
        same = by_id[d.sample_id]
        assert same.text == d.text
        assert same.generator.family == d.generator.family
        assert same.generator.seed == d.generator.seed
        assert same.split == d.split


def test_a_filtered_run_returns_only_that_family():
    fam = _held_in_names()[0]
    res = generate_random(60, POOL, _cfg(), backend="fake", only_family=fam)
    assert res.docs, "the filter produced nothing at all"
    assert {d.generator.family for d in res.docs} == {fam}
    assert res.stats["families_used"] == [fam]


def test_a_filtered_run_is_a_share_and_not_the_whole_corpus():
    """The failure this would have had if the filter were applied before assignment."""
    fam = _held_in_names()[0]
    whole = generate_random(60, POOL, _cfg(), backend="fake")
    part = generate_random(60, POOL, _cfg(), backend="fake", only_family=fam)
    assert 0 < len(part.docs) < len(whole.docs)


def test_an_unknown_family_is_refused_rather_than_silently_generating_nothing():
    """A typo in --only-family must not produce an empty parquet part and exit 0."""
    with pytest.raises(ValueError, match="not a held-in family"):
        generate_random(10, POOL, _cfg(), backend="fake", only_family="qwen2.5")


def test_a_held_out_family_cannot_be_selected_by_name():
    """--only-family is not a back door around the held-out rule."""
    held_out = sorted(f.family for f in held_out_families(parse_roster(_cfg())))
    with pytest.raises(ValueError, match="not a held-in family"):
        generate_random(10, POOL, _cfg(), backend="fake", only_family=held_out[0])


def test_parts_accumulate_instead_of_overwriting_each_other(tmp_path):
    """write_mirrors wrote part-000.parquet unconditionally, which is fine exactly once."""
    from forge.generation.run import write_mirrors

    total = 0
    for fam in _held_in_names():
        res = generate_random(60, POOL, _cfg(), backend="fake", only_family=fam)
        write_mirrors(res.docs, tmp_path, part=fam)
        total += len(res.docs)
    on_disk = sorted(tmp_path.glob("split=*/*.parquet"))
    named = {f.stem for f in on_disk}
    for fam in _held_in_names():
        assert f"part-{fam}" in named, f"{fam}'s part is not on disk"

    import pyarrow.parquet as pq
    rows = sum(pq.read_table(f).num_rows for f in on_disk)
    assert rows == total, "a later family overwrote an earlier one"


# ------------------------------------- the two arms must be validated the same way

def _mirror_validation():
    return load("configs/generation/mirror_minimal.yaml")["validation"]


def test_the_control_arm_takes_the_mirror_arms_length_band():
    """The arms are supposed to differ in MATCHING and in nothing else.

    Arm B read 0.6 to 1.6 from its config. Arm A used the signature defaults, 0.5 to 2.0,
    because the CLI passed nothing, so the control arm kept documents the mirror arm
    would have rejected. On the finished corpus arm A's documents ran a median of 306
    words against the human corpus's 257. A looser length policy on one arm is a second
    difference between the arms, and it points the same direction as the result.
    """
    v = _mirror_validation()
    res = generate_random(20, POOL, _cfg(), backend="fake", validation=v)
    assert res.stats["validation"]["length_ratio_min"] == v["length_ratio_min"]
    assert res.stats["validation"]["length_ratio_max"] == v["length_ratio_max"]
    assert res.stats["validation"]["max_retries"] == v["max_retries"]


def test_the_band_actually_applied_is_recorded_in_the_stats():
    """A band that is not written down is a band nobody can check afterwards."""
    res = generate_random(20, POOL, _cfg(), backend="fake")
    assert set(res.stats["validation"]) == {
        "length_ratio_min", "length_ratio_max", "max_retries",
    }


def test_the_mirror_config_band_is_narrower_than_the_old_control_default():
    """Pins the asymmetry that caused this, so a future edit cannot quietly undo it.

    0.5 to 2.0 permits double the target length but only half of it. Every generator
    tested writes long, so an asymmetric band does not average out; it shifts the arm.
    """
    v = _mirror_validation()
    assert v["length_ratio_min"] > 0.5
    assert v["length_ratio_max"] < 2.0


def test_an_explicit_argument_still_wins_over_an_empty_validation_block():
    res = generate_random(20, POOL, _cfg(), backend="fake",
                          length_ratio_min=0.9, length_ratio_max=1.1, validation={})
    assert res.stats["validation"]["length_ratio_min"] == 0.9
    assert res.stats["validation"]["length_ratio_max"] == 1.1
