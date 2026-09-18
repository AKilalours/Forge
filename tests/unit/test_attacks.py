"""The adversarial laboratory.

The central risk in this phase is producing a flattering robustness number. Three ways
that happens, each with tests below: an attack that silently does nothing, an attack
whose effect is removed by preprocessing rather than by the model, and a validity check
that filters out precisely the attacks that work.
"""

import pytest

from forge.adversarial.attacks import (
    ATTACKS,
    HOMOGLYPHS,
    RUNNABLE_OFFLINE,
    ModelAttack,
    apply_attack,
    fold_homoglyphs,
    is_noop,
    preserves_meaning,
    survives_preprocessing,
)
from forge.cleaning.normalize import normalize

TEXT = (
    "The harbour authority published its annual dredging schedule in March. "
    "It listed eleven separate operations across the estuary and two tidal basins. "
    "Silt accumulation had increased noticeably since the previous survey. "
    "Contractors were appointed in two lots because the work is difficult and important. "
    "A short consultation period followed, during which several associations objected. "
    "The committee agreed to begin the smaller basins first and to show the plan publicly."
)


# --------------------------------------------------------------- determinism

@pytest.mark.parametrize("name", RUNNABLE_OFFLINE)
def test_attacks_are_deterministic(name):
    """A robustness table must be exactly regenerable."""
    sev = ATTACKS[name].severities[-1]
    assert apply_attack(TEXT, name, "doc1", sev) == apply_attack(TEXT, name, "doc1", sev)


@pytest.mark.parametrize("name", RUNNABLE_OFFLINE)
def test_different_documents_get_different_perturbations(name):
    sev = ATTACKS[name].severities[-1]
    a = apply_attack(TEXT, name, "doc1", sev)
    b = apply_attack(TEXT, name, "doc2", sev)
    if a == TEXT and b == TEXT:
        pytest.skip("attack is a no-op on this fixture at this severity")
    assert a != b


# --------------------------------------------------------------- severity semantics

def test_severity_actually_scales_the_perturbation_rate():
    """Regression guard for a real bug.

    The first implementation combined `index % step == 0` with a random bit, which halves
    the rate at best and collapses to zero when eligible positions are sparse. Measured on
    a four-sentence fixture, synonym_swap and article_deletion perturbed NOTHING at their
    configured severities, and an attack that does nothing reports as perfect robustness.
    """
    low = apply_attack(TEXT, "case_perturb", "doc1", 0.02)
    high = apply_attack(TEXT, "case_perturb", "doc1", 0.30)
    n_low = sum(1 for a, b in zip(TEXT, low, strict=False) if a != b)
    n_high = sum(1 for a, b in zip(TEXT, high, strict=False) if a != b)
    assert n_low > 0, "a 2 percent attack must still perturb something on a long document"
    assert n_high > n_low * 3


def test_sparse_target_attacks_still_fire_when_targets_exist():
    """severity applies to ELIGIBLE positions, not to every character, so it means the
    same thing regardless of how many targets a document happens to contain."""
    out = apply_attack(TEXT, "synonym_swap", "doc1", 1.0)
    assert out != TEXT
    assert "large" in out or "significant" in out or "challenging" in out or "display" in out


def test_article_deletion_removes_articles():
    out = apply_attack(TEXT, "article_deletion", "doc1", 1.0)
    assert out.lower().count(" the ") < TEXT.lower().count(" the ")


def test_sentence_reorder_preserves_every_sentence():
    out = apply_attack(TEXT, "sentence_reorder", "doc1", 0.3)
    assert sorted(out.split(".")) == sorted(TEXT.split("."))


# --------------------------------------------------------------- preprocessing

DEFUSED = [n for n in RUNNABLE_OFFLINE if ATTACKS[n].defused_by_preprocessing]
SURVIVING = [n for n in RUNNABLE_OFFLINE if not ATTACKS[n].defused_by_preprocessing]


@pytest.mark.parametrize("name", DEFUSED)
def test_attacks_flagged_as_defused_really_are(name):
    """The flag is cross-checked against measurement so it cannot drift from the truth.

    This matters because credit for defeating these attacks belongs to two lines of
    normalize(), not to the model, and any deployment that skips preprocessing is
    unprotected.
    """
    for sev in ATTACKS[name].severities:
        assert not survives_preprocessing(TEXT, name, "doc1", sev), (
            f"{name} is flagged defused_by_preprocessing but survives normalize()"
        )


@pytest.mark.parametrize("name", SURVIVING)
def test_attacks_flagged_as_surviving_really_do(name):
    sev = ATTACKS[name].severities[-1]
    attacked = apply_attack(TEXT, name, "doc1", sev)
    if is_noop(TEXT, attacked):
        pytest.skip("no-op on this fixture")
    assert survives_preprocessing(TEXT, name, "doc1", sev), (
        f"{name} is flagged as surviving but normalize() removes it"
    )


def test_nfkc_does_not_remove_cyrillic_homoglyphs():
    """A common misconception. Cyrillic small a (U+0430) is a distinct letter, not a
    compatibility form of Latin a, so NFKC leaves it untouched. This is the attack in the
    defused set's blind spot and the one to take seriously."""
    attacked = apply_attack(TEXT, "homoglyph_substitute", "doc1", 0.2)
    assert normalize(attacked) != normalize(TEXT)
    assert any(g in normalize(attacked) for g in HOMOGLYPHS.values())


def test_zero_width_is_stripped_by_ingestion():
    attacked = apply_attack(TEXT, "zero_width_insert", "doc1", 0.05)
    assert "​" in attacked
    assert "​" not in normalize(attacked)


# --------------------------------------------------------------- validity

def test_homoglyph_attacks_are_not_wrongly_judged_invalid():
    """Regression guard for a real bug.

    preserves_meaning() originally compared raw token overlap. Cyrillic look-alikes are
    invisible to a reader but break every word containing them, so a perfectly readable
    homoglyph attack scored a Jaccard near zero and was discarded as vandalism. That
    filters out exactly the attacks that work and reports a flattering robustness number
    built on a threat that was excluded from the measurement.
    """
    attacked = apply_attack(TEXT, "homoglyph_substitute", "doc1", 0.2)
    assert preserves_meaning(TEXT, attacked)


def test_fold_homoglyphs_recovers_the_original():
    attacked = apply_attack(TEXT, "homoglyph_substitute", "doc1", 0.5)
    assert fold_homoglyphs(attacked) == TEXT


def test_genuine_vandalism_is_rejected():
    assert not preserves_meaning(TEXT, "".join(reversed(TEXT[:40])))
    assert not preserves_meaning(TEXT, "")


@pytest.mark.parametrize("name", RUNNABLE_OFFLINE)
def test_every_offline_attack_preserves_meaning_at_its_configured_severities(name):
    """Severities in the registry are the ones a real run uses, so all of them must
    produce valid evasions rather than vandalism."""
    for sev in ATTACKS[name].severities:
        out = apply_attack(TEXT, name, "doc1", sev)
        assert preserves_meaning(TEXT, out), f"{name} at severity {sev} destroys the text"


# --------------------------------------------------------------- no-ops and honesty

def test_a_noop_is_detectable():
    """A no-op scores as perfect robustness. On a document with no substitutable words,
    synonym_swap genuinely changes nothing, and counting that as a successful defence
    reports the lexicon's coverage as the model's strength."""
    bare = "Zzz qqq wwww. Zzz qqq wwww. Zzz qqq wwww."
    assert is_noop(bare, apply_attack(bare, "synonym_swap", "d", 0.05))


def test_model_attacks_refuse_rather_than_faking():
    """A stub paraphraser would produce a meaningless robustness number, and paraphrase is
    the attack that matters most because it is what commercial humanisers actually do."""
    for name in ("paraphrase_llm", "humanizer_tool", "ai_assisted_edit"):
        assert isinstance(ATTACKS[name].fn, ModelAttack)
        with pytest.raises(NotImplementedError):
            apply_attack(TEXT, name, "doc1", 1.0)


def test_unknown_attack_is_an_error():
    with pytest.raises(ValueError):
        apply_attack(TEXT, "not_an_attack", "d", 0.1)


def test_config_and_code_agree_in_both_directions():
    """This test already caught real drift: grammar_edit was configured but not
    implemented, and three implemented attacks were missing from the config."""
    from forge.common.config import load

    cfg = {a["id"]: a for a in load("configs/eval/attacks.yaml")["attacks"]}
    assert set(cfg) == set(ATTACKS), (
        f"configured but not implemented: {set(cfg) - set(ATTACKS)}; "
        f"implemented but not configured: {set(ATTACKS) - set(cfg)}"
    )


def test_config_severities_and_preprocessing_flags_match_the_code():
    from forge.common.config import load

    cfg = {a["id"]: a for a in load("configs/eval/attacks.yaml")["attacks"]}
    for name, attack in ATTACKS.items():
        assert tuple(cfg[name]["severities"]) == attack.severities, f"{name} severities differ"
        assert cfg[name]["defused_by_preprocessing"] == attack.defused_by_preprocessing, (
            f"{name} preprocessing flag differs"
        )
        assert cfg[name]["runnable_offline"] == (name in RUNNABLE_OFFLINE), (
            f"{name} runnable_offline differs"
        )


# --------------------------------------------------------------- the lab runner

def _corpus(n=40):
    return (
        [TEXT.replace("harbour", f"harbour{i}") for i in range(n)],
        [f"doc{i}" for i in range(n)],
    )


class CharacterSensitiveDetector:
    """Scores AI unless the text contains characters outside plain ASCII.

    A crude but realistic stand-in: token-level models genuinely fall apart when
    homoglyphs and zero-width characters break their vocabulary.
    """

    def __call__(self, texts):
        return [0.05 if any(ord(c) > 127 for c in t) else 0.95 for t in texts]


def test_preprocessing_defends_against_zero_width_and_the_lab_shows_it():
    """The whole point of measuring both conditions.

    Against a raw input path this attack works. Against FORGE's production path it does
    nothing, because strip_invisible() removes it. Reporting only one column would either
    credit normalize() to the model or describe a threat production already handles.
    """
    from forge.adversarial.lab import run_attacks

    texts, ids = _corpus()
    res = [r for r in run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                                  attacks=["zero_width_insert"])][0]
    assert res.delta_raw > 0.5, "the attack must work against a raw input path"
    assert abs(res.delta_preprocessed) < 1e-9, "and be fully defused by preprocessing"
    assert res.as_dict()["preprocessing_benefit"] > 0.5


def test_homoglyphs_defeat_preprocessing_and_the_lab_shows_that_too():
    from forge.adversarial.lab import run_attacks

    texts, ids = _corpus()
    res = [r for r in run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                                  attacks=["homoglyph_substitute"])
           if r.severity == 0.20][0]
    assert res.delta_preprocessed > 0.5, "NFKC does not remove Cyrillic look-alikes"
    assert res.as_dict()["preprocessing_benefit"] < 1e-9


def test_noops_are_excluded_from_the_score_not_counted_as_defences():
    from forge.adversarial.lab import run_attacks

    bare = ["Zzz qqq wwww. Zzz qqq wwww. Zzz qqq wwww."] * 10
    res = run_attacks(bare, [f"d{i}" for i in range(10)], CharacterSensitiveDetector(), 0.5,
                      attacks=["synonym_swap"])
    assert all(r.n_noop == 10 and r.n_scored == 0 for r in res)


def test_table_renders_worst_attack_first():
    from forge.adversarial.lab import render_table, run_attacks

    texts, ids = _corpus()
    res = run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                      attacks=["zero_width_insert", "homoglyph_substitute"])
    table = render_table(res)
    assert "homoglyph_substitute" in table.split("\n")[2]


# ---------------------------------------------------------------------------
# THE CANDIDATE DEFENCES, and the control that decides whether they are defences.
#
# Two attacks walk through normalisation untouched: homoglyph substitution at 0.306 and
# case perturbation at 0.978, both moving the preprocessed column by under a thousandth.
# Folding and casefolding are the two things that could be done about them at inference
# time, and both transform CLEAN documents as well as attacked ones. So each condition
# carries its own clean baseline, and a delta is measured against the baseline from the
# same condition. Scoring an attacked column under one transform against a clean column
# measured under another is the easiest way to publish a defence that does not work.
# ---------------------------------------------------------------------------


def test_each_condition_gets_its_own_clean_baseline() -> None:
    """THE CONTROL. Without it, a transform that costs clean accuracy reports as a win:
    its attacked FNR falls toward the clean FNR it just raised, and the delta shrinks for
    the wrong reason."""
    from forge.adversarial.lab import run_attacks

    texts, ids = _corpus()
    res = run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                      attacks=["homoglyph_substitute"],
                      conditions=("raw", "normalised", "folded"))
    for r in res:
        assert set(r.clean) == {"raw", "normalised", "folded"}
        assert set(r.fnr) == {"raw", "normalised", "folded"}
        for condition in r.fnr:
            assert r.delta(condition) == pytest.approx(
                r.fnr[condition] - r.clean[condition]
            ), "a delta must be against the baseline from its own condition"


def test_the_published_field_names_keep_their_old_meaning() -> None:
    """The artifact schema has always published fnr_raw, fnr_preprocessed and clean_fnr,
    and reports reference them. The conditions dict is additive, not a rename."""
    from forge.adversarial.lab import run_attacks

    texts, ids = _corpus()
    res = run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                      attacks=["homoglyph_substitute"])
    for r in res:
        assert r.fnr_raw == r.fnr["raw"]
        assert r.fnr_preprocessed == r.fnr["normalised"]
        assert r.clean_fnr == r.clean["normalised"]
        d = r.as_dict()
        for key in ("clean_fnr", "fnr_raw", "fnr_preprocessed", "delta_fnr_raw",
                    "delta_fnr_preprocessed", "preprocessing_benefit"):
            assert key in d, f"{key} disappeared from the artifact"
        assert "conditions" in d


def test_the_production_condition_cannot_be_dropped_from_a_run() -> None:
    """Every published delta in this repository is stated against the normalised path. A
    run without it produces a table whose numbers have no relationship to the deployed
    system, and nothing downstream would notice."""
    from forge.adversarial.lab import run_attacks

    texts, ids = _corpus()
    with pytest.raises(ValueError, match="production path"):
        run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                    attacks=["homoglyph_substitute"], conditions=("raw",))


def test_an_unknown_condition_is_refused_before_any_scoring() -> None:
    """Scoring 500 documents through 17 attacks before discovering a typo is an hour of
    CPU for an error message."""
    from forge.adversarial.lab import run_attacks

    texts, ids = _corpus()
    with pytest.raises(ValueError, match="unknown conditions"):
        run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                    attacks=["homoglyph_substitute"],
                    conditions=("normalised", "case_folded"))


def test_the_default_run_scores_exactly_what_it_always_did() -> None:
    """Each extra condition rescores every cell and the clean baseline, so the default has
    to stay at two or the existing lab run silently doubles in cost."""
    from forge.adversarial.lab import DEFAULT_CONDITIONS, run_attacks

    assert DEFAULT_CONDITIONS == ("raw", "normalised")
    texts, ids = _corpus()
    res = run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                      attacks=["homoglyph_substitute"])
    assert all(set(r.fnr) == {"raw", "normalised"} for r in res)


def test_the_table_shows_clean_and_attacked_for_every_condition() -> None:
    """A defence that trades clean accuracy for attacked accuracy must not be able to
    look like a straight win in the rendered table."""
    from forge.adversarial.lab import render_table, run_attacks

    texts, ids = _corpus()
    res = run_attacks(texts, ids, CharacterSensitiveDetector(), 0.5,
                      attacks=["homoglyph_substitute"],
                      conditions=("raw", "normalised", "folded"))
    header = render_table(res).split("\n")[0]
    for tag in ("raw.cln", "raw.atk", "normal.cln", "folded.cln", "folded.atk"):
        assert tag in header, f"{tag} missing from {header!r}"
