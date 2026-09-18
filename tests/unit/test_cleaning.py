from forge.cleaning import pii
from forge.cleaning.filters import check_length, check_quality, repeated_line_ratio, symbol_ratio
from forge.cleaning.normalize import normalize, strip_invisible, strip_markup

PROSE = (
    "The harbour authority published its annual dredging schedule in March, listing "
    "eleven separate operations across the estuary and the two tidal basins. Silt "
    "accumulation had increased noticeably since the previous survey, and the deeper "
    "berths required attention before the autumn shipping season began in earnest."
)


def test_markup_is_removed_before_anything_measures_length():
    html = f"<html><body><script>var x = 1;</script><p>{PROSE}</p></body></html>"
    out = normalize(html)
    assert "<p>" not in out and "var x" not in out
    assert out.startswith("The harbour authority")


def test_zero_width_characters_are_stripped():
    # also an adversarial attack vector, so human text must never carry them
    dirty = "hello​world﻿"
    assert strip_invisible(dirty) == "helloworld"


def test_normalize_is_idempotent():
    once = normalize(PROSE + "   \n\n\n\n  trailing  ")
    assert normalize(once) == once


def test_markdown_links_keep_their_text():
    assert "example" in strip_markup("see [example](https://a.b/c) here")
    assert "https" not in strip_markup("see [example](https://a.b/c) here")


def test_short_text_fails_length():
    assert "too_short_chars" in check_length("Too short.")


def test_symbol_soup_fails_quality():
    junk = "### >>> ||| " * 60
    assert symbol_ratio(junk) > 0.2
    assert check_quality(junk)


def test_repeated_boilerplate_is_detected():
    boiler = "Click here to continue.\n" * 40
    assert repeated_line_ratio(boiler) > 0.9


def test_good_prose_passes_both_gates():
    text = PROSE * 3
    assert not check_length(text)
    assert not check_quality(text)


def test_pii_is_replaced_with_a_typed_placeholder_not_deleted():
    text = "Reach jane.doe@example.com or 555-123-4567 today."
    out, flags = pii.redact(text)
    assert "[EMAIL]" in out and "[PHONE]" in out
    assert "jane.doe" not in out and "555-123-4567" not in out
    assert set(flags) == {"EMAIL", "PHONE"}


def test_luhn_stops_the_credit_card_regex_firing_on_any_long_number():
    # a 16-digit run that fails Luhn is not a card number
    assert "CREDIT_CARD" not in pii.detect("order reference 1234567812345678 shipped")
    # a valid Luhn number is caught
    assert "CREDIT_CARD" in pii.detect("card 4539578763621486 on file")


# ------------------------------------------------------- language identification

def test_language_detection_never_raises_regardless_of_environment():
    """Regression guard for a bug that only appeared on the GPU pod.

    detect() used to raise NotImplementedError whenever the fastText LIBRARY was
    importable, which is a proxy for "a model is usable", not the thing itself. On a
    laptop where the import failed the heuristic ran and tests passed; on the pod, where
    the data extra provides fasttext, the same call raised. Identical code, opposite
    behaviour, and `forge ingest` would have crashed on the first document from any
    source without an upstream score.
    """
    from forge.cleaning import langid

    langid.reset_model_cache()
    for text in [PROSE, "", "   ", "a", "Der Bericht beschreibt die Entwicklung."]:
        r = langid.detect(text)
        assert isinstance(r.language, str) and 0.0 <= r.score <= 1.0


def test_upstream_score_is_preferred_over_recomputing():
    """FineWeb already carries a fastText language score. Recomputing it would be work
    to reproduce a field the dataset hands us."""
    from forge.cleaning import langid

    r = langid.detect("anything at all", upstream=("en", 0.97))
    assert r.detector == "upstream" and r.score == 0.97


def test_a_missing_model_path_falls_back_rather_than_failing(monkeypatch):
    from forge.cleaning import langid

    monkeypatch.setenv(langid.MODEL_PATH_ENV, "/nonexistent/lid.176.bin")
    langid.reset_model_cache()
    r = langid.detect(PROSE)
    assert r.detector == "heuristic" and r.language == "en"
    langid.reset_model_cache()


def test_english_prose_is_detected_by_the_heuristic():
    from forge.cleaning import langid

    langid.reset_model_cache()
    assert langid.detect(PROSE).language == "en"


# ---------------------------------------------------------------------------
# CONFUSABLES FOLDING, which exists because the adversarial lab measured the gap.
#
# NFKC normalises compatibility characters. Cyrillic small a (U+0430) is a different
# letter, not a compatibility form of Latin a, so NFKC leaves it and the homoglyph attack
# walks through the whole preprocessing path. Measured on the v0.2-min baseline arm at
# severity 0.2: clean FNR 0.000, attacked 0.306 raw and 0.308 after normalisation. The
# preprocessing benefit was -0.002, which is to say there was no defence.
#
# Folding is opt in. normalize() is stage 3 and 4 of a frozen ingestion contract, and the
# deployed checkpoints were trained on unfolded text, so turning it on by default would
# change every document in the corpus and move inference away from training.
# ---------------------------------------------------------------------------


def test_nfkc_alone_does_not_touch_cyrillic_lookalikes() -> None:
    """The premise. If NFKC did fold these, the attack would already be defused and the
    whole folding stage would be dead code."""
    from forge.cleaning.normalize import normalize

    attacked = "Тhе асаdеmiс рарer"
    assert normalize(attacked) == attacked
    assert normalize(attacked) != "The academic paper"


def test_folding_recovers_the_homoglyph_attack_exactly() -> None:
    """Against the attack's own substitution table, so the test cannot drift from the
    threat it is written for."""
    from forge.adversarial.attacks import HOMOGLYPHS
    from forge.cleaning.normalize import normalize

    original = "The academic paper explores complex topics"
    attacked = "".join(HOMOGLYPHS.get(ch, ch) for ch in original)
    assert attacked != original, "the fixture must actually be attacked"
    assert normalize(attacked, fold_confusables=True) == original


def test_every_character_the_attack_can_insert_has_a_folding_entry() -> None:
    """THE DRIFT THIS PREVENTS. A new homoglyph added to the attack table with no folding
    entry leaves a hole that the lab would report as partial robustness rather than as a
    missing defence."""
    from forge.adversarial.attacks import HOMOGLYPHS
    from forge.cleaning.normalize import fold_confusable_letters

    for latin, lookalike in HOMOGLYPHS.items():
        folded = fold_confusable_letters(lookalike)
        assert folded == latin, (
            f"the attack substitutes {lookalike!r} for {latin!r} but folding gives "
            f"{folded!r}. Add it to _CONFUSABLES."
        )


def test_folding_is_idempotent_and_leaves_latin_alone() -> None:
    from forge.cleaning.normalize import fold_confusable_letters

    plain = "The quick brown fox, 42 times; naive coordination."
    assert fold_confusable_letters(plain) == plain
    once = fold_confusable_letters("Тhе асаdеmiс")
    assert fold_confusable_letters(once) == once


def test_normalize_stays_idempotent_with_folding_on() -> None:
    """The docstring promises normalize(normalize(x)) == normalize(x). Inserting a stage
    is the way that promise gets broken."""
    from forge.cleaning.normalize import normalize

    for text in ("Тhе  асаdеmiс\n\n\n рарer", "<b>Тext</b> &amp; more​"):
        once = normalize(text, fold_confusables=True)
        assert normalize(once, fold_confusables=True) == once


def test_folding_is_off_by_default_because_the_contract_is_frozen() -> None:
    """Not a style preference. normalize() is stages 3 and 4 of docs/data_spec_v1.md and
    the committed checkpoints were trained on what it emits today. A default flip would
    silently re-specify the corpus and invalidate the fitted thresholds."""
    from forge.cleaning.normalize import normalize

    attacked = "Тhе асаdеmiс"
    assert normalize(attacked) == attacked
    assert normalize(attacked, fold_confusables=True) == "The academic"
