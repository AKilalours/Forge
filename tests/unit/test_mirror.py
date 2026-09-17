"""Phase 2 tests.

The bar here: a mirror pipeline can produce a dataset that trains a 99-percent-accurate
detector which detects nothing except chat formatting. These tests exist to make that
failure mode loud.
"""

import pytest

from forge.common.config import load
from forge.common.schemas import Split
from forge.generation.assignment import (
    HeldOutLeakError,
    assert_no_held_out,
    assign_decoding,
    assign_family,
    held_in_families,
    held_out_families,
    parse_roster,
)
from forge.generation.attributes import (
    HeuristicExtractor,
    MirrorAttributes,
    VerbatimCopyError,
    assert_no_verbatim_copy,
    detect_structure,
)
from forge.generation.generators.base import (
    Decoding,
    UnpinnedRevisionError,
    VLLMGenerator,
    require_pinned_revision,
)
from forge.generation.mirror import (
    load_template,
    render_prompt,
    strip_wrapper,
    validate,
)
from forge.generation.run import HumanRef, generate_mirrors

SOURCE = (
    "The harbour authority published its annual dredging schedule in March, listing "
    "eleven separate operations across the estuary and the two tidal basins. Silt "
    "accumulation had increased noticeably since the previous survey, and the deeper "
    "berths required attention before the autumn shipping season began in earnest. "
    "Contractors were appointed in two lots, with the smaller basins handled under a "
    "framework agreement renewed the year before. A short consultation period followed."
)


# ---------------------------------------------------------------- attributes

def test_extractor_produces_all_prompt_fields():
    a = HeuristicExtractor().extract(SOURCE)
    fields = a.prompt_fields()
    for k in ("topic", "genre", "register", "target_tokens", "structure", "difficulty", "key_anchors"):
        assert fields[k], f"{k} is empty"


def test_target_tokens_matches_the_source_length():
    a = HeuristicExtractor().extract(SOURCE)
    assert a.target_tokens == len(SOURCE.split())


def test_anchors_are_keyphrases_not_sentences():
    a = HeuristicExtractor().extract(SOURCE)
    assert all(len(x.split()) <= 4 for x in a.key_anchors)


def test_verbatim_clause_in_an_anchor_is_rejected():
    """An LLM extractor asked to summarize will quote. If a clause survives into the
    prompt, the mirror shares surface text with its source and the classifier learns
    copy detection instead of an authorship signal."""
    bad = MirrorAttributes(
        topic="dredging", genre="explainer", register="formal", target_tokens=90,
        structure="prose", difficulty="moderate",
        key_anchors=["the harbour authority published its annual dredging schedule in march"],
    )
    with pytest.raises(VerbatimCopyError):
        assert_no_verbatim_copy(bad, SOURCE)


def test_short_topical_overlap_is_allowed():
    ok = MirrorAttributes(
        topic="dredging", genre="explainer", register="formal", target_tokens=90,
        structure="prose", difficulty="moderate", key_anchors=["dredging schedule", "tidal basins"],
    )
    assert_no_verbatim_copy(ok, SOURCE)  # must not raise


def test_structure_detection():
    assert detect_structure(SOURCE) == "prose"
    assert detect_structure("- one\n- two\n- three\n- four") == "listy"


# ---------------------------------------------------------------- assignment

def test_held_out_families_are_never_assigned():
    roster = parse_roster(load("configs/generation/generators.yaml"))
    held_out = {f.family for f in held_out_families(roster)}
    families = held_in_families(roster)
    assigned = {assign_family(f"doc_{i}", families).family for i in range(500)}
    assert not (assigned & held_out)


def test_leak_checker_raises():
    roster = parse_roster(load("configs/generation/generators.yaml"))
    with pytest.raises(HeldOutLeakError):
        assert_no_held_out(roster, {"llama", "gemma"})


def test_assignment_is_deterministic_so_a_run_can_resume():
    roster = held_in_families(parse_roster(load("configs/generation/generators.yaml")))
    grid = load("configs/generation/generators.yaml")["decoding_grid"]
    assert assign_family("fw_42", roster).family == assign_family("fw_42", roster).family
    assert assign_decoding("fw_42", grid) == assign_decoding("fw_42", grid)


def test_decoding_varies_across_documents():
    grid = load("configs/generation/generators.yaml")["decoding_grid"]
    seen = {(d.temperature, d.top_p) for d in (assign_decoding(f"d{i}", grid) for i in range(300))}
    assert len(seen) > 1, "a single temperature lets the detector learn that temperature's artifacts"


def test_unpinned_revision_is_refused():
    with pytest.raises(UnpinnedRevisionError):
        require_pinned_revision("qwen", "TODO_PIN_AT_FIRST_RUN")
    with pytest.raises(UnpinnedRevisionError):
        VLLMGenerator("qwen", "Qwen/Qwen2.5-7B-Instruct", "main")


# ---------------------------------------------------------------- prompt

def test_frozen_header_comment_never_reaches_the_model():
    tpl = load_template("src/forge/generation/prompts/mirror_v1.txt")
    assert "FROZEN" not in tpl and "mirror_v1" not in tpl


def test_render_fills_every_placeholder():
    a = HeuristicExtractor().extract(SOURCE)
    out = render_prompt(a, load_template("src/forge/generation/prompts/mirror_v1.txt"))
    assert "{" not in out


def test_render_rejects_an_unknown_placeholder():
    a = HeuristicExtractor().extract(SOURCE)
    with pytest.raises(KeyError):
        render_prompt(a, "write about {topic} in {nonexistent_field}")


# ---------------------------------------------------------------- validation

def _attrs():
    return HeuristicExtractor().extract(SOURCE)


def test_assistant_preamble_is_rejected_not_stripped():
    """The strongest AI tell in a carelessly built corpus, and trivially removed by an
    evader. Silently stripping it would hide a prompt problem behind a clean dataset."""
    a = _attrs()
    body = " ".join(["word"] * a.target_tokens)
    ok, reason = validate(f"Sure! Here's the piece you asked for:\n\n{body}", SOURCE, a)
    assert not ok and reason == "assistant_preamble"


def test_length_drift_is_rejected_in_both_directions():
    a = _attrs()
    assert validate(" ".join(["word"] * int(a.target_tokens * 0.3)), SOURCE, a)[1] == "too_short"
    assert validate(" ".join(["word"] * int(a.target_tokens * 3)), SOURCE, a)[1] == "too_long"


def test_a_regurgitated_source_is_rejected():
    a = _attrs()
    ok, reason = validate(SOURCE, SOURCE, a)
    assert not ok and reason == "near_duplicate_of_source"


def test_trailing_chat_meta_is_stripped():
    assert "let me know" not in strip_wrapper("Real content here.\nLet me know if you'd like changes!").lower()


def test_a_clean_generation_passes():
    a = _attrs()
    body = " ".join(["harbour", "silt", "berth", "survey", "contract"] * (a.target_tokens // 5))
    ok, reason = validate(body, SOURCE, a)
    assert ok, reason


# ---------------------------------------------------------------- runner

class BadGenerator:
    """Emits exactly the things validation exists to catch."""

    family = "bad"
    model_id = "forge/bad"
    revision = "v1"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def generate(self, prompts, decoding: Decoding):
        if self.mode == "preamble":
            return ["Certainly! Here is your text:\n\n" + " ".join(["word"] * 200) for _ in prompts]
        if self.mode == "short":
            return ["far too brief" for _ in prompts]
        raise ValueError(self.mode)


def _humans(n: int = 6) -> list[HumanRef]:
    return [
        HumanRef(f"t_{i}", f"grp_t_{i}", SOURCE.replace("harbour", f"harbour{i}"), "web",
                 Split.TRAIN if i % 3 else Split.TEST)
        for i in range(n)
    ]


def _cfgs():
    return load("configs/generation/generators.yaml"), load("configs/generation/mirror.yaml")


def test_runner_inherits_split_and_group_from_the_human():
    gcfg, mcfg = _cfgs()
    humans = _humans()
    res = generate_mirrors(humans, gcfg, mcfg, backend="fake")
    by_id = {h.doc_id: h for h in humans}
    assert res.docs
    for m in res.docs:
        h = by_id[m.source_human_id]
        assert m.split is h.split
        assert m.source_group_id == h.source_group_id


def test_runner_records_rejections_and_produces_nothing_from_a_bad_generator(monkeypatch):
    gcfg, mcfg = _cfgs()
    monkeypatch.setattr("forge.generation.run.build_generator", lambda spec, backend: BadGenerator("preamble"))
    res = generate_mirrors(_humans(), gcfg, mcfg, backend="fake")
    assert res.docs == []
    assert res.stats["rejected"]["assistant_preamble"] > 0
    assert res.stats["acceptance_rate"] == 0.0


def test_retries_are_counted_not_hidden(monkeypatch):
    gcfg, mcfg = _cfgs()
    monkeypatch.setattr("forge.generation.run.build_generator", lambda spec, backend: BadGenerator("short"))
    res = generate_mirrors(_humans(2), gcfg, mcfg, backend="fake")
    policy_retries = mcfg["validation"]["max_retries"]
    # each document is attempted (1 + max_retries) times before being abandoned
    assert res.stats["attempts"] == 2 * (1 + policy_retries)
    assert res.stats["rejected"]["too_short"] == 2 * (1 + policy_retries)


def test_only_held_in_families_appear_in_output():
    gcfg, mcfg = _cfgs()
    res = generate_mirrors(_humans(40), gcfg, mcfg, backend="fake")
    held_out = {f.family for f in held_out_families(parse_roster(gcfg))}
    assert not ({d.generator.family for d in res.docs} & held_out)


def test_fake_backend_is_labelled_in_the_record():
    """A fake-generated dataset must never be mistakable for a real one downstream."""
    gcfg, mcfg = _cfgs()
    res = generate_mirrors(_humans(3), gcfg, mcfg, backend="fake")
    assert all(d.mirror.attributes["backend"] == "fake" for d in res.docs)
    assert all(d.generator.revision == "fake" for d in res.docs)


# --------------------------------- the length instruction must be in the unit we measure

def test_the_active_mirror_prompt_asks_for_words_not_tokens():
    """The prompt said tokens; everything else in the pipeline means words.

    approx_token_count is len(text.split()), the extractor's target_tokens is a word
    count, and the validator's ratio divides words by that word target. Only the prompt
    said "tokens", so every model was told to hit a number in a unit nobody measured, and
    each one missed it differently: three families ran long and Qwen2.5-3B ran short
    enough to fail two thirds of its first attempts.

    Naming the unit correctly is right whether or not it fixes the yield. What it must
    not do is quietly revert.
    """
    from forge.common.config import load

    cfg = load("configs/generation/mirror_minimal.yaml")
    tpl = load_template(cfg["prompt_path"])
    assert "{target_tokens} words" in tpl
    assert "tokens." not in tpl, "the length instruction is back in the wrong unit"


def test_the_control_arm_prompt_asks_for_words_too():
    """Both arms or neither. A unit that differs between arms is another arm difference."""
    from forge.generation.random_synthetic import PROMPT

    assert "{target_tokens} words" in PROMPT
    assert "tokens." not in PROMPT


def test_the_frozen_v1_prompt_is_kept_rather_than_edited():
    """data_spec_v1: mirror_v1 is frozen, so a wording change is v2, not an edit to v1.

    Keeping the file means a document stamped mirror_v1 can still be read against the
    exact text that produced it.
    """
    import pathlib

    v1 = pathlib.Path("src/forge/generation/prompts/mirror_v1.txt")
    assert v1.exists(), "the frozen v1 prompt was deleted rather than superseded"
    assert "{target_tokens} tokens" in v1.read_text(), "v1 was edited in place"


def test_the_config_and_the_prompt_file_agree_on_the_version():
    from forge.common.config import load

    cfg = load("configs/generation/mirror_minimal.yaml")
    assert cfg["prompt_version"] == "mirror_v2"
    assert "mirror_v2" in cfg["prompt_path"]


def test_the_fake_generator_reads_the_target_in_either_unit():
    """The regression that caught the prompt change, pinned in both directions.

    _target_from_prompt matched "approximately N tokens" only. Correcting the prompts to
    say "words" made it miss, fall back to a 300-word default, and write 300 words
    against every target. The visible symptom was a mirror test rejecting all 18 of its
    documents as too_long, two modules away from the line that changed.
    """
    from forge.generation.generators.base import _target_from_prompt

    assert _target_from_prompt("- Length: approximately 275 words.") == 275
    assert _target_from_prompt("- Length: approximately 275 tokens.") == 275
    assert _target_from_prompt("no target here at all") == 300
