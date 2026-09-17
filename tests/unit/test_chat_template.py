"""Instruction-tuned models must be prompted as instruction-tuned models.

THE BUG, AND WHY IT SURVIVED. Both generation backends handed the arm-A prompt to the
model as a raw completion string. Every held-in family in the roster is instruction
tuned, and the prompt is plain instruction text: "Write a {genre} ... Output only the
text itself." Sent without a chat template, the models were not being instructed at all.
They were continuing a passage that happened to read like an instruction.

It produced documents, so nothing failed. A probe run on SmolLM2-1.7B-Instruct accepted
193 of 565 attempts: 169 empty, 201 too short, 2 assistant preambles. The last number is
the diagnosis. The validator rejects "Sure, here's a..." openings because a chat model
reaches for them; two in 565 is not a chat model being obedient, it is a model that was
never in chat mode.

The stakes are not throughput. An instruct model CONTINUING a prompt writes a different
distribution from the same model ANSWERING it, and the distribution of AI text is what
this project detects. A corpus built the first way and labelled as four instruction-tuned
families would have been mislabelled in the one dimension that matters.
"""

from __future__ import annotations

import pytest

from forge.generation.generators.base import (
    Decoding,
    NoChatTemplateError,
    VLLMGenerator,
    to_chat_prompt,
    to_chat_prompts,
)

PINNED = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


class FakeTokenizer:
    """Stands in for a real tokenizer's template behaviour, not its tokenisation."""

    name_or_path = "fake/Instruct"
    chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False, "the backend needs the rendered string, not token ids"
        assert add_generation_prompt is True, (
            "without the generation prompt the model is still being asked to continue"
        )
        body = "".join(m["content"] for m in messages)
        return f"<|user|>{body}<|assistant|>"


class BaseModelTokenizer:
    name_or_path = "fake/Base"
    chat_template = None


def test_the_instruction_is_wrapped_in_the_models_own_template():
    out = to_chat_prompt("Write an essay about tides.", FakeTokenizer())
    assert out == "<|user|>Write an essay about tides.<|assistant|>"


def test_the_instruction_survives_wrapping_unaltered():
    """Wrapping must not paraphrase, truncate or reorder the prompt."""
    prompt = "Write a {genre} in a {register} register about {topic}."
    assert prompt in to_chat_prompt(prompt, FakeTokenizer())


def test_the_generation_prompt_is_requested():
    """add_generation_prompt=False leaves the model continuing the user turn.

    That is the same failure in a subtler costume, so the fake asserts on it and this
    test exists to say why the assertion is there.
    """
    assert to_chat_prompt("x", FakeTokenizer()).endswith("<|assistant|>")


def test_a_model_without_a_template_raises_instead_of_falling_back():
    """The silent fallback IS the bug. It must never be the default again."""
    with pytest.raises(NoChatTemplateError, match="no chat template"):
        to_chat_prompt("Write an essay.", BaseModelTokenizer())


def test_every_prompt_in_a_batch_is_wrapped():
    outs = to_chat_prompts(["a", "b", "c"], FakeTokenizer())
    assert len(outs) == 3
    assert all(o.startswith("<|user|>") for o in outs)


class _Recorder:
    """Captures what the engine was actually handed."""

    def __init__(self):
        self.seen: list[str] = []

    def generate(self, prompts, params, use_tqdm=False):
        self.seen = list(prompts)
        return [type("O", (), {"outputs": [type("T", (), {"text": "ok"})()]})() for _ in prompts]


def _wired(monkeypatch):
    gen = VLLMGenerator("qwen", "Qwen/Qwen2.5-3B-Instruct", PINNED)
    engine = _Recorder()
    monkeypatch.setattr(gen, "_load", lambda: engine)
    monkeypatch.setattr(gen, "_tokenizer", lambda: FakeTokenizer())
    monkeypatch.setattr(gen, "_params", lambda d: object())
    return gen, engine


def test_the_vllm_batch_path_sends_chat_prompts_not_raw_ones(monkeypatch):
    """generate_many is the path the real runs take. It was sending raw prompts."""
    gen, engine = _wired(monkeypatch)
    d = Decoding(0.7, 0.95, 256, 1)
    gen.generate_many(["Write an essay about tides."], [d])
    assert engine.seen == ["<|user|>Write an essay about tides.<|assistant|>"]


def test_the_vllm_single_decoding_path_sends_chat_prompts_too(monkeypatch):
    """Two paths, one mistake each; fixing only the batch one leaves a live copy."""
    gen, engine = _wired(monkeypatch)
    gen.generate(["Write an essay about tides."], Decoding(0.7, 0.95, 256, 1))
    assert engine.seen == ["<|user|>Write an essay about tides.<|assistant|>"]


def test_closing_drops_the_tokenizer_with_the_engine(monkeypatch):
    """Weights are purged between families on a small disk; the tokenizer goes too."""
    gen = VLLMGenerator("qwen", "Qwen/Qwen2.5-3B-Instruct", PINNED)
    gen._tok = FakeTokenizer()
    gen.close()
    assert gen._tok is None
