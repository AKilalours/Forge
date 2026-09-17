"""An error that names the wrong cause is worse than one that names none.

Training died with "non-finite loss at step 1. With DeBERTa-v3 this is usually fp16
overflow in disentangled attention; use bf16 or fp32." on a run whose config said
precision: bf16, whose code honoured it, and on a GPU that supports it. bf16 carries
fp32's exponent range and does not overflow the way fp16 does, so the hint was confidently
wrong and pointed at the one setting that was already correct.

The real mechanism is in the model: CrossEntropyLoss(ignore_index=-100) reduces by mean,
so a batch in which EVERY token label is ignored divides by zero and returns NaN silently.
Token labels are built from character spans through the tokenizer's offset mapping, which
is the kind of thing a transformers major-version bump changes underneath you.

Two fixes, both here. The model refuses the degenerate batch where the condition is known
and says what it is. The training loop reports the precision actually in use and offers
the fp16 hypothesis only when fp16 is actually running.
"""

from __future__ import annotations

import pathlib
import re

import pytest

TRAIN = pathlib.Path(__file__).resolve().parents[2] / "src" / "forge" / "training" / "train.py"
ENCODER = pathlib.Path(__file__).resolve().parents[2] / "src" / "forge" / "modeling" / "encoder.py"


def test_the_fp16_hint_is_conditional_on_fp16_actually_running():
    body = TRAIN.read_text()
    block = body[body.index("if not torch.isfinite(loss):"):]
    block = block[:block.index("raise RuntimeError") + 400]
    assert 'precision == "fp16"' in block, (
        "the fp16 diagnosis is still offered unconditionally"
    )


def test_the_non_finite_error_reports_the_precision_in_use():
    """The reader should not have to go and look up what the run was actually doing."""
    body = TRAIN.read_text()
    assert "precision={precision}" in body


def test_the_model_refuses_an_all_ignored_token_label_batch():
    body = ENCODER.read_text()
    assert "token_labels != -100" in body
    assert "ignore_index" in body
    assert "NaN" in body


def test_the_refusal_tells_the_reader_where_token_labels_come_from():
    """The actionable part: spans, offset mapping, tokenizer version."""
    body = ENCODER.read_text()
    start = body.index("every token label in this batch")
    msg = body[start:start + 900]
    for word in ("offset mapping", "tokenizer", "token_loss_weight"):
        assert word in msg, f"the message does not mention {word!r}"


def test_cross_entropy_over_zero_elements_really_is_nan():
    """The premise, verified rather than asserted from memory.

    If a future torch returns 0.0 here instead of NaN, the guard above becomes dead code
    and this test is where that gets noticed.
    """
    torch = pytest.importorskip("torch")
    from torch import nn

    logits = torch.randn(4, 3)
    labels = torch.full((4,), -100, dtype=torch.long)
    loss = nn.CrossEntropyLoss(ignore_index=-100)(logits, labels)
    assert torch.isnan(loss), "the zero-denominator premise no longer holds"


def test_a_normal_batch_is_unaffected():
    torch = pytest.importorskip("torch")
    from torch import nn

    logits = torch.randn(4, 3)
    labels = torch.tensor([0, -100, 2, 1])
    loss = nn.CrossEntropyLoss(ignore_index=-100)(logits, labels)
    assert torch.isfinite(loss)


def test_no_unconditional_precision_claim_survives_in_the_message():
    """Belt and braces: the old sentence must not still be sitting there."""
    body = TRAIN.read_text()
    assert not re.search(r"this is\s+\"?\s*\"?usually fp16 overflow", body)
