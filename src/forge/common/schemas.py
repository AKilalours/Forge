"""Canonical record schemas for FORGE.

These are the contract defined in docs/data_spec_v1.md section 3. If you change a
field here, you must change the spec and bump its version. tests/unit/test_schemas.py
enforces the parts that the pipeline silently depends on.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Label(str, Enum):
    HUMAN = "human"
    AI = "ai"


class TokenLabel(str, Enum):
    """Exactly three, in this order. The token head's output dimension is len(TokenLabel)."""

    HUMAN = "human"
    AI_ASSISTED = "ai_assisted"
    AI_GENERATED = "ai_generated"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Quality(BaseModel):
    edu_score: float | None = None
    length_ok: bool = True
    pii_flags: list[str] = Field(default_factory=list)


class HumanDocument(BaseModel):
    # `register` shadows a BaseModel attribute, so the python field is text_register
    # while the on-disk/JSON key stays "register" as written in data_spec_v1.
    model_config = ConfigDict(populate_by_name=True)

    doc_id: str
    source_group_id: str
    text: str
    source: str
    source_config: str | None = None
    source_record_id: str | None = None
    license: str
    domain: str
    text_register: str = Field(alias="register")
    language: str = "en"
    language_score: float
    date: str | None = None
    acquired_at: datetime
    processing_version: str
    content_sha256: str
    token_count: int
    quality: Quality = Field(default_factory=Quality)
    split: Split
    redistributable: bool = False

    @field_validator("content_sha256")
    @classmethod
    def _sha_shape(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError("content_sha256 must be a 64-char hex digest")
        try:
            bytes.fromhex(v)
        except ValueError as error:
            raise ValueError("content_sha256 must be a 64-char hex digest") from error
        return v


class GeneratorSpec(BaseModel):
    provider: Literal["open_source", "api"]
    family: str
    model_id: str
    # Mandatory. "Qwen 7B" is not reproducible; a repo revision is.
    revision: str
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int | None = None


class MirrorSpec(BaseModel):
    prompt_version: str
    target_tokens: int
    topic_match: bool
    length_match: bool
    style_match: bool
    attributes: dict[str, str] = Field(default_factory=dict)


class SyntheticDocument(BaseModel):
    sample_id: str
    source_human_id: str
    source_group_id: str
    label: Label = Label.AI
    text: str
    generator: GeneratorSpec
    mirror: MirrorSpec
    domain: str
    language: str = "en"
    transformations: list[str] = Field(default_factory=list)
    generated_at: datetime
    license: str = "synthetic"
    split: Split


class TokenLabelSpan(BaseModel):
    """Character spans, not token indices. Token indices break when the tokenizer changes."""

    start_char: int
    end_char: int
    label: TokenLabel

    @field_validator("end_char")
    @classmethod
    def _ordered(cls, v: int, info):  # type: ignore[no-untyped-def]
        start = info.data.get("start_char")
        if start is not None and v <= start:
            raise ValueError("end_char must be greater than start_char")
        return v


class FailureRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    sample_id: str
    true_label: Label
    prediction: Label
    confidence: float
    domain: str
    source: str
    text_register: str | None = Field(default=None, alias="register")
    embedding_id: str | None = None
    cluster: int | None = None
    model_version: str
    failure_type: Literal[
        "human_false_positive", "ai_false_negative", "adversarial_evasion", "drift"
    ]
    discovered_at: datetime
    discovered_by: str
