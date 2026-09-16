"""Length and quality filtering. Stages 5 and 6.

The quality heuristics are the standard web-corpus ones and they exist for a specific
reason in this project: navigation boilerplate, link farms and template text are the
human documents most likely to look machine-generated to a detector, because they are
repetitive and low-entropy. Letting them through inflates the false-positive rate for
reasons that have nothing to do with authorship.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"\b\w+\b", re.UNICODE)
_ALNUM = re.compile(r"[\w\s]", re.UNICODE)


@dataclass(frozen=True)
class LengthPolicy:
    min_chars: int = 200
    min_tokens: int = 50
    max_tokens: int = 20_000


@dataclass(frozen=True)
class QualityPolicy:
    max_symbol_ratio: float = 0.20        # non-alphanumeric, non-space share
    max_repeated_line_ratio: float = 0.30  # share of lines that are duplicates
    min_mean_word_length: float = 2.5
    max_mean_word_length: float = 12.0
    min_unique_word_ratio: float = 0.20    # type/token ratio floor


def approx_token_count(text: str) -> int:
    """Whitespace-word count as a tokenizer-free proxy.

    Real subword counts run about 1.3x this for English. That factor is irrelevant to
    filtering, and depending on a tokenizer here would make ingestion require torch.
    """
    return len(text.split())


def symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    non_alnum = sum(1 for ch in text if not _ALNUM.match(ch))
    return non_alnum / len(text)


def repeated_line_ratio(text: str) -> float:
    lines = [ln for ln in (raw.strip() for raw in text.split("\n")) if ln]
    if len(lines) < 2:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def mean_word_length(text: str) -> float:
    words = _WORD.findall(text)
    return sum(len(w) for w in words) / len(words) if words else 0.0


def unique_word_ratio(text: str) -> float:
    words = _WORD.findall(text.lower())
    return len(set(words)) / len(words) if words else 0.0


def check_length(text: str, policy: LengthPolicy = LengthPolicy()) -> list[str]:
    reasons: list[str] = []
    if len(text) < policy.min_chars:
        reasons.append("too_short_chars")
    n = approx_token_count(text)
    if n < policy.min_tokens:
        reasons.append("too_short_tokens")
    if n > policy.max_tokens:
        reasons.append("too_long_tokens")
    return reasons


def check_quality(text: str, policy: QualityPolicy = QualityPolicy()) -> list[str]:
    reasons: list[str] = []
    if symbol_ratio(text) > policy.max_symbol_ratio:
        reasons.append("high_symbol_ratio")
    if repeated_line_ratio(text) > policy.max_repeated_line_ratio:
        reasons.append("repetitive_lines")
    mwl = mean_word_length(text)
    if not policy.min_mean_word_length <= mwl <= policy.max_mean_word_length:
        reasons.append("anomalous_word_length")
    if unique_word_ratio(text) < policy.min_unique_word_ratio:
        reasons.append("low_lexical_diversity")
    return reasons
