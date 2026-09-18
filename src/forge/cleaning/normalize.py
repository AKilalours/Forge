"""Text normalization. Stages 3 and 4 of the spec pipeline.

Order inside this file matters: markup is stripped before whitespace is collapsed,
because stripping tags leaves ragged whitespace behind that the collapse then cleans
up. Doing it the other way round leaves the ragged whitespace in the output.

ftfy is used when installed but is not required. Mojibake repair improves quality; it
is not worth making the whole Phase 1 pipeline undeployable over.
"""

from __future__ import annotations

import re
import unicodedata

try:  # optional
    from ftfy import fix_text as _ftfy_fix
except ImportError:  # pragma: no cover - exercised only when ftfy is absent
    _ftfy_fix = None

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_WS_RUN = re.compile(r"[ \t  - ]+")
_NL_RUN = re.compile(r"\n{3,}")

# Invisible characters. These are also an adversarial attack vector (RAID's
# zero-width-space attack), so stripping them at ingestion means the detector never
# learns to rely on their presence in human text.
_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E]
)

# CONFUSABLES, and why they are NOT folded by default.
#
# NFKC normalises compatibility characters. Cyrillic small a (U+0430) is not a
# compatibility form of Latin a, it is a different letter, so NFKC leaves it alone and the
# homoglyph attack survives FORGE's whole preprocessing path. The adversarial lab measured
# that at severity 0.2: FNR 0.000 clean against 0.306 attacked, and 0.308 after
# normalisation, a preprocessing benefit of -0.002. The defence simply is not there.
#
# Folding is the fix, in the Unicode TR39 sense: map each confusable to the Latin letter it
# imitates before the text reaches the tokenizer. It is deliberately OPT IN, for two
# reasons that both outrank the attack.
#
# 1. The ingestion contract is frozen in docs/data_spec_v1.md, and normalize() is stage 3
#    and 4 of it. Changing what it emits by default changes every document in the corpus.
# 2. The deployed checkpoints were TRAINED on unfolded text and their thresholds were
#    fitted under that tokenisation. Folding only at inference moves every document away
#    from the training distribution, which can cost more on clean text than it buys on
#    attacked text. That is a measurement, not an opinion, which is why the adversarial lab
#    now scores a folded condition WITH ITS OWN CLEAN BASELINE rather than reusing the
#    unfolded one. A defence whose clean column moves is not free.
#
# This table is the inverse of forge.adversarial.attacks.HOMOGLYPHS plus the Greek and
# Cyrillic letters that table does not use. It is intentionally letters only: folding
# punctuation and digits belongs to NFKC, which already does it.
_CONFUSABLES = {
    # Cyrillic to Latin
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "к": "k", "м": "m",
    "о": "o", "р": "p", "ѕ": "s", "т": "t", "х": "x", "у": "y", "і": "i",
    "ј": "j", "ԁ": "d", "ѡ": "w", "ց": "g",
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Ѕ": "S", "Т": "T", "Х": "X", "У": "Y", "І": "I",
    "Ј": "J", "Ԁ": "D", "Ѡ": "W", "Г": "r", "Ь": "b", "Я": "R", "Ф": "O",
    # Greek to Latin
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "χ": "x",
    "ι": "i", "κ": "k", "μ": "u", "ε": "e",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}
_CONFUSABLE_TABLE = str.maketrans(_CONFUSABLES)

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
}


def strip_markup(text: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    for ent, rep in _HTML_ENTITIES.items():
        text = text.replace(ent, rep)
    return _ENTITY.sub(" ", text)


def strip_invisible(text: str) -> str:
    return text.translate(_INVISIBLE)


def fold_confusable_letters(text: str) -> str:
    """Map Cyrillic and Greek lookalikes to the Latin letters they imitate.

    Lossy on purpose, and only defensible because this corpus is English: the language
    filter in Phase 1 keeps English documents, so a Cyrillic letter inside one is far more
    likely to be an attack or an encoding accident than content. On a multilingual corpus
    this transform would be vandalism, which is why it is a named stage and not folded into
    NFKC's job.

    Idempotent: the Latin output contains nothing left to fold.
    """
    return text.translate(_CONFUSABLE_TABLE)


def collapse_whitespace(text: str) -> str:
    text = _WS_RUN.sub(" ", text)
    text = _NL_RUN.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def normalize(text: str, *, use_ftfy: bool = True,
              fold_confusables: bool = False) -> str:
    """Full normalization. Deterministic and idempotent: normalize(normalize(x)) == normalize(x).

    fold_confusables defaults to False. See the comment above _CONFUSABLES: turning it on
    changes the frozen ingestion contract and moves every document away from the
    distribution the deployed checkpoints were trained on. It is a candidate defence to be
    measured, not a default to be assumed.

    Folding runs AFTER NFKC and before markup stripping. After NFKC, because NFKC can
    itself produce a letter worth folding; before the whitespace collapse, because folding
    never introduces whitespace and the collapse must be last for the idempotence above.
    """
    if use_ftfy and _ftfy_fix is not None:
        text = _ftfy_fix(text)
    text = unicodedata.normalize("NFKC", text)
    if fold_confusables:
        text = fold_confusable_letters(text)
    text = strip_invisible(text)
    text = strip_markup(text)
    return collapse_whitespace(text)
