"""The adversarial laboratory.

Ours are controlled so a degradation can be attributed to one specific transformation.
RAID's twelve attacks are the independent external check, because attacks we wrote
ourselves are attacks we implicitly designed around.

--------------------------------------------------------------------------------
The thing that makes this measurement subtle
--------------------------------------------------------------------------------
FORGE's own ingestion pipeline NEUTRALISES several of these attacks before the model ever
sees the text. `normalize()` applies NFKC, strips zero-width characters and collapses
whitespace runs. So homoglyph, zero-width and whitespace attacks are largely defused by
preprocessing, not by the model.

That means a single robustness number is meaningless, and it can be wrong in either
direction:

  measure WITH preprocessing only  -> overstates the model's robustness. The credit
                                      belongs to two lines of `normalize()`, and any
                                      deployment that skips them is unprotected.
  measure WITHOUT preprocessing    -> overstates vulnerability, and reports a threat the
                                      production path already handles.

So every attack is evaluated in both conditions and the report shows both columns. The
gap between them is the value of the preprocessing defence, stated as a number instead of
assumed. See `Attack.defused_by_preprocessing`.

--------------------------------------------------------------------------------
Validity
--------------------------------------------------------------------------------
An attack that mangles text until a human cannot read it is not an evasion, it is
vandalism, and counting it inflates the reported vulnerability. Every attack carries a
`preserves_meaning` check and the runner records attacks that failed it rather than
scoring them.

All attacks are deterministic, seeded per document, so a robustness table can be
regenerated exactly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from forge.cleaning.normalize import normalize
from forge.dedup.minhash import MinHash, estimated_jaccard

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*\b")

# Latin characters with visually identical Cyrillic/Greek counterparts.
HOMOGLYPHS = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р",
    "x": "х", "y": "у", "A": "А", "B": "В", "C": "С",
    "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О",
    "P": "Р", "T": "Т", "X": "Х",
}
ZERO_WIDTH = "​"

# Small, deliberately boring. A real synonym attack uses a lexicon or a model; this is
# enough to measure the mechanism and it never changes meaning.
SYNONYMS = {
    "big": "large", "small": "little", "begin": "start", "end": "finish",
    "show": "display", "use": "employ", "help": "assist", "need": "require",
    "make": "create", "get": "obtain", "many": "numerous", "important": "significant",
    "however": "nevertheless", "because": "since", "about": "regarding",
    "often": "frequently", "quickly": "rapidly", "difficult": "challenging",
}


def _u(seed_key: str, i: int) -> float:
    """Deterministic uniform in [0,1) for position i.

    Severity means "probability that each ELIGIBLE position is perturbed", and this is
    what makes that true. The earlier implementation combined `i % step == 0` with a
    random bit, which halves the rate at best and drops it to zero whenever the eligible
    positions are sparse. Measured on a four-sentence fixture, synonym_swap and
    article_deletion perturbed NOTHING at their configured severities, and an attack that
    does nothing reports as "the model is robust to it".
    """
    h = hashlib.blake2b(f"{seed_key}|{i}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


def _fires(seed_key: str, i: int, severity: float) -> bool:
    return _u(seed_key, i) < severity


class AttackFn(Protocol):
    def __call__(self, text: str, seed_key: str, severity: float) -> str: ...


@dataclass(frozen=True)
class Attack:
    name: str
    fn: AttackFn
    severities: tuple[float, ...]
    # True when FORGE's own ingestion removes the perturbation before the model sees it.
    defused_by_preprocessing: bool
    description: str

    def apply(self, text: str, seed_key: str, severity: float) -> str:
        return self.fn(text=text, seed_key=seed_key, severity=severity)


# ------------------------------------------------------------------ transformations

def whitespace_perturb(text: str, seed_key: str, severity: float) -> str:
    """Insert extra spaces. Defused by collapse_whitespace() in normalize()."""
    out = []
    for i, ch in enumerate(text):
        out.append(ch)
        if ch == " " and _fires(seed_key, i, severity):
            out.append(" ")
    return "".join(out)


def zero_width_insert(text: str, seed_key: str, severity: float) -> str:
    """Insert U+200B between characters. Defused by strip_invisible() in normalize()."""
    return "".join(
        ch + (ZERO_WIDTH if _fires(seed_key, i, severity) else "") for i, ch in enumerate(text)
    )


def homoglyph_substitute(text: str, seed_key: str, severity: float) -> str:
    """Swap Latin letters for identical-looking Cyrillic. NOT defused by NFKC.

    A common misconception worth stating: NFKC normalises compatibility characters, but
    Cyrillic small a (U+0430) is a distinct letter, not a compatibility form of Latin a.
    NFKC leaves it exactly as it is. This attack survives FORGE's preprocessing, which is
    why it is flagged defused_by_preprocessing=False and is one to take seriously.
    """
    out = []
    for i, ch in enumerate(text):
        out.append(HOMOGLYPHS[ch] if (ch in HOMOGLYPHS and _fires(seed_key, i, severity)) else ch)
    return "".join(out)


def synonym_swap(text: str, seed_key: str, severity: float) -> str:
    """Severity applies to ELIGIBLE words (those with a synonym), not to all words.

    Applying it to all words would make the effective rate depend on how many synonyms
    the lexicon happens to contain, so "severity 0.15" would mean something different for
    every document.
    """
    eligible = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal eligible
        w = m.group()
        low = w.lower()
        if low not in SYNONYMS:
            return w
        fire = _fires(seed_key, eligible, severity)
        eligible += 1
        if not fire:
            return w
        syn = SYNONYMS[low]
        return syn.capitalize() if w[0].isupper() else syn

    return _WORD.sub(repl, text)


def sentence_reorder(text: str, seed_key: str, severity: float) -> str:
    """Swap adjacent sentence pairs. Preserves every sentence, changes only order."""
    parts = _SENT_SPLIT.split(text)
    if len(parts) < 2:
        return text
    out = list(parts)
    n_swaps = max(int(len(parts) * severity), 1)
    for k in range(n_swaps):
        i = int(_u(seed_key, 1000 + k) * (len(out) - 1))
        out[i], out[i + 1] = out[i + 1], out[i]
    return " ".join(out)


def case_perturb(text: str, seed_key: str, severity: float) -> str:
    return "".join(
        (ch.upper() if ch.islower() else ch.lower()) if _fires(seed_key, i, severity) else ch
        for i, ch in enumerate(text)
    )


def article_deletion(text: str, seed_key: str, severity: float) -> str:
    """Drop 'the'/'a'/'an'. One of RAID's attacks; degrades fluency slightly."""
    eligible = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal eligible
        if m.group().lower() not in {"the", "a", "an"}:
            return m.group()
        fire = _fires(seed_key, eligible, severity)
        eligible += 1
        return "" if fire else m.group()

    return re.sub(r"\s{2,}", " ", _WORD.sub(repl, text)).strip()


def paragraph_insert(text: str, seed_key: str, severity: float) -> str:
    """Insert paragraph breaks. Defused: normalize() collapses runs of newlines."""
    parts = _SENT_SPLIT.split(text)
    if len(parts) < 2:
        return text
    return " ".join(p + ("\n\n" if _fires(seed_key, i, severity) else "") for i, p in enumerate(parts))


class ModelAttack:
    """Paraphrase, humaniser and AI-assisted edit all need a model.

    Deliberately not faked. A stub that shuffles words and calls itself a paraphraser
    would produce a robustness number that means nothing, and a paraphrase attack is the
    single most important one to measure honestly, because it is what commercial
    humaniser tools actually do.
    """

    def __init__(self, name: str, model_id: str | None = None) -> None:
        self.name, self.model_id = name, model_id

    def __call__(self, text: str, seed_key: str, severity: float) -> str:
        raise NotImplementedError(
            f"{self.name} needs a paraphrase model. Wire one in Phase 6 deployment; a "
            "faked paraphraser would produce a meaningless robustness number."
        )


# ------------------------------------------------------------------ registry

ATTACKS: dict[str, Attack] = {
    a.name: a
    for a in [
        Attack("whitespace_perturb", whitespace_perturb, (0.02, 0.10), True,
               "extra spaces; removed by collapse_whitespace()"),
        Attack("zero_width_insert", zero_width_insert, (0.01, 0.05), True,
               "U+200B insertion; removed by strip_invisible()"),
        # Measured, not assumed: normalize() collapses runs of THREE OR MORE newlines to
        # two, so a single inserted blank line survives preprocessing untouched.
        Attack("paragraph_insert", paragraph_insert, (0.2, 0.5), False,
               "spurious paragraph breaks; survives normalize(), which only collapses 3+ newlines"),
        Attack("homoglyph_substitute", homoglyph_substitute, (0.01, 0.05, 0.20), False,
               "Cyrillic look-alikes; NFKC does NOT remove these"),
        Attack("synonym_swap", synonym_swap, (0.05, 0.15), False,
               "lexical substitution preserving meaning"),
        Attack("sentence_reorder", sentence_reorder, (0.1, 0.3), False,
               "adjacent sentence swaps; content identical, order changed"),
        Attack("case_perturb", case_perturb, (0.02, 0.10), False,
               "random case flips; survives normalisation"),
        Attack("article_deletion", article_deletion, (0.2, 0.5), False,
               "drops the/a/an; mild fluency loss"),
        Attack("paraphrase_llm", ModelAttack("paraphrase_llm"), (1.0,), False,
               "model paraphrase; the attack that matters most, needs a model"),
        Attack("humanizer_tool", ModelAttack("humanizer_tool"), (1.0,), False,
               "commercial humaniser; needs a real service"),
        Attack("ai_assisted_edit", ModelAttack("ai_assisted_edit"), (1.0,), False,
               "human text partially rewritten by a model; needs a model"),
        Attack("grammar_edit", ModelAttack("grammar_edit"), (1.0,), False,
               "grammar and fluency correction; needs a model, and note this is the "
               "attack most likely to be applied innocently by a real user"),
    ]
}

RUNNABLE_OFFLINE = tuple(
    name for name, a in ATTACKS.items() if not isinstance(a.fn, ModelAttack)
)


REVERSE_HOMOGLYPHS = {v: k for k, v in HOMOGLYPHS.items()}


def fold_homoglyphs(text: str) -> str:
    """Map look-alike characters back to Latin, for validity checking only.

    Never used on training or inference input. It exists solely so preserves_meaning()
    asks the right question.
    """
    return "".join(REVERSE_HOMOGLYPHS.get(ch, ch) for ch in text)


def preserves_meaning(original: str, attacked: str, min_jaccard: float = 0.5) -> bool:
    """Would a human read the same thing? An attack that mangles text past readability is
    vandalism, not evasion, and counting it inflates the reported vulnerability.

    Homoglyphs are folded before comparison. Without that fold, a token-overlap check
    systematically misjudges every character-level attack as invalid: Cyrillic small a
    looks identical to a reader but breaks every word containing it, so a perfectly
    readable homoglyph attack scores a Jaccard near zero. Discarding those would throw
    away precisely the attacks that work best, and would report a flattering robustness
    number built on a filtered-out threat.

    Compared after normalisation too, so an attack whose entire effect is removed by
    preprocessing still reads as meaning-preserving, which it is.
    """
    m = MinHash()
    a = normalize(fold_homoglyphs(original))
    b = normalize(fold_homoglyphs(attacked))
    if not a or not b:
        return False
    return estimated_jaccard(m.signature(a), m.signature(b)) >= min_jaccard


def is_noop(original: str, attacked: str) -> bool:
    """Did the attack actually change anything?

    A no-op scores as perfect robustness. Sparse-target attacks such as synonym_swap on a
    document containing few substitutable words genuinely produce no change at low
    severity, and a robustness table that counts those as successful defences is
    reporting the lexicon's coverage as the model's strength.
    """
    return original == attacked


def apply_attack(text: str, attack: str, seed_key: str, severity: float) -> str:
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; known: {sorted(ATTACKS)}")
    return ATTACKS[attack].apply(text, seed_key, severity)


def survives_preprocessing(text: str, attack: str, seed_key: str, severity: float) -> bool:
    """Does the perturbation still exist after FORGE's ingestion normalisation?

    This is the empirical version of `Attack.defused_by_preprocessing`, and the two are
    cross-checked in the tests so the flag cannot drift away from the truth.
    """
    attacked = apply_attack(text, attack, seed_key, severity)
    return normalize(attacked) != normalize(text)
