"""Near-duplicate detection: MinHash with LSH banding. Stage 9.

Written directly rather than pulling in datasketch, for two reasons. The dependency is
avoidable at this scale, and more importantly the banding parameters are a real
decision in this project that should be visible in the repo rather than hidden behind
a default.

How it works, concretely. Each document becomes a set of 5-word shingles. Hashing every
shingle 128 different ways and keeping the minimum of each gives a 128-number
signature; the fraction of positions where two signatures agree estimates their Jaccard
similarity. Comparing every pair would be O(n^2), so the signature is cut into `bands`
chunks and documents sharing any identical chunk become candidates. Two documents that
are 80 percent similar almost always collide in at least one band; two that are 30
percent similar almost never do.

Threshold 0.8 with 128 permutations gives bands=16, rows=8, whose S-curve crosses 0.5
probability at about (1/16)^(1/8) = 0.70 similarity. That is deliberately a little
below 0.8 so the cheap candidate stage over-collects and the exact Jaccard check
decides. Missing a near-duplicate is worse than paying for an extra comparison,
because a missed one can straddle train and test.

This is also used in Phase 2, to reject a generated mirror that came out too close to
its human source.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_WORD = re.compile(r"\b\w+\b", re.UNICODE)
_MERSENNE = (1 << 61) - 1
_MAX32 = (1 << 32) - 1


def shingles(text: str, k: int = 5) -> set[int]:
    """Hashed k-word shingles. Word-level, not character-level: character shingles make
    every English document look similar because English letter statistics dominate."""
    words = _WORD.findall(text.lower())
    if len(words) < k:
        return {int.from_bytes(hashlib.sha1(" ".join(words).encode()).digest()[:4], "big")} if words else set()
    out = set()
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i : i + k]).encode()
        out.add(int.from_bytes(hashlib.sha1(gram).digest()[:4], "big"))
    return out


def _permutations(n: int, seed: int = 42) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficients for h(x) = (a*x + b) mod prime."""
    rng = _Lcg(seed)
    return [((rng.next() % (_MERSENNE - 1)) + 1, rng.next() % _MERSENNE) for _ in range(n)]


class _Lcg:
    """Tiny deterministic PRNG so signatures are reproducible without numpy's global state."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) % (1 << 64)
        return self.state


@dataclass(frozen=True)
class LshParams:
    num_perm: int = 128
    bands: int = 16
    threshold: float = 0.8
    shingle_k: int = 5

    @property
    def rows(self) -> int:
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be divisible by bands")
        return self.num_perm // self.bands


class MinHash:
    def __init__(self, params: LshParams = LshParams()) -> None:
        self.params = params
        self._perms = _permutations(params.num_perm)

    def signature(self, text: str) -> tuple[int, ...]:
        sh = shingles(text, self.params.shingle_k)
        if not sh:
            return tuple([_MAX32] * self.params.num_perm)
        return tuple(min(((a * x + b) % _MERSENNE) & _MAX32 for x in sh) for a, b in self._perms)


def estimated_jaccard(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if len(sig_a) != len(sig_b):
        raise ValueError("signatures must have the same length")
    return sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b) / len(sig_a)


class MinHashLSH:
    """Banded index. `add` then `query`, or use `add_if_new` for a streaming dedup pass."""

    def __init__(self, params: LshParams = LshParams()) -> None:
        self.params = params
        self._hasher = MinHash(params)
        self._buckets: list[dict[tuple[int, ...], set[str]]] = [
            {} for _ in range(params.bands)
        ]
        self._sigs: dict[str, tuple[int, ...]] = {}

    def _bands_of(self, sig: tuple[int, ...]):
        r = self.params.rows
        for i in range(self.params.bands):
            yield i, sig[i * r : (i + 1) * r]

    def add(self, doc_id: str, text: str) -> tuple[int, ...]:
        sig = self._hasher.signature(text)
        self._sigs[doc_id] = sig
        for i, band in self._bands_of(sig):
            self._buckets[i].setdefault(band, set()).add(doc_id)
        return sig

    def query(self, text: str) -> list[tuple[str, float]]:
        """Return (doc_id, estimated_jaccard) above threshold, most similar first."""
        sig = self._hasher.signature(text)
        candidates: set[str] = set()
        for i, band in self._bands_of(sig):
            candidates |= self._buckets[i].get(band, set())
        scored = [(d, estimated_jaccard(sig, self._sigs[d])) for d in candidates]
        hits = [(d, s) for d, s in scored if s >= self.params.threshold]
        return sorted(hits, key=lambda t: -t[1])

    def add_if_new(self, doc_id: str, text: str) -> str | None:
        """Streaming dedup. Returns the doc_id of a near-duplicate, or None after adding."""
        hits = self.query(text)
        if hits:
            return hits[0][0]
        self.add(doc_id, text)
        return None

    def __len__(self) -> int:
        return len(self._sigs)
