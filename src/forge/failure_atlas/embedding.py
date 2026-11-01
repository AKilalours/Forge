"""Embedders for the Failure Atlas.

Pluggable, with a deterministic offline implementation, for the same reason the mirror
engine has a fake generator: the clustering and selection logic has to be verifiable
without downloading a model, and a bug in selection is far more damaging than a slightly
worse embedding.

HashingEmbedder is a real technique (the hashing trick over character n-grams), not a
stub. It captures surface style, which is a genuine part of what makes a document look
machine-written, and it is what the tests use. It does NOT capture semantics, so it is
not what a real run should use. SentenceTransformerEmbedder is.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np

_WORD = re.compile(r"\b\w+\b", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Character n-gram hashing. Deterministic, dependency-free, no download."""

    name = "hashing_v1"

    def __init__(self, dim: int = 256, ngram: int = 4) -> None:
        self.dim = dim
        self.ngram = ngram

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            s = t.lower()
            for j in range(max(len(s) - self.ngram + 1, 0)):
                gram = s[j : j + self.ngram]
                h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
                # signed hashing: the sign bit keeps collisions from all adding up
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - needs a download
    """What a real mining run should use. Semantics, not just surface form."""

    name = "sentence_transformers"

    def __init__(self, model_id: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.model_id = model_id
        self._m: Any = None
        self.dim = 768

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            self._m = SentenceTransformer(self.model_id)
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float64)
