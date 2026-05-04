from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")


class EmbeddingModel:
    dimensions: int

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class HashEmbeddingModel(EmbeddingModel):
    """Deterministic local embeddings for tests and offline demos."""

    dimensions: int = 256

    def embed(self, text: str) -> list[float]:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

        vector = [0.0 for _ in range(self.dimensions)]
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return normalize(vector)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right, strict=True))

