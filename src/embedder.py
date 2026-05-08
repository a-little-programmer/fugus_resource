"""Embedding backends.

The production-oriented backend is sentence-transformers. A deterministic
standard-library fallback keeps the prototype runnable in constrained
environments and in tests.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Protocol

from .normalization import normalize_text


DEFAULT_FINE_TUNED_MODEL = "models/fugus-entity-embedding"
DEFAULT_BASE_MODEL = "BAAI/bge-small-zh-v1.5"


class Embedder(Protocol):
    backend_name: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    backend_name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


class CharNgramEmbedder:
    backend_name = "char-ngram-fallback"

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        normalized = normalize_text(text)
        features = Counter[str]()
        compact = normalized.replace(" ", "")
        tokens = normalized.split()

        for token in tokens:
            features[f"tok:{token}"] += 2
        for n in (1, 2, 3, 4):
            for i in range(max(0, len(compact) - n + 1)):
                features[f"ng{n}:{compact[i:i+n]}"] += 1

        vector = [0.0] * self.dimensions
        for feature, value in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] += float(value)
        return normalize_vector(vector)


def resolve_model_name(model_name: str | None) -> str:
    requested = model_name or DEFAULT_FINE_TUNED_MODEL
    if requested == DEFAULT_FINE_TUNED_MODEL and not Path(requested).exists():
        logging.warning(
            "未找到微调模型 %s，正在使用预训练 Base 模型 %s，检索效果可能受限。",
            DEFAULT_FINE_TUNED_MODEL,
            DEFAULT_BASE_MODEL,
        )
        return DEFAULT_BASE_MODEL
    return requested


def create_embedder(model_name: str | None = None, backend: str = "auto") -> Embedder:
    if backend == "char-ngram":
        return CharNgramEmbedder()
    if backend not in {"auto", "sentence-transformers"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")

    resolved_model_name = resolve_model_name(model_name)
    try:
        return SentenceTransformerEmbedder(resolved_model_name)
    except Exception as exc:
        if backend == "sentence-transformers":
            raise
        logging.warning(
            "sentence-transformers backend unavailable; using char-ngram fallback. Reason: %s",
            exc,
        )
        return CharNgramEmbedder()


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
