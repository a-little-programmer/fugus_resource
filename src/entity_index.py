"""Multi-vector entity index with max-score aggregation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alias_matcher import entity_names
from .embedder import Embedder, cosine_similarity


@dataclass(frozen=True)
class SearchResult:
    entity: dict[str, Any]
    source: str
    matched_name: str
    score: float
    low_confidence: bool
    ambiguous: bool = False


class EntityIndex:
    def __init__(
        self,
        entities: list[dict[str, Any]],
        records: list[dict[str, Any]],
        backend_name: str,
        model_name: str,
    ) -> None:
        self.entities = entities
        self.records = records
        self.backend_name = backend_name
        self.model_name = model_name
        self.entities_by_id = {str(entity["entity_id"]): entity for entity in entities}

    @classmethod
    def build(
        cls,
        entities: list[dict[str, Any]],
        embedder: Embedder,
        model_name: str,
    ) -> "EntityIndex":
        texts: list[str] = []
        metadata: list[tuple[str, str]] = []
        for entity in entities:
            entity_id = str(entity["entity_id"])
            for name in entity_names(entity):
                texts.append(name)
                metadata.append((entity_id, name))

        vectors = embedder.encode(texts) if texts else []
        records = [
            {"entity_id": entity_id, "name": name, "vector": vector}
            for (entity_id, name), vector in zip(metadata, vectors)
        ]
        return cls(entities, records, embedder.backend_name, model_name)

    def search(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 5,
        threshold: float = 0.82,
    ) -> list[SearchResult]:
        query_vector = embedder.encode([query])[0]
        best_by_entity: dict[str, tuple[float, str]] = {}
        for record in self.records:
            score = cosine_similarity(query_vector, record["vector"])
            entity_id = str(record["entity_id"])
            current = best_by_entity.get(entity_id)
            if current is None or score > current[0]:
                best_by_entity[entity_id] = (score, str(record["name"]))

        ranked = sorted(best_by_entity.items(), key=lambda item: item[1][0], reverse=True)
        return [
            SearchResult(
                entity=self.entities_by_id[entity_id],
                source="embedding",
                matched_name=matched_name,
                score=score,
                low_confidence=score < threshold,
            )
            for entity_id, (score, matched_name) in ranked[:top_k]
        ]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(
                {
                    "entities": self.entities,
                    "records": self.records,
                    "backend_name": self.backend_name,
                    "model_name": self.model_name,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "EntityIndex":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        return cls(
            payload["entities"],
            payload["records"],
            payload.get("backend_name", "unknown"),
            payload.get("model_name", ""),
        )
