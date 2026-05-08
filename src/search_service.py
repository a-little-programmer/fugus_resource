"""Shared retrieval service for CLI and web UI."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .alias_matcher import match_alias
from .embedder import Embedder
from .entity_index import EntityIndex, SearchResult
from .latin_expander import build_latin_lookup, expand_latin_abbreviation
from .normalization import display_normalized_text


def retrieve(
    query: str,
    alias_dict: dict[str, dict[str, Any]],
    index: EntityIndex,
    embedder: Embedder | Callable[[], Embedder] | None,
    top_k: int = 5,
    threshold: float = 0.82,
) -> dict[str, Any]:
    entities_by_id = index.entities_by_id

    alias_hit = match_alias(alias_dict, query)
    if alias_hit:
        if len(alias_hit.entity_ids) == 1:
            result = SearchResult(
                entity=entities_by_id[alias_hit.entity_ids[0]],
                source="exact",
                matched_name=alias_hit.matched_name,
                score=1.0,
                low_confidence=False,
            )
            return make_payload("Exact match found.", [result])

        results = [
            SearchResult(
                entity=entities_by_id[entity_id],
                source="ambiguous_alias",
                matched_name=alias_hit.matched_name,
                score=1.0,
                low_confidence=False,
                ambiguous=True,
            )
            for entity_id in alias_hit.entity_ids
        ]
        return make_payload("Ambiguous exact alias; choose one candidate.", results)

    latin_lookup = build_latin_lookup(index.entities)
    expansion = expand_latin_abbreviation(query, entities_by_id, latin_lookup)
    if expansion:
        if expansion.status == "expanded_exact":
            entity = expansion.candidates[0]
            result = SearchResult(
                entity=entity,
                source="expanded_exact",
                matched_name=str(entity.get("scientific_name", "")),
                score=1.0,
                low_confidence=False,
            )
            return make_payload(
                f"Latin abbreviation expanded from {display_normalized_text(query)}.",
                [result],
            )

        results = [
            SearchResult(
                entity=entity,
                source="ambiguous_abbreviation",
                matched_name=str(entity.get("scientific_name", "")),
                score=1.0,
                low_confidence=False,
                ambiguous=True,
            )
            for entity in expansion.candidates
        ]
        return make_payload("Ambiguous Latin abbreviation; embedding search skipped.", results)

    if embedder is None:
        raise ValueError("Embedding search requires an embedder when no exact match is found.")

    resolved_embedder = embedder() if callable(embedder) else embedder
    results = index.search(query, resolved_embedder, top_k, threshold)
    return make_payload("No exact match found; showing most similar embedding candidates.", results)


def make_payload(message: str, results: list[SearchResult]) -> dict[str, Any]:
    return {"message": message, "results": [result_to_dict(result) for result in results]}


def result_to_dict(result: SearchResult) -> dict[str, Any]:
    entity = result.entity
    return {
        "entity_id": entity.get("entity_id"),
        "standard_name_cn": entity.get("standard_name_cn"),
        "scientific_name": entity.get("scientific_name"),
        "taxon_rank": entity.get("taxon_rank"),
        "matched_name": result.matched_name,
        "source": result.source,
        "score": round(result.score, 6),
        "low_confidence": result.low_confidence,
        "ambiguous": result.ambiguous,
    }
