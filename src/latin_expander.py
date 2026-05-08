"""Catalog-backed Latin genus abbreviation expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalization import normalize_text, parse_latin_abbreviation


@dataclass(frozen=True)
class LatinExpansion:
    status: str
    query: str
    candidates: list[dict[str, Any]]


def build_latin_lookup(entities: list[dict[str, Any]]) -> dict[tuple[str, str], list[str]]:
    lookup: dict[tuple[str, str], list[str]] = {}
    for entity in entities:
        scientific_name = normalize_text(str(entity.get("scientific_name") or ""))
        parts = scientific_name.split()
        if len(parts) < 2:
            continue
        genus, epithet = parts[0], parts[1]
        if not genus or not epithet:
            continue
        key = (genus[0], epithet)
        entity_id = str(entity["entity_id"])
        lookup.setdefault(key, [])
        if entity_id not in lookup[key]:
            lookup[key].append(entity_id)
    return lookup


def expand_latin_abbreviation(
    query: str,
    entities_by_id: dict[str, dict[str, Any]],
    lookup: dict[tuple[str, str], list[str]],
) -> LatinExpansion | None:
    parsed = parse_latin_abbreviation(query)
    if not parsed:
        return None

    entity_ids = lookup.get(parsed)
    if not entity_ids:
        return None

    candidates = [entities_by_id[entity_id] for entity_id in entity_ids]
    if len(candidates) == 1:
        return LatinExpansion("expanded_exact", query, candidates)
    return LatinExpansion("ambiguous_abbreviation", query, candidates)
