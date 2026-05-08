"""Exact alias dictionary for taxon entities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalization import normalize_text


@dataclass(frozen=True)
class AliasHit:
    normalized_query: str
    entity_ids: list[str]
    matched_name: str


def entity_names(entity: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("standard_name_cn", "scientific_name"):
        value = entity.get(key)
        if value:
            names.append(str(value))
    for key in ("aliases", "former_names"):
        values = entity.get(key) or []
        names.extend(str(value) for value in values if value)
    return list(dict.fromkeys(names))


def build_alias_dict(entities: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    alias_dict: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity["entity_id"])
        for name in entity_names(entity):
            normalized = normalize_text(name)
            if not normalized:
                continue
            entry = alias_dict.setdefault(
                normalized,
                {"entity_ids": [], "matched_names": []},
            )
            if entity_id not in entry["entity_ids"]:
                entry["entity_ids"].append(entity_id)
            if name not in entry["matched_names"]:
                entry["matched_names"].append(name)
    return alias_dict


def match_alias(alias_dict: dict[str, dict[str, Any]], query: str) -> AliasHit | None:
    normalized = normalize_text(query)
    entry = alias_dict.get(normalized)
    if not entry:
        return None
    matched_names = entry.get("matched_names") or [query]
    return AliasHit(
        normalized_query=normalized,
        entity_ids=list(entry["entity_ids"]),
        matched_name=str(matched_names[0]),
    )


def save_alias_dict(alias_dict: dict[str, dict[str, Any]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(alias_dict, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_alias_dict(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
