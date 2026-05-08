#!/usr/bin/env python3
"""Build triplet training data and eval splits for taxon embedding tuning."""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.alias_matcher import entity_names
from src.data_io import read_jsonl, write_jsonl
from src.normalization import normalize_text


def canonical_name(entity: dict[str, Any]) -> str:
    return str(entity.get("standard_name_cn") or entity.get("scientific_name") or entity["entity_id"])


def genus(entity: dict[str, Any]) -> str:
    scientific_name = str(entity.get("scientific_name") or "")
    return scientific_name.split()[0].casefold() if scientific_name.split() else ""


def unique_names(entity: dict[str, Any]) -> list[str]:
    names = [name for name in entity_names(entity) if normalize_text(name)]
    return list(dict.fromkeys(names))


def split_entities(
    entities: list[dict[str, Any]],
    entity_eval_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(entities)
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, round(len(shuffled) * entity_eval_ratio)) if shuffled else 0
    return shuffled[eval_size:], shuffled[:eval_size]


def choose_alias_holdouts(
    train_entities: list[dict[str, Any]],
    ratio: float,
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed + 17)
    holdouts: dict[str, str] = {}
    for entity in train_entities:
        names = unique_names(entity)
        canonical = canonical_name(entity)
        candidates = [name for name in names if normalize_text(name) != normalize_text(canonical)]
        if len(names) >= 3 and candidates and rng.random() < ratio:
            holdouts[str(entity["entity_id"])] = candidates[-1]
    return holdouts


def hard_negative_for(
    entity: dict[str, Any],
    train_entities: list[dict[str, Any]],
    by_genus: dict[str, list[dict[str, Any]]],
) -> str:
    entity_id = str(entity["entity_id"])
    same_genus = [candidate for candidate in by_genus.get(genus(entity), []) if str(candidate["entity_id"]) != entity_id]
    if same_genus:
        return canonical_name(same_genus[0])

    rank = entity.get("taxon_rank")
    same_rank = [
        candidate
        for candidate in train_entities
        if str(candidate["entity_id"]) != entity_id and candidate.get("taxon_rank") == rank
    ]
    if same_rank:
        return canonical_name(same_rank[0])

    for candidate in train_entities:
        if str(candidate["entity_id"]) != entity_id:
            return canonical_name(candidate)
    return canonical_name(entity)


def build_training_rows(
    train_entities: list[dict[str, Any]],
    alias_holdouts: dict[str, str],
    max_pairs_per_entity: int,
) -> list[dict[str, Any]]:
    by_genus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entity in train_entities:
        by_genus[genus(entity)].append(entity)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entity in train_entities:
        entity_id = str(entity["entity_id"])
        canonical = canonical_name(entity)
        heldout = alias_holdouts.get(entity_id)
        names = [name for name in unique_names(entity) if normalize_text(name) != normalize_text(heldout or "")]
        anchors = [name for name in names if normalize_text(name) != normalize_text(canonical)]
        if not anchors and len(names) >= 2:
            anchors = [names[1]]
        hard_negative = hard_negative_for(entity, train_entities, by_genus)
        for anchor in anchors[:max_pairs_per_entity]:
            key = (normalize_text(anchor), normalize_text(canonical), normalize_text(hard_negative))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "anchor": anchor,
                    "positive": canonical,
                    "hard_negative": hard_negative,
                    "entity_id": entity_id,
                }
            )
    return rows


def build_eval_rows(
    train_entities: list[dict[str, Any]],
    entity_eval_entities: list[dict[str, Any]],
    alias_holdouts: dict[str, str],
    max_queries_per_entity: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_by_id = {str(entity["entity_id"]): entity for entity in train_entities}

    for entity_id, query in alias_holdouts.items():
        entity = train_by_id[entity_id]
        rows.append(
            {
                "query": query,
                "target_entity_id": entity_id,
                "target_name": canonical_name(entity),
                "split": "alias",
            }
        )

    for entity in entity_eval_entities:
        names = unique_names(entity)
        canonical = canonical_name(entity)
        queries = [name for name in names if normalize_text(name) != normalize_text(canonical)]
        if not queries:
            queries = names[:1]
        for query in queries[:max_queries_per_entity]:
            rows.append(
                {
                    "query": query,
                    "target_entity_id": str(entity["entity_id"]),
                    "target_name": canonical,
                    "split": "entity",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build embedding training and evaluation data")
    parser.add_argument("--entities", default="data/species_entities.jsonl")
    parser.add_argument("--train-output", default="data/train_triplets.jsonl")
    parser.add_argument("--eval-output", default="data/eval_queries.jsonl")
    parser.add_argument("--entity-eval-ratio", type=float, default=0.2)
    parser.add_argument("--alias-eval-ratio", type=float, default=0.4)
    parser.add_argument("--max-pairs-per-entity", type=int, default=8)
    parser.add_argument("--max-eval-queries-per-entity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    entities = read_jsonl(args.entities)
    train_entities, entity_eval_entities = split_entities(entities, args.entity_eval_ratio, args.seed)
    alias_holdouts = choose_alias_holdouts(train_entities, args.alias_eval_ratio, args.seed)
    train_rows = build_training_rows(train_entities, alias_holdouts, args.max_pairs_per_entity)
    eval_rows = build_eval_rows(
        train_entities,
        entity_eval_entities,
        alias_holdouts,
        args.max_eval_queries_per_entity,
    )

    write_jsonl(train_rows, args.train_output)
    write_jsonl(eval_rows, args.eval_output)
    print(f"Wrote {len(train_rows)} training triplets to {args.train_output}")
    print(f"Wrote {len(eval_rows)} eval queries to {args.eval_output}")
    print(f"Train entities: {len(train_entities)}; entity-split eval entities: {len(entity_eval_entities)}")


if __name__ == "__main__":
    main()
