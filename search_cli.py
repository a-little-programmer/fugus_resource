#!/usr/bin/env python3
"""CLI for hybrid taxon entity retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.alias_matcher import build_alias_dict, load_alias_dict, match_alias, save_alias_dict
from src.embedder import create_embedder
from src.entity_index import EntityIndex, SearchResult
from src.latin_expander import build_latin_lookup, expand_latin_abbreviation
from src.normalization import display_normalized_text
from src.search_service import result_to_dict


DEFAULT_DATA_PATH = "data/species_entities.jsonl"
DEFAULT_ALIAS_PATH = "artifacts/alias_dict.json"
DEFAULT_INDEX_PATH = "artifacts/species_index.pkl"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_THRESHOLD = 0.82


def load_entities(path: str | Path) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entity = json.loads(line)
            if "entity_id" not in entity or "scientific_name" not in entity:
                raise ValueError(f"Invalid entity at line {line_number}: missing required fields")
            entities.append(entity)
    return entities


def build_command(args: argparse.Namespace) -> None:
    entities = load_entities(args.data)
    alias_dict = build_alias_dict(entities)
    save_alias_dict(alias_dict, args.alias)

    embedder = create_embedder(args.model, args.backend)
    index = EntityIndex.build(entities, embedder, args.model)
    index.save(args.index)

    print(f"Built alias dictionary: {args.alias} ({len(alias_dict)} aliases)")
    print(
        f"Built vector index: {args.index} "
        f"({len(index.entities)} entities, {len(index.records)} name vectors, {index.backend_name})"
    )


def query_command(args: argparse.Namespace) -> None:
    alias_dict = load_alias_dict(args.alias)
    index = EntityIndex.load(args.index)
    entities_by_id = index.entities_by_id

    alias_hit = match_alias(alias_dict, args.query)
    if alias_hit:
        if len(alias_hit.entity_ids) == 1:
            result = SearchResult(
                entity=entities_by_id[alias_hit.entity_ids[0]],
                source="exact",
                matched_name=alias_hit.matched_name,
                score=1.0,
                low_confidence=False,
            )
            print_results([result], args.format, message="Exact match found.")
            return
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
        print_results(results, args.format, message="Ambiguous exact alias; choose one candidate.")
        return

    latin_lookup = build_latin_lookup(index.entities)
    expansion = expand_latin_abbreviation(args.query, entities_by_id, latin_lookup)
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
            print_results(
                [result],
                args.format,
                message=f"Latin abbreviation expanded from {display_normalized_text(args.query)}.",
            )
            return

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
        print_results(
            results,
            args.format,
            message="Ambiguous Latin abbreviation; embedding search skipped.",
        )
        return

    embedder = create_embedder(index.model_name or args.model, args.backend)
    results = index.search(args.query, embedder, args.top_k, args.threshold)
    print_results(
        results,
        args.format,
        message="No exact match found; showing most similar embedding candidates.",
    )


def eval_command(args: argparse.Namespace) -> None:
    index = EntityIndex.load(args.index)
    embedder = create_embedder(index.model_name or args.model, args.backend)
    probes = args.query or [
        "黄色葡萄球菌",
        "表皮葡萄球菌",
        "大肠",
        "Escherichia colli",
        "沙门氏菌",
        "链球菌",
    ]

    payload = []
    for probe in probes:
        results = index.search(probe, embedder, args.top_k, args.threshold)
        payload.append(
            {
                "query": probe,
                "results": [result_to_dict(result) for result in results],
            }
        )

    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for item in payload:
        print(f"Query: {item['query']}")
        for index_number, result in enumerate(item["results"], start=1):
            low_confidence = " low_confidence" if result["low_confidence"] else ""
            print(
                f"  {index_number}. {result['standard_name_cn']} | {result['scientific_name']} "
                f"| score={result['score']:.4f}{low_confidence} | matched={result['matched_name']}"
            )


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


def print_results(results: list[SearchResult], output_format: str, message: str) -> None:
    payload = {"message": message, "results": [result_to_dict(result) for result in results]}
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(message)
    for index, result in enumerate(payload["results"], start=1):
        low_confidence = " low_confidence" if result["low_confidence"] else ""
        ambiguous = " ambiguous" if result["ambiguous"] else ""
        print(
            f"{index}. {result['standard_name_cn']} | {result['scientific_name']} "
            f"| source={result['source']} | score={result['score']:.4f}"
            f"{low_confidence}{ambiguous} | matched={result['matched_name']}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid taxon entity retrieval prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build alias dictionary and vector index")
    build_parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    build_parser.add_argument("--alias", default=DEFAULT_ALIAS_PATH)
    build_parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    build_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    build_parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sentence-transformers", "char-ngram"),
    )
    build_parser.set_defaults(func=build_command)

    query_parser = subparsers.add_parser("query", help="Query taxon entities")
    query_parser.add_argument("query")
    query_parser.add_argument("--alias", default=DEFAULT_ALIAS_PATH)
    query_parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    query_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    query_parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sentence-transformers", "char-ngram"),
    )
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    query_parser.add_argument("--format", choices=("table", "json"), default="table")
    query_parser.set_defaults(func=query_command)

    eval_parser = subparsers.add_parser("eval", help="Print probe scores for threshold calibration")
    eval_parser.add_argument("query", nargs="*")
    eval_parser.add_argument("--index", default=DEFAULT_INDEX_PATH)
    eval_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    eval_parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sentence-transformers", "char-ngram"),
    )
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    eval_parser.add_argument("--format", choices=("table", "json"), default="table")
    eval_parser.set_defaults(func=eval_command)
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
