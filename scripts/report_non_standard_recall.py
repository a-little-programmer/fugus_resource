#!/usr/bin/env python3
"""Generate a recall report for aliases and former names."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_cli import load_entities
from src.alias_matcher import build_alias_dict
from src.embedder import create_embedder
from src.entity_index import EntityIndex
from src.search_service import retrieve


def expected_entity(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": str(entity["entity_id"]),
        "standard_name_cn": entity.get("standard_name_cn"),
        "scientific_name": entity.get("scientific_name"),
    }


def actual_entity(top_result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not top_result:
        return None
    return {
        "entity_id": top_result.get("entity_id"),
        "standard_name_cn": top_result.get("standard_name_cn"),
        "scientific_name": top_result.get("scientific_name"),
        "source": top_result.get("source"),
        "score": top_result.get("score"),
        "low_confidence": top_result.get("low_confidence"),
        "ambiguous": top_result.get("ambiguous"),
    }


def non_standard_inputs(entity: dict[str, Any]) -> list[tuple[str, str]]:
    inputs: list[tuple[str, str]] = []
    for value in entity.get("aliases") or []:
        if value:
            inputs.append(("alias", str(value)))
    for value in entity.get("former_names") or []:
        if value:
            inputs.append(("former_name", str(value)))
    return inputs


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    entities = load_entities(args.entities)
    alias_dict = build_alias_dict(entities)
    index = EntityIndex.load(args.index)
    embedder = create_embedder(index.model_name, args.backend)

    cases = []
    for entity in entities:
        expected = expected_entity(entity)
        for input_type, input_text in non_standard_inputs(entity):
            payload = retrieve(input_text, alias_dict, index, embedder, args.top_k, args.threshold)
            results = payload.get("results") or []
            top = results[0] if results else None
            cases.append(
                {
                    "input": input_text,
                    "input_type": input_type,
                    "expected": expected,
                    "actual_top1": actual_entity(top),
                    "hit_top1": bool(top and top.get("entity_id") == expected["entity_id"]),
                    "message": payload.get("message"),
                }
            )

    hits = sum(1 for case in cases if case["hit_top1"])
    summary = {
        "total_cases": len(cases),
        "top1_hits": hits,
        "top1_accuracy": round(hits / len(cases), 4) if cases else 0,
        "alias_cases": sum(1 for case in cases if case["input_type"] == "alias"),
        "former_name_cases": sum(1 for case in cases if case["input_type"] == "former_name"),
        "misses": [case for case in cases if not case["hit_top1"]],
    }
    return {"summary": summary, "cases": cases}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report recall for non-standard taxon expressions")
    parser.add_argument("--entities", default="data/species_entities.jsonl")
    parser.add_argument("--index", default="artifacts/species_index.pkl")
    parser.add_argument("--output", default="reports/non_standard_recall_inputs.json")
    parser.add_argument("--backend", choices=("auto", "sentence-transformers", "char-ngram"), default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.82)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    report = build_report(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = report["summary"]
    print(f"total_cases={summary['total_cases']}")
    print(f"top1_hits={summary['top1_hits']}")
    print(f"top1_accuracy={summary['top1_accuracy']}")
    print(f"alias_cases={summary['alias_cases']}")
    print(f"former_name_cases={summary['former_name_cases']}")
    print(f"misses={len(summary['misses'])}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
