#!/usr/bin/env python3
"""Evaluate embedding models on taxon normalization queries."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.alias_matcher import entity_names
from src.data_io import read_jsonl
from src.embedder import SentenceTransformerEmbedder, cosine_similarity


DEFAULT_BASE_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_FINE_TUNED_MODEL = "models/fugus-entity-embedding"


def candidate_texts(entity: dict[str, Any], mode: str) -> list[str]:
    if mode == "all-names":
        return entity_names(entity)
    texts = []
    for key in ("standard_name_cn", "scientific_name"):
        value = entity.get(key)
        if value:
            texts.append(str(value))
    return list(dict.fromkeys(texts))


def build_candidate_vectors(
    entities: list[dict[str, Any]],
    embedder: SentenceTransformerEmbedder,
    candidate_mode: str,
) -> list[dict[str, Any]]:
    texts: list[str] = []
    metadata: list[tuple[str, str]] = []
    for entity in entities:
        for text in candidate_texts(entity, candidate_mode):
            texts.append(text)
            metadata.append((str(entity["entity_id"]), text))
    vectors = embedder.encode(texts)
    return [
        {"entity_id": entity_id, "text": text, "vector": vector}
        for (entity_id, text), vector in zip(metadata, vectors)
    ]


def search(
    query: str,
    embedder: SentenceTransformerEmbedder,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    query_vector = embedder.encode([query])[0]
    best_by_entity: dict[str, tuple[float, str]] = {}
    for candidate in candidates:
        score = cosine_similarity(query_vector, candidate["vector"])
        entity_id = candidate["entity_id"]
        current = best_by_entity.get(entity_id)
        if current is None or score > current[0]:
            best_by_entity[entity_id] = (score, candidate["text"])
    ranked = sorted(best_by_entity.items(), key=lambda item: item[1][0], reverse=True)
    return [
        {"entity_id": entity_id, "score": score, "matched_text": matched_text}
        for entity_id, (score, matched_text) in ranked[:top_k]
    ]


def evaluate_model(
    model_name: str,
    entities: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    top_k: int,
    candidate_mode: str,
) -> dict[str, Any]:
    embedder = SentenceTransformerEmbedder(model_name)
    candidates = build_candidate_vectors(entities, embedder, candidate_mode)
    details = []
    metrics = defaultdict(lambda: {"count": 0, "top1": 0, "top3": 0, "mrr": 0.0})

    for row in eval_rows:
        results = search(row["query"], embedder, candidates, top_k)
        target = row["target_entity_id"]
        rank = next((index + 1 for index, result in enumerate(results) if result["entity_id"] == target), None)
        split = row.get("split", "unknown")
        for bucket in ("all", split):
            metrics[bucket]["count"] += 1
            metrics[bucket]["top1"] += int(rank == 1)
            metrics[bucket]["top3"] += int(rank is not None and rank <= 3)
            metrics[bucket]["mrr"] += 0.0 if rank is None else 1.0 / rank
        details.append(
            {
                "query": row["query"],
                "target_entity_id": target,
                "target_name": row.get("target_name"),
                "split": split,
                "rank": rank,
                "top_results": results,
            }
        )

    summary = {}
    for bucket, values in metrics.items():
        count = values["count"] or 1
        summary[bucket] = {
            "count": values["count"],
            "top1_accuracy": round(values["top1"] / count, 4),
            "top3_recall": round(values["top3"] / count, 4),
            "mrr": round(values["mrr"] / count, 4),
        }
    return {"model": model_name, "candidate_mode": candidate_mode, "summary": summary, "details": details}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate taxon embedding models")
    parser.add_argument("--entities", default="data/species_entities.jsonl")
    parser.add_argument("--eval", default="data/eval_queries.jsonl")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument(
        "--candidate-mode",
        choices=("canonical", "all-names"),
        default="canonical",
        help="canonical uses standard Chinese/scientific names only; all-names also indexes aliases and former names.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="reports/embedding_eval.json")
    parser.add_argument("--offline", action="store_true", help="Load models from the local Hugging Face cache only.")
    return parser


def default_models() -> list[str]:
    models = [DEFAULT_BASE_MODEL]
    if Path(DEFAULT_FINE_TUNED_MODEL).exists():
        models.append(DEFAULT_FINE_TUNED_MODEL)
    return models


def main() -> None:
    args = make_parser().parse_args()
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    entities = read_jsonl(args.entities)
    eval_rows = read_jsonl(args.eval)
    model_names = args.models or default_models()
    payload = [
        evaluate_model(model_name, entities, eval_rows, args.top_k, args.candidate_mode)
        for model_name in model_names
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    for result in payload:
        print(result["model"])
        for split, metrics in sorted(result["summary"].items()):
            print(
                f"  {split}: top1={metrics['top1_accuracy']:.4f} "
                f"top3={metrics['top3_recall']:.4f} mrr={metrics['mrr']:.4f} n={metrics['count']}"
            )
    print(f"Wrote detailed results to {args.output}")


if __name__ == "__main__":
    main()
