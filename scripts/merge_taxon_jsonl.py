#!/usr/bin/env python3
"""Merge taxon JSONL files by normalized scientific name."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge(base_path: str, candidate_path: str, output_path: str) -> None:
    rows = load_jsonl(base_path)
    seen = {row["scientific_name"].casefold() for row in rows}
    next_id = next_numeric_id(rows)
    for candidate in load_jsonl(candidate_path):
        scientific_name = candidate["scientific_name"].casefold()
        if scientific_name in seen:
            continue
        candidate = dict(candidate)
        candidate["entity_id"] = f"TAXON:{next_id:04d}"
        next_id += 1
        seen.add(scientific_name)
        rows.append(candidate)

    with Path(output_path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def next_numeric_id(rows: list[dict]) -> int:
    values = []
    for row in rows:
        entity_id = str(row.get("entity_id", ""))
        if entity_id.startswith("TAXON:") and entity_id[6:].isdigit():
            values.append(int(entity_id[6:]))
    return max(values, default=0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge taxon candidates into the demo data")
    parser.add_argument("--base", default="data/species_entities.jsonl")
    parser.add_argument("--candidates", default="data/cicc_product_candidates.jsonl")
    parser.add_argument("--output", default="data/species_entities.merged.jsonl")
    args = parser.parse_args()
    merge(args.base, args.candidates, args.output)


if __name__ == "__main__":
    main()
