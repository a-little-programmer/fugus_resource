#!/usr/bin/env python3
"""Run both canonical and all-names embedding evaluations."""

from __future__ import annotations

import os
import subprocess
import sys


def run(candidate_mode: str, output: str) -> None:
    cmd = [
        sys.executable,
        "evaluate_embedding.py",
        "--candidate-mode",
        candidate_mode,
        "--output",
        output,
        "--offline",
    ]
    print(f"\n== {candidate_mode} ==")
    subprocess.run(cmd, check=True)


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    run("canonical", "reports/embedding_eval_canonical.json")
    run("all-names", "reports/embedding_eval_all_names.json")


if __name__ == "__main__":
    main()
