#!/usr/bin/env python3
"""Fine-tune an embedding model for taxon entity normalization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.data_io import read_jsonl


DEFAULT_BASE_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_OUTPUT = "models/fugus-entity-embedding"


def load_training_dataset(path: str):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for training.") from exc

    rows = read_jsonl(path)
    records = []
    for row in rows:
        record = {
            "anchor": row["anchor"],
            "positive": row["positive"],
        }
        hard_negative = row.get("hard_negative")
        if hard_negative:
            record["hard_negative"] = hard_negative
        records.append(record)
    if not records:
        raise ValueError(f"No training rows found in {path}")
    return Dataset.from_list(records)


def load_training_examples(path: str):
    from sentence_transformers import InputExample

    rows = read_jsonl(path)
    examples = []
    for row in rows:
        texts = [row["anchor"], row["positive"]]
        if row.get("hard_negative"):
            texts.append(row["hard_negative"])
        examples.append(InputExample(texts=texts))
    if not examples:
        raise ValueError(f"No training rows found in {path}")
    return examples


def train(args: argparse.Namespace) -> None:
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        train_with_trainer(args)
    except (RuntimeError, ImportError) as exc:
        message = str(exc)
        if "datasets" not in message and "accelerate" not in message:
            raise
        if args.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "gradient accumulation requires the Trainer stack. "
                "Install requirements.txt or set --gradient-accumulation-steps 1."
            ) from exc
        print("Trainer dependencies are unavailable; falling back to SentenceTransformer.fit without gradient accumulation.")
        train_with_fit(args)


def train_with_trainer(args: argparse.Namespace) -> None:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    dataset = load_training_dataset(args.train)
    model = SentenceTransformer(args.base_model)
    loss = MultipleNegativesRankingLoss(
        model,
        hardness_mode="hard_negatives" if "hard_negative" in dataset.column_names else None,
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        save_strategy="no",
        logging_steps=args.logging_steps,
        report_to="none",
        use_cpu=args.cpu,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    )
    trainer.train()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved fine-tuned embedding model to {args.output}")


def train_with_fit(args: argparse.Namespace) -> None:
    from torch.utils.data import DataLoader
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    examples = load_training_examples(args.train)
    model = SentenceTransformer(args.base_model)
    dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, round(len(dataloader) * args.epochs * args.warmup_ratio))
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=int(args.epochs),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=args.output,
        use_amp=args.fp16,
        show_progress_bar=True,
    )
    print(f"Saved fine-tuned embedding model to {args.output}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a taxon entity embedding model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train", default="data/train_triplets.jsonl")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Load models from the local Hugging Face cache only.")
    parser.add_argument("--logging-steps", type=int, default=20)
    return parser


def main() -> None:
    train(make_parser().parse_args())


if __name__ == "__main__":
    main()
