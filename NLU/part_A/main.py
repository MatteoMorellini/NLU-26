"""Compute final ATIS test results from a trained checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch.nn as nn

from functions import eval_loop, load_checkpoint_model
from utils import (
    DEVICE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZER_NAME,
    build_dataloaders,
    default_bin_dir,
    default_dataset_dir,
    get_gpt2_tokenizer,
    load_atis_splits,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate the best GPT-2 ATIS checkpoint.")
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--checkpoint", type=Path, default=default_bin_dir() / "best_model.pt")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-subtoken-strategy", choices=["first", "last"], default="first")
    return parser.parse_args()


def main() -> None:
    """Load a checkpoint and print final test metrics."""

    args = parse_args()
    model, label_vocab = load_checkpoint_model(args.checkpoint)
    tokenizer = get_gpt2_tokenizer(args.tokenizer)
    splits = load_atis_splits(args.dataset_dir, seed=args.seed)
    _, _, test_loader = build_dataloaders(
        splits=splits,
        label_vocab=label_vocab,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        max_length=args.max_length,
        device=DEVICE,
        seed=args.seed,
        slot_subtoken_strategy=args.slot_subtoken_strategy,
    )

    criterion_slots = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_intents = nn.CrossEntropyLoss()
    result = eval_loop(test_loader, criterion_slots, criterion_intents, model, label_vocab)
    print(f"Slot F1: {result.slot_f1:.4f}")
    print(f"Intent Accuracy: {result.intent_accuracy:.4f}")


if __name__ == "__main__":
    main()
