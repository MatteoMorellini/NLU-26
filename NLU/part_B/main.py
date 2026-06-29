"""Run Part B fine-tuning for BERT and GPT-2 on ATIS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from functions import TrainConfig, run_training, write_experiment_results
from model import ModelConfig
from utils import (
    DEFAULT_BERT_MODEL,
    DEFAULT_GPT2_MODEL,
    DEFAULT_MAX_LENGTH,
    DEVICE,
    ModelType,
    build_dataloaders,
    build_label_vocab,
    default_bin_dir,
    default_dataset_dir,
    get_tokenizer,
    load_atis_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT/GPT-2 for ATIS intent classification and slot filling.")
    parser.add_argument("--model", choices=["bert", "gpt2", "all"], default="all")
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--gpt2-model", type=str, default=DEFAULT_GPT2_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=default_bin_dir())
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--intent-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[tuple[ModelType, str]]:
    if args.model == "bert":
        return [("bert", args.bert_model)]
    if args.model == "gpt2":
        return [("gpt2", args.gpt2_model)]
    return [("bert", args.bert_model), ("gpt2", args.gpt2_model)]


def main() -> None:
    args = parse_args()
    splits = load_atis_splits(args.dataset_dir, seed=args.seed)
    label_vocab = build_label_vocab(splits)
    train_config = TrainConfig(
        learning_rate=args.lr,
        n_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        intent_loss_weight=args.intent_loss_weight,
        slot_loss_weight=args.slot_loss_weight,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    results = []
    for model_type, pretrained_model_name in selected_models(args):
        tokenizer = get_tokenizer(model_type, pretrained_model_name)
        train_loader, dev_loader, test_loader = build_dataloaders(
            splits=splits,
            label_vocab=label_vocab,
            tokenizer=tokenizer,
            model_type=model_type,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=DEVICE,
            seed=args.seed,
        )
        model_config = ModelConfig(
            model_type=model_type,
            pretrained_model_name=pretrained_model_name,
            slots_size=len(label_vocab.slot2id),
            n_intents=len(label_vocab.intent2id),
            pad_token_id=tokenizer.pad_token_id,
            dropout=args.dropout,
        )
        checkpoint_path = args.output_dir / f"best_{model_type}.pt"
        result = run_training(
            model_config=model_config,
            train_config=train_config,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            label_vocab=label_vocab,
            checkpoint_path=checkpoint_path,
        )
        results.append(result)
        print(
            f"{model_type.upper()} | "
            f"dev slot F1: {result.best_dev_slot_f1:.4f} | "
            f"dev intent acc: {result.best_dev_intent_accuracy:.4f} | "
            f"test slot F1: {result.test_slot_f1:.4f} | "
            f"test intent acc: {result.test_intent_accuracy:.4f}"
        )

    write_experiment_results(results, args.output_dir / "part_b_results.csv")


if __name__ == "__main__":
    main()
