"""Fine-tune pretrained BERT and GPT-2 on ATIS intent and slot tasks.

This module is the explicit Part 2.B fine-tuning entry point. It reuses the
existing Part B components:

- ``utils.py`` loads ATIS and aligns word-level slot labels to first subtokens.
- ``model.py`` defines BERT and GPT-2 multitask heads.
- ``functions.py`` performs multitask optimization and CoNLL slot evaluation.

Run examples:

    python part_B/fine_tuning.py --model all
    python part_B/fine_tuning.py --model bert-base
    python part_B/fine_tuning.py --model bert-large
    python part_B/fine_tuning.py --model gpt2 --gpt2-model gpt2-medium
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from functions import ExperimentResult, TrainConfig, run_training, write_experiment_results
from model import ModelConfig
from utils import (
    DEFAULT_BERT_MODEL,
    DEFAULT_GPT2_MODEL,
    DEFAULT_MAX_LENGTH,
    DEVICE,
    DatasetSplits,
    LabelVocab,
    ModelType,
    build_dataloaders,
    build_label_vocab,
    default_bin_dir,
    default_dataset_dir,
    get_tokenizer,
    load_atis_splits,
)


BERT_LARGE_MODEL = "bert-large-uncased"
GPT2_MEDIUM_MODEL = "gpt2-medium"
DEFAULT_RUN_SEEDS = (42, 101, 27)


@dataclass(frozen=True)
class FineTuningConfig:
    """Configuration shared by BERT and GPT-2 fine-tuning runs."""

    dataset_dir: Path
    output_dir: Path
    max_length: int = DEFAULT_MAX_LENGTH
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 100
    patience: int = 3
    learning_rate: float = 5e-5
    seed: int = 42
    dropout: float = 0.1
    intent_loss_weight: float = 1.0
    slot_loss_weight: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    def to_train_config(self) -> TrainConfig:
        """Convert CLI-level configuration to the training-loop config."""

        return TrainConfig(
            learning_rate=self.learning_rate,
            n_epochs=self.epochs,
            patience=self.patience,
            seed=self.seed,
            intent_loss_weight=self.intent_loss_weight,
            slot_loss_weight=self.slot_loss_weight,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
        )


def checkpoint_name(model_type: ModelType, pretrained_model_name: str) -> str:
    """Build a filesystem-friendly checkpoint name."""

    safe_model_name = pretrained_model_name.replace("/", "_")
    return f"best_{model_type}_{safe_model_name}.pt"


def average_results(results: list[ExperimentResult]) -> dict[str, str]:
    """Average metrics from repeated seed runs for one pretrained model."""

    if not results:
        raise ValueError("Cannot average an empty result list")

    best_result = max(results, key=lambda result: (result.best_dev_slot_f1, result.best_dev_intent_accuracy))
    return {
        "model_type": best_result.model_type,
        "pretrained_model_name": best_result.pretrained_model_name,
        "runs": str(len(results)),
        "seeds": ",".join(str(result.seed) for result in results),
        "learning_rate": f"{best_result.learning_rate:g}",
        "best_dev_slot_f1_mean": f"{mean(result.best_dev_slot_f1 for result in results):.4f}",
        "best_dev_intent_accuracy_mean": f"{mean(result.best_dev_intent_accuracy for result in results):.4f}",
        "test_slot_f1_mean": f"{mean(result.test_slot_f1 for result in results):.4f}",
        "test_intent_accuracy_mean": f"{mean(result.test_intent_accuracy for result in results):.4f}",
        "epochs_ran_mean": f"{mean(result.epochs_ran for result in results):.2f}",
        "best_checkpoint_path": best_result.checkpoint_path,
    }


def write_averaged_results(rows: list[dict[str, str]], output_path: str | Path) -> None:
    """Write model-level averaged metrics to CSV."""

    if not rows:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fine_tune_model(
    model_type: ModelType,
    pretrained_model_name: str,
    splits: DatasetSplits,
    label_vocab: LabelVocab,
    config: FineTuningConfig,
) -> ExperimentResult:
    """Fine-tune one pretrained model on ATIS.

    BERT uses the hidden state of the first special token for intent
    classification. GPT-2 uses the final non-padding token hidden state instead,
    because it is decoder-only and has no CLS token. Slot filling is learned at
    token level for both models, with non-first subtokens masked out by
    ``IGNORE_SLOT_ID`` in ``utils.py``.
    """

    tokenizer = get_tokenizer(model_type, pretrained_model_name)
    train_loader, dev_loader, test_loader = build_dataloaders(
        splits=splits,
        label_vocab=label_vocab,
        tokenizer=tokenizer,
        model_type=model_type,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        max_length=config.max_length,
        device=DEVICE,
        seed=config.seed,
    )

    model_config = ModelConfig(
        model_type=model_type,
        pretrained_model_name=pretrained_model_name,
        slots_size=len(label_vocab.slot2id),
        n_intents=len(label_vocab.intent2id),
        pad_token_id=tokenizer.pad_token_id,
        dropout=config.dropout,
    )
    checkpoint_path = config.output_dir / checkpoint_name(model_type, pretrained_model_name)

    return run_training(
        model_config=model_config,
        train_config=config.to_train_config(),
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        label_vocab=label_vocab,
        checkpoint_path=checkpoint_path,
    )


def selected_models(args: argparse.Namespace) -> list[tuple[ModelType, str]]:
    """Return the pretrained models requested from the command line."""

    if args.model == "bert-base":
        return [("bert", args.bert_model)]
    if args.model == "bert-large":
        return [("bert", args.bert_large_model)]
    if args.model == "gpt2":
        return [("gpt2", args.gpt2_model)]
    if args.model == "gpt2-medium":
        return [("gpt2", args.gpt2_medium_model)]
    return [
        ("gpt2", args.gpt2_model),
        ("gpt2", args.gpt2_medium_model),
        ("bert", args.bert_model),
        ("bert", args.bert_large_model),
    ]


def parse_args() -> argparse.Namespace:
    """Parse fine-tuning command-line options."""

    parser = argparse.ArgumentParser(
        description="Fine-tune pretrained BERT/GPT-2 on ATIS intent classification and slot filling.",
    )
    parser.add_argument("--model", choices=["gpt2", "gpt2-medium", "bert-base", "bert-large", "all"], default="all")
    parser.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL)
    parser.add_argument("--bert-large-model", type=str, default=BERT_LARGE_MODEL)
    parser.add_argument("--gpt2-model", type=str, default=DEFAULT_GPT2_MODEL)
    parser.add_argument("--gpt2-medium-model", type=str, default=GPT2_MEDIUM_MODEL)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=default_bin_dir())
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the fixed train/dev split.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_RUN_SEEDS), help="Training seeds to average.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--intent-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    """Run requested fine-tuning experiments and write results."""

    args = parse_args()
    config = FineTuningConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        seed=args.seed,
        dropout=args.dropout,
        intent_loss_weight=args.intent_loss_weight,
        slot_loss_weight=args.slot_loss_weight,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    splits = load_atis_splits(config.dataset_dir, seed=config.seed)
    label_vocab = build_label_vocab(splits)

    results: list[ExperimentResult] = []
    averaged_results: list[dict[str, str]] = []
    for model_type, pretrained_model_name in selected_models(args):
        model_results: list[ExperimentResult] = []
        for seed in args.seeds:
            run_config = replace(config, seed=seed, output_dir=config.output_dir / f"seed_{seed}")
            result = fine_tune_model(model_type, pretrained_model_name, splits, label_vocab, run_config)
            results.append(result)
            model_results.append(result)
            print(
                f"{model_type.upper()} ({pretrained_model_name}) seed={seed} | "
                f"dev slot F1: {result.best_dev_slot_f1:.4f} | "
                f"dev intent acc: {result.best_dev_intent_accuracy:.4f} | "
                f"test slot F1: {result.test_slot_f1:.4f} | "
                f"test intent acc: {result.test_intent_accuracy:.4f}"
            )
        averages = average_results(model_results)
        averaged_results.append(averages)
        print(
            f"{model_type.upper()} ({pretrained_model_name}) averaged over seeds {averages['seeds']} | "
            f"dev slot F1: {averages['best_dev_slot_f1_mean']} | "
            f"dev intent acc: {averages['best_dev_intent_accuracy_mean']} | "
            f"test slot F1: {averages['test_slot_f1_mean']} | "
            f"test intent acc: {averages['test_intent_accuracy_mean']}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_results(results, config.output_dir / "part_b_fine_tuning_results.csv")
    write_averaged_results(averaged_results, config.output_dir / "part_b_fine_tuning_averages.csv")


if __name__ == "__main__":
    main()
