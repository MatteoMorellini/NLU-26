"""Run one-at-a-time hyperparameter sweeps for ATIS part A.

Each sweep starts from the same baseline configuration and changes only one
model hyperparameter. Learning rate is treated as a training hyperparameter and
can also be used as the first step of a sequential search.
"""

from __future__ import annotations

import argparse
import os
import shutil
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional


PART_DIR = Path(__file__).resolve().parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.chdir(PART_DIR)

from functions import (  # noqa: E402
    AggregateExperimentResult,
    ExperimentResult,
    TrainConfig,
    run_training,
    summarize_repeated_runs,
    write_aggregate_results,
    write_experiment_results,
)
from model import ModelConfig  # noqa: E402
from utils import (  # noqa: E402
    DEVICE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZER_NAME,
    DatasetSplits,
    LabelVocab,
    build_dataloaders,
    build_label_vocab,
    default_bin_dir,
    default_dataset_dir,
    get_gpt2_tokenizer,
    load_atis_splits,
)


BASE_TRAIN_CONFIG = TrainConfig(
    learning_rate=5e-4,
    n_epochs=100,
    patience=3,
    seed=42,
    initialization="gpt2",
)

BASE_MODEL_CONFIG = {
    "d_model": 128,
    "n_heads": 1,
    "num_layers": 1,
    "ff_dim": 512,
    "dropout": 0.0,
}

SWEEPS: dict[str, tuple[dict[str, Any], ...]] = {
    "learning_rate": (
        {"learning_rate": 5e-2},
        {"learning_rate": 1e-2},
        {"learning_rate": 5e-3},
        {"learning_rate": 1e-3},
        {"learning_rate": 5e-4},
        {"learning_rate": 1e-4},
        {"learning_rate": 5e-5},
        {"learning_rate": 1e-5},
    ),
    "d_model": (
        {"d_model": 64},
        {"d_model": 128},
        {"d_model": 256},
        {"d_model": 384},
        {"d_model": 512},
    ),
    "n_heads": (
        {"n_heads": 1},
        {"n_heads": 2},
        {"n_heads": 4},
        {"n_heads": 8},
        {"n_heads": 10},
        {"n_heads": 16}
    ),
    "num_layers": (
        {"num_layers": 1},
        {"num_layers": 2},
        {"num_layers": 4},
        {"num_layers": 6},
        {"num_layers": 8},
    ),
    "ff_dim": (
        {"ff_dim": 64},
        {"ff_dim": 128},
        {"ff_dim": 256},
        {"ff_dim": 512},
        {"ff_dim": 1024},
        {"ff_dim": 2048}
    ),
    "dropout": (
        {"dropout": 0.0},
        {"dropout": 0.1},
        {"dropout": 0.2},
        {"dropout": 0.3},
        {"dropout": 0.4},
    ),
}
SEQUENTIAL_SWEEP_ORDER = ("learning_rate", "d_model", "n_heads", "num_layers", "ff_dim")


def parse_args() -> Namespace:
    """Parse command-line options for choosing a sweep."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        choices=(*SWEEPS.keys(), "all", "sequential"),
        default="all",
        help=(
            "Hyperparameter to tune. Use 'all' for independent one-at-a-time "
            "sweeps, or 'sequential' to carry the best config into the next sweep."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=default_bin_dir())
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=BASE_TRAIN_CONFIG.n_epochs)
    parser.add_argument("--patience", type=int, default=BASE_TRAIN_CONFIG.patience)
    parser.add_argument("--split-seed", type=int, default=BASE_TRAIN_CONFIG.seed, help="Seed used for the train/dev split.")
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[42, 101, 27])
    parser.add_argument("--initialization", choices=["gpt2", "lm", "xavier"], default=BASE_TRAIN_CONFIG.initialization)
    parser.add_argument("--base-learning-rate", type=float, default=BASE_TRAIN_CONFIG.learning_rate)
    parser.add_argument("--base-d-model", type=int, default=BASE_MODEL_CONFIG["d_model"])
    parser.add_argument("--base-n-heads", type=int, default=BASE_MODEL_CONFIG["n_heads"])
    parser.add_argument("--base-num-layers", type=int, default=BASE_MODEL_CONFIG["num_layers"])
    parser.add_argument("--base-ff-dim", type=int, default=BASE_MODEL_CONFIG["ff_dim"])
    parser.add_argument("--dropout", type=float, default=BASE_MODEL_CONFIG["dropout"])
    parser.add_argument("--slot-subtoken-strategy", choices=["first", "last"], default="first")
    return parser.parse_args()


def base_model_config(args: Namespace, tokenizer: Any, label_vocab: LabelVocab) -> ModelConfig:
    """Create the baseline model config for ATIS tuning."""

    return ModelConfig(
        vocab_size=len(tokenizer),
        slots_size=len(label_vocab.slot2id),
        n_intents=len(label_vocab.intent2id),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pos_emb_size=args.max_length,
        d_model=args.base_d_model,
        n_heads=args.base_n_heads,
        num_layers=args.base_num_layers,
        ff_dim=args.base_ff_dim,
        dropout=args.dropout,
    )


def config_for_trial(
    sweep_name: str,
    trial: dict[str, Any],
    base_config: ModelConfig,
    base_learning_rate: float,
    name_prefix: str = "tune",
) -> tuple[str, ModelConfig, float]:
    """Create a named model/training config pair for one sweep trial."""

    tuned_value = trial[sweep_name]
    if sweep_name == "learning_rate":
        learning_rate = float(tuned_value)
        name = f"{name_prefix}_lr_{format_lr(learning_rate)}"
        return name, base_config, learning_rate

    name = f"{name_prefix}_{sweep_name}_{tuned_value}"
    return name, replace(base_config, **trial), base_learning_rate


def append_result(output_path: Path, summary: AggregateExperimentResult) -> None:
    """Append one aggregate trial result to a tuning CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_rows = list(_read_aggregate_rows(output_path))
        write_aggregate_results([*existing_rows, summary], output_path)
        return
    write_aggregate_results([summary], output_path)


def run_trial(
    sweep_name: str,
    trial: dict[str, Any],
    base_config: ModelConfig,
    base_learning_rate: float,
    args: Namespace,
    splits: DatasetSplits,
    label_vocab: LabelVocab,
    tokenizer: Any,
    name_prefix: str = "tune",
) -> tuple[ModelConfig, float, AggregateExperimentResult]:
    """Run one trial, persist its metrics, and return config/learning-rate/result."""

    name, model_config, learning_rate = config_for_trial(
        sweep_name=sweep_name,
        trial=trial,
        base_config=base_config,
        base_learning_rate=base_learning_rate,
        name_prefix=name_prefix,
    )
    if model_config.d_model % model_config.n_heads != 0:
        raise ValueError(f"Invalid trial {name}: d_model must be divisible by n_heads")

    print(f"Tuning step: {name}")
    repeated_results = run_repeated_experiment(
        name=name,
        model_config=model_config,
        learning_rate=learning_rate,
        args=args,
        splits=splits,
        label_vocab=label_vocab,
        tokenizer=tokenizer,
    )
    summary = summarize_repeated_runs(name, repeated_results)
    print_summary(summary)
    write_experiment_results(repeated_results, args.output_dir / f"{name}_results.csv")
    append_result(args.output_dir / "tuning_results_summary.csv", summary)
    return model_config, learning_rate, summary


def run_sequential_sweeps(
    args: Namespace,
    splits: DatasetSplits,
    label_vocab: LabelVocab,
    tokenizer: Any,
) -> tuple[ModelConfig, float, AggregateExperimentResult]:
    """Tune sweeps in order, carrying each best config into the next sweep."""

    current_base = base_model_config(args, tokenizer, label_vocab)
    current_learning_rate = args.base_learning_rate
    final_summary: Optional[AggregateExperimentResult] = None

    for step_idx, sweep_name in enumerate(SEQUENTIAL_SWEEP_ORDER, start=1):
        best_config: Optional[ModelConfig] = None
        best_learning_rate = current_learning_rate
        best_summary: Optional[AggregateExperimentResult] = None

        print(f"Sequential sweep {step_idx}/{len(SEQUENTIAL_SWEEP_ORDER)}: {sweep_name}")
        print(f"Base config for this sweep: {current_base}")
        print(f"Base learning rate for this sweep: {current_learning_rate}")

        for trial in SWEEPS[sweep_name]:
            try:
                model_config, learning_rate, summary = run_trial(
                    sweep_name=sweep_name,
                    trial=trial,
                    base_config=current_base,
                    base_learning_rate=current_learning_rate,
                    args=args,
                    splits=splits,
                    label_vocab=label_vocab,
                    tokenizer=tokenizer,
                    name_prefix=f"seq{step_idx}",
                )
            except ValueError as exc:
                print(f"Skipping trial: {exc}")
                continue

            if best_summary is None or summary.dev_slot_f1_mean > best_summary.dev_slot_f1_mean:
                best_config = model_config
                best_learning_rate = learning_rate
                best_summary = summary

        if best_config is None or best_summary is None:
            raise RuntimeError(f"No trials were run for sweep {sweep_name}")

        current_base = best_config
        current_learning_rate = best_learning_rate
        final_summary = best_summary
        print(
            f"Best after {sweep_name} by dev Slot F1: {best_summary.name} "
            f"dev_slot_f1={best_summary.dev_slot_f1_mean:.4f} "
            f"test_slot_f1={best_summary.test_slot_f1_mean:.4f}"
        )

    if final_summary is None:
        raise RuntimeError("No sequential tuning trials completed")
    return current_base, current_learning_rate, final_summary


def main() -> None:
    """Run selected one-at-a-time sweeps and write their results."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_gpt2_tokenizer(args.tokenizer)
    splits = load_atis_splits(args.dataset_dir, seed=args.split_seed)
    label_vocab = build_label_vocab(splits)

    if args.sweep == "sequential":
        final_config, final_learning_rate, final_summary = run_sequential_sweeps(args, splits, label_vocab, tokenizer)
        shutil.copyfile(final_summary.best_checkpoint_path, args.output_dir / "best_model.pt")
        print(f"Final selected config: {final_config}")
        print(f"Final selected learning rate: {final_learning_rate}")
        return

    base_config = base_model_config(args, tokenizer, label_vocab)
    selected_sweeps = SWEEPS if args.sweep == "all" else {args.sweep: SWEEPS[args.sweep]}
    for sweep_name, trials in selected_sweeps.items():
        for trial in trials:
            try:
                run_trial(
                    sweep_name=sweep_name,
                    trial=trial,
                    base_config=base_config,
                    base_learning_rate=args.base_learning_rate,
                    args=args,
                    splits=splits,
                    label_vocab=label_vocab,
                    tokenizer=tokenizer,
                )
            except ValueError as exc:
                print(f"Skipping trial: {exc}")


def run_repeated_experiment(
    name: str,
    model_config: ModelConfig,
    learning_rate: float,
    args: argparse.Namespace,
    splits: DatasetSplits,
    label_vocab: LabelVocab,
    tokenizer: Any,
) -> list[ExperimentResult]:
    """Run one model configuration over all requested seeds."""

    repeated_results: list[ExperimentResult] = []
    for seed in args.run_seeds:
        train_loader, dev_loader, test_loader = build_dataloaders(
            splits=splits,
            label_vocab=label_vocab,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=DEVICE,
            seed=seed,
            slot_subtoken_strategy=args.slot_subtoken_strategy,
        )
        result = run_training(
            name=name,
            model_config=model_config,
            train_config=TrainConfig(
                learning_rate=learning_rate,
                n_epochs=args.epochs,
                patience=args.patience,
                seed=seed,
                initialization=args.initialization,
            ),
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            label_vocab=label_vocab,
            checkpoint_path=args.output_dir / f"{name}_seed_{seed}.pt",
        )
        repeated_results.append(result)

    return repeated_results


def print_summary(summary: AggregateExperimentResult) -> None:
    """Print the lab-style mean/std metrics for repeated runs."""

    print(
        f"{summary.name}: "
        f"Dev Slot F1 {summary.dev_slot_f1_mean:.3f} +- {summary.dev_slot_f1_std:.3f}; "
        f"Dev Intent Acc {summary.dev_intent_accuracy_mean:.3f} +- "
        f"{summary.dev_intent_accuracy_std:.3f}; "
        f"Test Slot F1 {summary.test_slot_f1_mean:.3f} +- {summary.test_slot_f1_std:.3f}; "
        f"Test Intent Acc {summary.test_intent_accuracy_mean:.3f} +- "
        f"{summary.test_intent_accuracy_std:.3f}"
    )


def format_lr(lr: float) -> str:
    """Format a learning rate for filenames."""

    return f"{lr:.0e}".replace("-", "m").replace("+", "")


def _read_aggregate_rows(output_path: Path) -> list[AggregateExperimentResult]:
    """Read existing aggregate rows before appending with the project writer."""

    import csv

    with output_path.open("r", newline="", encoding="utf-8") as handle:
        return [AggregateExperimentResult(**_coerce_aggregate_row(row)) for row in csv.DictReader(handle)]


def _coerce_aggregate_row(row: dict[str, str]) -> dict[str, Any]:
    """Coerce CSV strings back to aggregate result field types."""

    int_fields = {"d_model", "n_heads", "num_layers", "ff_dim", "runs"}
    float_fields = {
        "learning_rate",
        "dropout",
        "dev_slot_f1_mean",
        "dev_slot_f1_std",
        "dev_intent_accuracy_mean",
        "dev_intent_accuracy_std",
        "test_slot_f1_mean",
        "test_slot_f1_std",
        "test_intent_accuracy_mean",
        "test_intent_accuracy_std",
    }
    coerced: dict[str, Any] = {}
    for key, value in row.items():
        if key in int_fields:
            coerced[key] = int(value)
        elif key in float_fields:
            coerced[key] = float(value)
        else:
            coerced[key] = value
    return coerced


if __name__ == "__main__":
    main()
