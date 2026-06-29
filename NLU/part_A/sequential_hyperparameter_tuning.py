"""Run sequential hyperparameter tuning for ATIS part A.

The search order is learning_rate -> d_model -> n_heads -> num_layers -> ff_dim.
Each sweep starts from the best configuration found by the previous sweep,
using average development Slot F1 as the selection criterion.
"""

from __future__ import annotations

import csv
import os
import shutil
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any


PART_DIR = Path(__file__).resolve().parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.chdir(PART_DIR)

DEFAULT_ORDER = ("learning_rate", "d_model", "n_heads", "num_layers", "ff_dim", "dropout")
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 3
DEFAULT_TOKENIZER_NAME = "openai-community/gpt2"
DEFAULT_MAX_LENGTH = 1024
BASE_LEARNING_RATE = 5e-4
BASE_D_MODEL = 256
BASE_N_HEADS = 1
BASE_NUM_LAYERS = 1
BASE_FF_DIM = 1024
BASE_DROPOUT = 0.0

def load_training_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """Import training modules after CLI parsing so --help has no torch dependency."""

    from functions import summarize_repeated_runs, write_aggregate_results, write_experiment_results  # noqa: PLC0415
    from model import ModelConfig  # noqa: PLC0415
    from tuning import (  # noqa: PLC0415
        SWEEPS,
        append_result,
        base_model_config,
        format_lr,
        print_summary,
        run_repeated_experiment,
    )
    from utils import build_label_vocab, get_gpt2_tokenizer, load_atis_splits  # noqa: PLC0415

    return (
        SWEEPS,
        append_result,
        base_model_config,
        format_lr,
        print_summary,
        run_repeated_experiment,
        summarize_repeated_runs,
        write_aggregate_results,
        write_experiment_results,
        (ModelConfig, build_label_vocab, get_gpt2_tokenizer, load_atis_splits),
    )


def parse_args() -> Namespace:
    """Parse command-line options for the sequential search."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PART_DIR / "dataset" / "ATIS")
    parser.add_argument("--output-dir", type=Path, default=PART_DIR / "bin")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--split-seed", type=int, default=42, help="Seed used for the train/dev split.")
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[42, 101, 27])
    parser.add_argument("--initialization", choices=["gpt2", "lm", "xavier"], default="gpt2")
    parser.add_argument("--base-learning-rate", type=float, default=BASE_LEARNING_RATE)
    parser.add_argument("--base-d-model", type=int, default=BASE_D_MODEL)
    parser.add_argument("--base-n-heads", type=int, default=BASE_N_HEADS)
    parser.add_argument("--base-num-layers", type=int, default=BASE_NUM_LAYERS)
    parser.add_argument("--base-ff-dim", type=int, default=BASE_FF_DIM)
    parser.add_argument("--dropout", type=float, default=BASE_DROPOUT)
    parser.add_argument("--slot-subtoken-strategy", choices=["first", "last"], default="first")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sequential_tuning_results.csv"),
        help="CSV file where every aggregate trial result is appended.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("sequential_tuning_summary.csv"),
        help="CSV file where the best config after each sweep is written.",
    )
    return parser.parse_args()


def config_for_trial(
    base_config: Any,
    sweep_name: str,
    trial: dict[str, Any],
    base_learning_rate: float,
    name_prefix: str = "seq",
) -> tuple[str, Any, float]:
    """Create a trial config by changing one sweep option on the current best config."""

    tuned_value = trial[sweep_name]
    if sweep_name == "learning_rate":
        from tuning import format_lr  # noqa: PLC0415

        learning_rate = float(tuned_value)
        name = f"{name_prefix}_lr_{format_lr(learning_rate)}"
        return name, base_config, learning_rate

    name = f"{name_prefix}_{sweep_name}_{tuned_value}"
    return name, replace(base_config, **trial), base_learning_rate


def append_summary(
    output_path: Path,
    step: int,
    sweep_name: str,
    config: Any,
    learning_rate: float,
    summary: Any,
) -> None:
    """Append the winning configuration after a sequential tuning step."""

    fieldnames = [
        "step",
        "sweep",
        "name",
        "d_model",
        "n_heads",
        "num_layers",
        "ff_dim",
        "dropout",
        "learning_rate",
        "runs",
        "seeds",
        "dev_slot_f1_mean",
        "dev_slot_f1_std",
        "dev_intent_accuracy_mean",
        "dev_intent_accuracy_std",
        "test_slot_f1_mean",
        "test_slot_f1_std",
        "test_intent_accuracy_mean",
        "test_intent_accuracy_std",
        "best_checkpoint_path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists()
    if output_path.exists():
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames
        if existing_fieldnames != fieldnames:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "step": step,
                "sweep": sweep_name,
                "name": summary.name,
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "num_layers": config.num_layers,
                "ff_dim": config.ff_dim,
                "dropout": config.dropout,
                "learning_rate": learning_rate,
                "runs": summary.runs,
                "seeds": summary.seeds,
                "dev_slot_f1_mean": f"{summary.dev_slot_f1_mean:.4f}",
                "dev_slot_f1_std": f"{summary.dev_slot_f1_std:.4f}",
                "dev_intent_accuracy_mean": f"{summary.dev_intent_accuracy_mean:.4f}",
                "dev_intent_accuracy_std": f"{summary.dev_intent_accuracy_std:.4f}",
                "test_slot_f1_mean": f"{summary.test_slot_f1_mean:.4f}",
                "test_slot_f1_std": f"{summary.test_slot_f1_std:.4f}",
                "test_intent_accuracy_mean": f"{summary.test_intent_accuracy_mean:.4f}",
                "test_intent_accuracy_std": f"{summary.test_intent_accuracy_std:.4f}",
                "best_checkpoint_path": summary.best_checkpoint_path,
            }
        )


def main() -> None:
    """Run the sequential sweep and keep the best config between steps."""

    args = parse_args()
    (
        sweeps,
        append_result,
        base_model_config,
        _format_lr,
        print_summary,
        run_repeated_experiment,
        summarize_repeated_runs,
        write_aggregate_results,
        write_experiment_results,
        support_modules,
    ) = load_training_modules()
    _model_config, build_label_vocab, get_gpt2_tokenizer, load_atis_splits = support_modules

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_gpt2_tokenizer(args.tokenizer)
    splits = load_atis_splits(args.dataset_dir, seed=args.split_seed)
    label_vocab = build_label_vocab(splits)
    current_best_config = base_model_config(args, tokenizer, label_vocab)
    current_learning_rate = args.base_learning_rate

    all_summaries: list[Any] = []
    all_results: list[Any] = []

    for step, sweep_name in enumerate(DEFAULT_ORDER, start=1):
        print(f"\nSequential tuning step {step}: {sweep_name}")
        print(f"Starting from config: {current_best_config}")
        print(f"Starting from learning rate: {current_learning_rate}")

        step_best_config: Any | None = None
        step_best_learning_rate = current_learning_rate
        step_best_summary: Any | None = None

        for trial in sweeps[sweep_name]:
            name, config, learning_rate = config_for_trial(
                base_config=current_best_config,
                sweep_name=sweep_name,
                trial=trial,
                base_learning_rate=current_learning_rate,
            )

            if config.d_model % config.n_heads != 0:
                print(
                    f"Skipping {name}: d_model={config.d_model} "
                    f"is not divisible by n_heads={config.n_heads}"
                )
                continue

            repeated_results = run_repeated_experiment(
                name=name,
                model_config=config,
                learning_rate=learning_rate,
                args=args,
                splits=splits,
                label_vocab=label_vocab,
                tokenizer=tokenizer,
            )
            summary = summarize_repeated_runs(name, repeated_results)
            print_summary(summary)
            all_results.extend(repeated_results)
            all_summaries.append(summary)
            append_result(output_path=args.output_dir / args.output, summary=summary)

            if step_best_summary is None or summary.dev_slot_f1_mean > step_best_summary.dev_slot_f1_mean:
                step_best_config = config
                step_best_learning_rate = learning_rate
                step_best_summary = summary

        if step_best_config is None or step_best_summary is None:
            raise RuntimeError(f"No valid trials completed for sweep {sweep_name!r}")

        current_best_config = step_best_config
        current_learning_rate = step_best_learning_rate
        append_summary(
            output_path=args.output_dir / args.summary_output,
            step=step,
            sweep_name=sweep_name,
            config=current_best_config,
            learning_rate=current_learning_rate,
            summary=step_best_summary,
        )
        print(
            f"Best after {sweep_name} by dev Slot F1: {step_best_summary.name} "
            f"dev_slot_f1={step_best_summary.dev_slot_f1_mean:.4f} "
            f"dev_intent_acc={step_best_summary.dev_intent_accuracy_mean:.4f} "
            f"test_slot_f1={step_best_summary.test_slot_f1_mean:.4f} "
            f"test_intent_acc={step_best_summary.test_intent_accuracy_mean:.4f}"
        )

    write_experiment_results(all_results, args.output_dir / "sequential_tuning_per_seed_results.csv")
    write_aggregate_results(all_summaries, args.output_dir / "sequential_tuning_all_summary.csv")
    shutil.copyfile(step_best_summary.best_checkpoint_path, args.output_dir / "best_model.pt")

    print("\nFinal sequential best config:")
    print(current_best_config)
    print(f"Learning rate: {current_learning_rate}")


if __name__ == "__main__":
    main()
