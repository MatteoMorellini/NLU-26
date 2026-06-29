"""Run sequential hyperparameter tuning for Part B pretrained fine-tuning.

The search is intentionally sequential: for each requested pretrained model,
the script sweeps one hyperparameter at a time, keeps the best value by
development slot F1, then uses that configuration as the starting point for the
next sweep.

Examples:

    python part_B/sequential_fine_tuning.py
    python part_B/sequential_fine_tuning.py --models gpt2 bert-base
    python part_B/sequential_fine_tuning.py --models gpt2 gpt2-medium bert-base bert-large
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from fine_tuning import (  # noqa: E402
    BERT_LARGE_MODEL,
    GPT2_MEDIUM_MODEL,
    FineTuningConfig,
    fine_tune_model,
)
from functions import ExperimentResult  # noqa: E402
from utils import (  # noqa: E402
    DEFAULT_BERT_MODEL,
    DEFAULT_GPT2_MODEL,
    DEFAULT_MAX_LENGTH,
    ModelType,
    build_label_vocab,
    default_bin_dir,
    default_dataset_dir,
    load_atis_splits,
)


MODEL_CHOICES = ("gpt2", "gpt2-medium", "bert-base", "bert-large")
DEFAULT_MODELS = MODEL_CHOICES
DEFAULT_SWEEP_ORDER = (
    "learning_rate",
    "dropout",
    "weight_decay",
    "warmup_ratio",
    "slot_loss_weight",
    "max_length",
)
SWEEPS: dict[str, list[Any]] = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "weight_decay": [0.0, 0.01, 0.05, 0.1],
    "warmup_ratio": [0.0, 0.06, 0.1],
    "slot_loss_weight": [0.75, 1.0, 1.5, 2.0],
    "max_length": [64, 128],
}


@dataclass(frozen=True)
class TrialSummary:
    """Aggregate metrics for one hyperparameter trial."""

    model_key: str
    model_type: ModelType
    pretrained_model_name: str
    sweep: str
    value: Any
    config: FineTuningConfig
    runs: int
    seeds: str
    dev_slot_f1_mean: float
    dev_slot_f1_std: float
    dev_intent_accuracy_mean: float
    dev_intent_accuracy_std: float
    test_slot_f1_mean: float
    test_slot_f1_std: float
    test_intent_accuracy_mean: float
    test_intent_accuracy_std: float
    best_checkpoint_path: str


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sequential fine-tuning."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(DEFAULT_MODELS))
    parser.add_argument("--sweeps", nargs="+", choices=tuple(SWEEPS), default=list(DEFAULT_SWEEP_ORDER))
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=default_bin_dir())
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--intent-loss-weight", type=float, default=1.0)
    parser.add_argument("--slot-loss-weight", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--results-file", type=Path, default=Path("part_b_sequential_tuning_results.csv"))
    parser.add_argument("--summary-file", type=Path, default=Path("part_b_sequential_tuning_summary.csv"))
    return parser.parse_args()


def model_spec(model_key: str) -> tuple[ModelType, str]:
    """Map the public model key to the model family and Hugging Face name."""

    if model_key == "gpt2":
        return "gpt2", DEFAULT_GPT2_MODEL
    if model_key == "gpt2-medium":
        return "gpt2", GPT2_MEDIUM_MODEL
    if model_key == "bert-base":
        return "bert", DEFAULT_BERT_MODEL
    if model_key == "bert-large":
        return "bert", BERT_LARGE_MODEL
    raise ValueError(f"Unsupported model key: {model_key}")


def base_config(args: argparse.Namespace) -> FineTuningConfig:
    """Build the initial fine-tuning configuration from CLI defaults."""

    return FineTuningConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
        seed=args.split_seed,
        dropout=args.dropout,
        intent_loss_weight=args.intent_loss_weight,
        slot_loss_weight=args.slot_loss_weight,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )


def config_with_value(config: FineTuningConfig, sweep: str, value: Any) -> FineTuningConfig:
    """Return a config with one tuned hyperparameter changed."""

    if sweep == "learning_rate":
        return replace(config, learning_rate=float(value))
    if sweep in {"dropout", "weight_decay", "warmup_ratio", "slot_loss_weight"}:
        return replace(config, **{sweep: float(value)})
    if sweep in {"batch_size", "max_length"}:
        return replace(config, **{sweep: int(value)})
    raise ValueError(f"Unsupported sweep: {sweep}")


def safe_value(value: Any) -> str:
    """Format a hyperparameter value for filenames."""

    return str(value).replace(".", "p").replace("-", "m").replace("/", "_")


def trial_checkpoint_dir(output_dir: Path, model_key: str, sweep: str, value: Any) -> Path:
    """Return the output folder for checkpoints created by one trial."""

    return output_dir / "sequential_checkpoints" / model_key / f"{sweep}_{safe_value(value)}"


def summarize_trial(
    model_key: str,
    model_type: ModelType,
    pretrained_model_name: str,
    sweep: str,
    value: Any,
    config: FineTuningConfig,
    results: list[ExperimentResult],
) -> TrialSummary:
    """Aggregate repeated seed runs for a trial."""

    if not results:
        raise ValueError("Cannot summarize an empty trial")

    best_result = max(results, key=lambda result: (result.best_dev_slot_f1, result.best_dev_intent_accuracy))
    dev_slot = [result.best_dev_slot_f1 for result in results]
    dev_intent = [result.best_dev_intent_accuracy for result in results]
    test_slot = [result.test_slot_f1 for result in results]
    test_intent = [result.test_intent_accuracy for result in results]
    seeds = ",".join(str(result.seed) for result in results)

    return TrialSummary(
        model_key=model_key,
        model_type=model_type,
        pretrained_model_name=pretrained_model_name,
        sweep=sweep,
        value=value,
        config=config,
        runs=len(results),
        seeds=seeds,
        dev_slot_f1_mean=mean(dev_slot),
        dev_slot_f1_std=pstdev(dev_slot) if len(dev_slot) > 1 else 0.0,
        dev_intent_accuracy_mean=mean(dev_intent),
        dev_intent_accuracy_std=pstdev(dev_intent) if len(dev_intent) > 1 else 0.0,
        test_slot_f1_mean=mean(test_slot),
        test_slot_f1_std=pstdev(test_slot) if len(test_slot) > 1 else 0.0,
        test_intent_accuracy_mean=mean(test_intent),
        test_intent_accuracy_std=pstdev(test_intent) if len(test_intent) > 1 else 0.0,
        best_checkpoint_path=best_result.checkpoint_path,
    )


def summary_row(summary: TrialSummary, step: int | None = None) -> dict[str, Any]:
    """Convert a trial summary to a CSV row."""

    row = {
        "model": summary.model_key,
        "model_type": summary.model_type,
        "pretrained_model_name": summary.pretrained_model_name,
        "sweep": summary.sweep,
        "value": summary.value,
        "max_length": summary.config.max_length,
        "batch_size": summary.config.batch_size,
        "eval_batch_size": summary.config.eval_batch_size,
        "epochs": summary.config.epochs,
        "patience": summary.config.patience,
        "learning_rate": summary.config.learning_rate,
        "dropout": summary.config.dropout,
        "intent_loss_weight": summary.config.intent_loss_weight,
        "slot_loss_weight": summary.config.slot_loss_weight,
        "warmup_ratio": summary.config.warmup_ratio,
        "weight_decay": summary.config.weight_decay,
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
    if step is not None:
        row = {"step": step, **row}
    return row


def append_csv_row(output_path: Path, row: dict[str, Any]) -> None:
    """Append a row to a CSV file, creating the header when needed."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def run_trial(
    model_key: str,
    model_type: ModelType,
    pretrained_model_name: str,
    sweep: str,
    value: Any,
    config: FineTuningConfig,
    splits,
    label_vocab,
    run_seeds: list[int],
) -> TrialSummary:
    """Run one hyperparameter value, optionally repeated across seeds."""

    results: list[ExperimentResult] = []
    for seed in run_seeds:
        run_config = replace(
            config,
            seed=seed,
            output_dir=trial_checkpoint_dir(config.output_dir, model_key, sweep, value) / f"seed_{seed}",
        )
        print(
            f"\n{model_key} | {sweep}={value} | seed={seed} | "
            f"lr={run_config.learning_rate} batch={run_config.batch_size} "
            f"dropout={run_config.dropout} wd={run_config.weight_decay}"
        )
        results.append(fine_tune_model(model_type, pretrained_model_name, splits, label_vocab, run_config))

    return summarize_trial(model_key, model_type, pretrained_model_name, sweep, value, config, results)


def is_better(candidate: TrialSummary, current_best: TrialSummary | None) -> bool:
    """Rank tuning trials by dev slot F1 first, then intent accuracy."""

    if current_best is None:
        return True
    return (
        candidate.dev_slot_f1_mean,
        candidate.dev_intent_accuracy_mean,
    ) > (
        current_best.dev_slot_f1_mean,
        current_best.dev_intent_accuracy_mean,
    )


def main() -> None:
    """Run each requested model and sweep sequence."""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_atis_splits(args.dataset_dir, seed=args.split_seed)
    label_vocab = build_label_vocab(splits)

    for model_key in args.models:
        model_type, pretrained_model_name = model_spec(model_key)
        current_config = base_config(args)

        print(f"\nStarting sequential tuning for {model_key} ({pretrained_model_name})")
        for step, sweep in enumerate(args.sweeps, start=1):
            print(f"\nStep {step}: sweeping {sweep}")
            step_best: TrialSummary | None = None
            step_best_config: FineTuningConfig | None = None

            for value in SWEEPS[sweep]:
                trial_config = config_with_value(current_config, sweep, value)
                summary = run_trial(
                    model_key=model_key,
                    model_type=model_type,
                    pretrained_model_name=pretrained_model_name,
                    sweep=sweep,
                    value=value,
                    config=trial_config,
                    splits=splits,
                    label_vocab=label_vocab,
                    run_seeds=args.run_seeds,
                )
                append_csv_row(args.output_dir / args.results_file, summary_row(summary))
                print(
                    f"{model_key} {sweep}={value} | "
                    f"dev slot F1={summary.dev_slot_f1_mean:.4f} | "
                    f"dev intent acc={summary.dev_intent_accuracy_mean:.4f} | "
                    f"test slot F1={summary.test_slot_f1_mean:.4f}"
                )

                if is_better(summary, step_best):
                    step_best = summary
                    step_best_config = trial_config

            if step_best is None or step_best_config is None:
                raise RuntimeError(f"No trials completed for {model_key} sweep {sweep}")

            current_config = step_best_config
            append_csv_row(args.output_dir / args.summary_file, summary_row(step_best, step=step))
            print(
                f"Best after {sweep}: {step_best.value} "
                f"(dev slot F1={step_best.dev_slot_f1_mean:.4f}, "
                f"dev intent acc={step_best.dev_intent_accuracy_mean:.4f})"
            )

        print(f"\nFinal config for {model_key}: {current_config}")


if __name__ == "__main__":
    main()
