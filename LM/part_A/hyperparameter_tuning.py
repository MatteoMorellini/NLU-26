"""Run one-at-a-time hyperparameter sweeps for Part A.

Each sweep starts from the same baseline configuration and changes only one
model hyperparameter. Learning rate is treated as a training readjustment for
larger models, not as the tuned model hyperparameter.
"""

from __future__ import annotations

import csv
import os
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch


PART_DIR = Path(__file__).resolve().parent
os.chdir(PART_DIR)

from functions import count_trainable_parameters  # noqa: E402
from main import ExperimentConfig, run_experiment  # noqa: E402
from model import GPT2  # noqa: E402
from utils import load_tokenizer  # noqa: E402


BASE_CONFIG = ExperimentConfig(
    name="baseline",
    learning_rate=3e-4,
    d_model=512,
    n_heads=8,
    num_layers=10,
    ff_dim=3072,
    dropout=0.3,
    n_epochs=100,
    patience=3,
    seed=42,
)

SWEEPS: dict[str, tuple[dict[str, Any], ...]] = {
    # "d_model": (
    #     {"d_model": 256, "ff_dim": 1024, "learning_rate": 8e-4},
    #     {"d_model": 384, "ff_dim": 1536, "learning_rate": 6e-4},
    #     {"d_model": 512, "ff_dim": 2048, "learning_rate": 5e-4},
    #     {"d_model": 768, "ff_dim": 3072, "learning_rate": 5e-4},
    # ),
    # "n_heads": (
    #     {"n_heads": 2},
    #     {"n_heads": 4},
    #     {"n_heads": 8},
    #     {"n_heads": 16},
    # ),
    # "num_layers": (
    #     {"num_layers": 6, "learning_rate": 5e-4},
    #     {"num_layers": 8, "learning_rate": 5e-4},
    #     {"num_layers": 10, "learning_rate": 3e-4},
    # ),
    # "ff_dim": (
    #     {"ff_dim": 1024, "learning_rate": 8e-4},
    #     {"ff_dim": 2048, "learning_rate": 7e-4},
    #     {"ff_dim": 3072, "learning_rate": 5e-4},
    # ),
    # "dropout": (
    #     {"dropout": 0.0},
    #     {"dropout": 0.1},
    #     {"dropout": 0.2},
    #     {"dropout": 0.3},
    #     {"dropout": 0.4},
    # ),
    "weight_decay": (
        {"weight_decay": 0.0},
        {"weight_decay": 0.001},
        {"weight_decay": 0.01},
        {"weight_decay": 0.05},
        {"weight_decay": 0.1},
    ),
    "lr_schedule": (
        {"lr_schedule": "none"},
        {"lr_schedule": "linear"},
        {"lr_schedule": "cosine"},
        {"lr_schedule": "inverse_sqrt"},
    ),
    "warmup_steps": (
        {"warmup_steps": 0},
        {"warmup_steps": 50},
        {"warmup_steps": 100},
        {"warmup_steps": 200},
        {"warmup_steps": 500},
    ),
}
SEQUENTIAL_SWEEP_ORDER = ("weight_decay", "lr_schedule", "warmup_steps")


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
    parser.add_argument(
        "--epochs",
        type=int,
        default=BASE_CONFIG.n_epochs,
        help="Maximum number of epochs for each trial.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=BASE_CONFIG.patience,
        help="Early-stopping patience for each trial.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tuning_results.csv"),
        help="CSV file where trial results are appended.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=BASE_CONFIG.weight_decay,
        help="Weight decay used by AdamW.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "linear", "cosine", "inverse_sqrt"),
        default=BASE_CONFIG.lr_schedule,
        help="Step-wise learning-rate schedule used during training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=BASE_CONFIG.warmup_steps,
        help="Number of optimizer steps used for LR warmup.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=BASE_CONFIG.gradient_clip,
        help="Max gradient norm. Leave unset to disable clipping.",
    )
    return parser.parse_args()


def config_with_training_flags(config: ExperimentConfig, args: Namespace) -> ExperimentConfig:
    """Apply training schedule, regularization, and clipping flags to a config."""

    return replace(
        config,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
    )


def config_for_trial(
    sweep_name: str,
    trial: dict[str, Any],
    epochs: int,
    patience: int,
    base_config: ExperimentConfig = BASE_CONFIG,
    name_prefix: str = "tune",
) -> ExperimentConfig:
    """Create a named experiment config for one sweep trial."""

    tuned_value = trial[sweep_name]
    trial_values = dict(trial)
    if sweep_name == "d_model":
        trial_values["ff_dim"] = 4 * int(tuned_value)

    name = f"{name_prefix}_{sweep_name}_{tuned_value}"
    return replace(
        base_config,
        **trial_values,
        name=name,
        n_epochs=epochs,
        patience=patience,
    )


def count_trainable_weights(config: ExperimentConfig, vocab_size: int) -> int:
    """Count trainable weights for a trial config without allocating tensors."""

    with torch.device("meta"):
        model = GPT2(
            vocab_size=vocab_size,
            pos_emb_size=config.pos_emb_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            tie_weights=config.tie_weights,
        )
    return count_trainable_parameters(model)


def append_result(output_path: Path, sweep_name: str, config: ExperimentConfig, dev_ppl: float, test_ppl: float) -> None:
    """Append one trial result to the tuning CSV file."""

    fieldnames = [
        "sweep",
        "name",
        "d_model",
        "n_heads",
        "num_layers",
        "ff_dim",
        "dropout",
        "tie_weights",
        "weight_decay",
        "lr_schedule",
        "warmup_steps",
        "gradient_clip",
        "learning_rate",
        "seed",
        "best_dev_ppl",
        "test_ppl",
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "sweep": sweep_name,
                "name": config.name,
                "d_model": config.d_model,
                "n_heads": config.n_heads,
                "num_layers": config.num_layers,
                "ff_dim": config.ff_dim,
                "dropout": config.dropout,
                "tie_weights": config.tie_weights,
                "weight_decay": config.weight_decay,
                "lr_schedule": config.lr_schedule,
                "warmup_steps": config.warmup_steps,
                "gradient_clip": config.gradient_clip,
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "best_dev_ppl": f"{dev_ppl:.4f}",
                "test_ppl": f"{test_ppl:.4f}",
            }
        )


def run_trial(
    sweep_name: str,
    trial: dict[str, Any],
    base_config: ExperimentConfig,
    args: Namespace,
    vocab_size: int,
    name_prefix: str = "tune",
) -> tuple[ExperimentConfig, float, float]:
    """Run one trial, persist its metrics, and return config/dev/test results."""

    config = config_for_trial(
        sweep_name=sweep_name,
        trial=trial,
        epochs=args.epochs,
        patience=args.patience,
        base_config=base_config,
        name_prefix=name_prefix,
    )
    trainable_weights = count_trainable_weights(config, vocab_size)
    print(f"Tuning step: {config.name}")
    print(f"Trainable weights before tuning step: {trainable_weights:,}")
    result = run_experiment(config)
    dev_ppl = result.best_dev.perplexity
    test_ppl = result.test.perplexity
    append_result(
        output_path=args.output,
        sweep_name=sweep_name,
        config=config,
        dev_ppl=dev_ppl,
        test_ppl=test_ppl,
    )
    return config, dev_ppl, test_ppl


def run_sequential_sweeps(args: Namespace, vocab_size: int) -> ExperimentConfig:
    """Tune sweeps in order, carrying each best config into the next sweep."""

    current_base = replace(
        BASE_CONFIG,
        n_epochs=args.epochs,
        patience=args.patience,
    )
    current_base = config_with_training_flags(current_base, args)
    for step_idx, sweep_name in enumerate(SEQUENTIAL_SWEEP_ORDER, start=1):
        best_config: Optional[ExperimentConfig] = None
        best_dev_ppl = float("inf")
        best_test_ppl = float("inf")

        print(f"Sequential sweep {step_idx}/{len(SEQUENTIAL_SWEEP_ORDER)}: {sweep_name}")
        print(f"Base config for this sweep: {current_base}")

        for trial in SWEEPS[sweep_name]:
            config, dev_ppl, test_ppl = run_trial(
                sweep_name=sweep_name,
                trial=trial,
                base_config=current_base,
                args=args,
                vocab_size=vocab_size,
                name_prefix=f"seq{step_idx}",
            )
            if dev_ppl < best_dev_ppl:
                best_config = config
                best_dev_ppl = dev_ppl
                best_test_ppl = test_ppl

        if best_config is None:
            raise RuntimeError(f"No trials were run for sweep {sweep_name}")

        current_base = replace(best_config, name=f"best_after_{sweep_name}")
        print(
            f"Best after {sweep_name}: {current_base} "
            f"dev_ppl={best_dev_ppl:.4f} test_ppl={best_test_ppl:.4f}"
        )

    return current_base


def main() -> None:
    """Run selected one-at-a-time sweeps and write their results."""

    args = parse_args()
    vocab_size = len(load_tokenizer())

    if args.sweep == "sequential":
        final_config = run_sequential_sweeps(args, vocab_size)
        print(f"Final selected config: {final_config}")
        return

    selected_sweeps = SWEEPS if args.sweep == "all" else {args.sweep: SWEEPS[args.sweep]}
    base_config = config_with_training_flags(BASE_CONFIG, args)
    for sweep_name, trials in selected_sweeps.items():
        for trial in trials:
            run_trial(
                sweep_name=sweep_name,
                trial=trial,
                base_config=base_config,
                args=args,
                vocab_size=vocab_size,
            )


if __name__ == "__main__":
    main()
