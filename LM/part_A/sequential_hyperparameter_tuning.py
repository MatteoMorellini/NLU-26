"""Run sequential hyperparameter tuning for Part A.

The search order is weight_decay -> lr_schedule -> warmup_steps.
Each sweep starts from the best configuration found by the previous sweep,
using best dev perplexity as the selection criterion.
"""

from __future__ import annotations

import csv
from argparse import ArgumentParser, Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any


DEFAULT_ORDER = ("weight_decay", "lr_schedule", "warmup_steps")
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_LR_SCHEDULE = "none"
DEFAULT_WARMUP_STEPS = 0


def load_training_modules() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Import training modules after CLI parsing so --help has no torch dependency."""

    from hyperparameter_tuning import (  # noqa: PLC0415
        BASE_CONFIG,
        SWEEPS,
        append_result,
        count_trainable_weights,
    )
    from main import ExperimentConfig, run_experiment  # noqa: PLC0415
    from utils import load_tokenizer  # noqa: PLC0415

    return (
        BASE_CONFIG,
        SWEEPS,
        append_result,
        count_trainable_weights,
        ExperimentConfig,
        run_experiment,
        load_tokenizer,
    )


def parse_args() -> Namespace:
    """Parse command-line options for the sequential search."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of epochs for each trial.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience for each trial.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sequential_tuning_results.csv"),
        help="CSV file where every trial result is appended.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("sequential_tuning_summary.csv"),
        help="CSV file where the best config after each sweep is written.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay used by AdamW.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "linear", "cosine", "inverse_sqrt"),
        default=DEFAULT_LR_SCHEDULE,
        help="Step-wise learning-rate schedule used during training.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Number of optimizer steps used for LR warmup.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Max gradient norm. Leave unset to disable clipping.",
    )
    return parser.parse_args()


def config_for_trial(
    base_config: Any,
    sweep_name: str,
    trial: dict[str, Any],
    epochs: int,
    patience: int,
) -> Any:
    """Create a trial config by changing one sweep option on the current best config."""

    tuned_value = trial[sweep_name]
    trial_values = dict(trial)
    if sweep_name == "d_model":
        trial_values["ff_dim"] = 4 * int(tuned_value)

    name = f"seq_{sweep_name}_{tuned_value}"
    return replace(
        base_config,
        **trial_values,
        name=name,
        n_epochs=epochs,
        patience=patience,
    )


def append_summary(
    output_path: Path,
    step: int,
    sweep_name: str,
    config: Any,
    best_dev_ppl: float,
    test_ppl: float,
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "step": step,
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
                "best_dev_ppl": f"{best_dev_ppl:.4f}",
                "test_ppl": f"{test_ppl:.4f}",
            }
        )


def main() -> None:
    """Run the sequential sweep and keep the best config between steps."""

    args = parse_args()
    (
        base_config,
        sweeps,
        append_result,
        count_trainable_weights,
        _experiment_config,
        run_experiment,
        load_tokenizer,
    ) = load_training_modules()
    vocab_size = len(load_tokenizer())
    current_best_config = replace(
        base_config,
        name="seq_baseline",
        n_epochs=args.epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
    )

    for step, sweep_name in enumerate(DEFAULT_ORDER, start=1):
        print(f"\nSequential tuning step {step}: {sweep_name}")
        print(f"Starting from config: {current_best_config}")

        step_best_config: Any | None = None
        step_best_dev_ppl = float("inf")
        step_best_test_ppl = float("inf")

        for trial in sweeps[sweep_name]:
            config = config_for_trial(
                base_config=current_best_config,
                sweep_name=sweep_name,
                trial=trial,
                epochs=args.epochs,
                patience=args.patience,
            )

            if config.d_model % config.n_heads != 0:
                print(
                    f"Skipping {config.name}: d_model={config.d_model} "
                    f"is not divisible by n_heads={config.n_heads}"
                )
                continue

            trainable_weights = count_trainable_weights(config, vocab_size)
            print(f"Tuning trial: {config.name}")
            print(f"Trainable weights before tuning trial: {trainable_weights:,}")

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

            if dev_ppl < step_best_dev_ppl:
                step_best_config = config
                step_best_dev_ppl = dev_ppl
                step_best_test_ppl = test_ppl

        if step_best_config is None:
            raise RuntimeError(f"No valid trials completed for sweep {sweep_name!r}")

        current_best_config = step_best_config
        append_summary(
            output_path=args.summary_output,
            step=step,
            sweep_name=sweep_name,
            config=current_best_config,
            best_dev_ppl=step_best_dev_ppl,
            test_ppl=step_best_test_ppl,
        )
        print(
            f"Best after {sweep_name}: {current_best_config.name} "
            f"dev_ppl={step_best_dev_ppl:.2f} test_ppl={step_best_test_ppl:.2f}"
        )

    print("\nFinal sequential best config:")
    print(current_best_config)


if __name__ == "__main__":
    main()
