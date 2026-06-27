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
from typing import Any


PART_DIR = Path(__file__).resolve().parent
os.chdir(PART_DIR)

from main import ExperimentConfig, run_experiment  # noqa: E402


BASE_CONFIG = ExperimentConfig(
    name="baseline",
    learning_rate=5e-4,
    d_model=512,
    n_heads=1,
    num_layers=1,
    ff_dim=1024,
    dropout=0.0,
    n_epochs=100,
    patience=3,
    seed=42,
)

SWEEPS: dict[str, tuple[dict[str, Any], ...]] = {
    "d_model": (
        {"d_model": 256, "ff_dim": 1024, "learning_rate": 8e-4},
        {"d_model": 384, "ff_dim": 1536, "learning_rate": 6e-4},
        {"d_model": 512, "ff_dim": 2048, "learning_rate": 5e-4},
        {"d_model": 768, "ff_dim": 3072, "learning_rate": 5e-4},
    ),
    "n_heads": (
        {"n_heads": 2},
        {"n_heads": 4},
        {"n_heads": 8},
        {"n_heads": 16},
    ),
    "num_layers": (
        {"num_layers": 6, "learning_rate": 7e-4},
        {"num_layers": 8, "learning_rate": 5e-4},
        {"num_layers": 10, "learning_rate": 3e-4},
    ),
    "ff_dim": (
        {"ff_dim": 1024, "learning_rate": 8e-4},
        {"ff_dim": 2048, "learning_rate": 7e-4},
        {"ff_dim": 3072, "learning_rate": 5e-4},
    ),
}


def parse_args() -> Namespace:
    """Parse command-line options for choosing a sweep."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        choices=(*SWEEPS.keys(), "all"),
        default="all",
        help="Hyperparameter to tune. Use 'all' to run every one-at-a-time sweep.",
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
    return parser.parse_args()


def config_for_trial(sweep_name: str, trial: dict[str, Any], epochs: int, patience: int) -> ExperimentConfig:
    """Create a named experiment config for one sweep trial."""

    tuned_value = trial[sweep_name]
    name = f"tune_{sweep_name}_{tuned_value}"
    return replace(
        BASE_CONFIG,
        **trial,
        name=name,
        n_epochs=epochs,
        patience=patience,
    )


def append_result(output_path: Path, sweep_name: str, config: ExperimentConfig, dev_ppl: float, test_ppl: float) -> None:
    """Append one trial result to the tuning CSV file."""

    fieldnames = [
        "sweep",
        "name",
        "d_model",
        "n_heads",
        "num_layers",
        "ff_dim",
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
                "learning_rate": config.learning_rate,
                "seed": config.seed,
                "best_dev_ppl": f"{dev_ppl:.4f}",
                "test_ppl": f"{test_ppl:.4f}",
            }
        )


def main() -> None:
    """Run selected one-at-a-time sweeps and write their results."""

    args = parse_args()
    selected_sweeps = SWEEPS if args.sweep == "all" else {args.sweep: SWEEPS[args.sweep]}

    for sweep_name, trials in selected_sweeps.items():
        for trial in trials:
            config = config_for_trial(
                sweep_name=sweep_name,
                trial=trial,
                epochs=args.epochs,
                patience=args.patience,
            )
            result = run_experiment(config)
            append_result(
                output_path=args.output,
                sweep_name=sweep_name,
                config=config,
                dev_ppl=result.best_dev.perplexity,
                test_ppl=result.test.perplexity,
            )


if __name__ == "__main__":
    main()
