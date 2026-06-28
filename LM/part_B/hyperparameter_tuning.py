"""Run rank and alpha sweeps for Part B LoRA fine-tuning."""

from __future__ import annotations

import csv
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


PART_DIR = Path(__file__).resolve().parent
os.chdir(PART_DIR)
CHECKPOINT_DIR = PART_DIR / "bin"
RESULTS_PATH = CHECKPOINT_DIR / "lora_sweep_results.csv"
DEFAULT_EXPERIMENT = "lora_r8_a8"
RANK_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16)
EXPERIMENT_SPECS: tuple[dict[str, Any], ...] = tuple(
    {
        "name": f"lora_r{rank}_a{2 * rank}",
        "rank": rank,
        "alpha": float(2 * rank),
        "learning_rate": 5e-4,
    }
    for rank in RANK_SWEEP
)
EXPERIMENTS_BY_NAME = {spec["name"]: spec for spec in EXPERIMENT_SPECS}


def load_training_modules() -> tuple[Any, Any, Any]:
    """Import training modules after CLI parsing so --help has no torch dependency."""

    from main import ExperimentConfig, resolve_lora_targets, run_experiment  # noqa: PLC0415

    return ExperimentConfig, resolve_lora_targets, run_experiment


def build_experiments(experiment_config: Any) -> tuple[Any, ...]:
    """Create typed experiment configs from the lightweight sweep specs."""

    return tuple(
        experiment_config(
            name=spec["name"],
            rank=spec["rank"],
            alpha=spec["alpha"],
            learning_rate=spec["learning_rate"],
        )
        for spec in EXPERIMENT_SPECS
    )


def parse_args() -> Namespace:
    """Parse command-line options for selecting the LoRA tuning run."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(*EXPERIMENTS_BY_NAME.keys(), "all"),
        default=DEFAULT_EXPERIMENT,
        help="Named experiment to run. Use 'all' for the full rank/alpha sweep.",
    )
    parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="Hugging Face model name or local path to use as the GPT-2 base.",
    )
    parser.add_argument(
        "--lora-targets",
        choices=("lab", "paper"),
        default="lab",
        help=(
            "Use 'lab' for query/key/value adapters, or 'paper' for the "
            "query/value setup used in the LoRA GPT experiments."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="CSV file where LoRA sweep metrics are written.",
    )
    return parser.parse_args()


def write_results(results: list[Any], path: Path = RESULTS_PATH) -> None:
    """Write metrics for report-ready comparison across LoRA hyperparameters."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "lora_targets",
                "rank",
                "alpha",
                "learning_rate",
                "pretrained_dev_ppl",
                "best_dev_ppl",
                "test_ppl",
                "total_parameters",
                "trainable_parameters",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "lora_targets": result.lora_targets,
                    "rank": result.rank,
                    "alpha": result.alpha,
                    "learning_rate": result.learning_rate,
                    "pretrained_dev_ppl": result.pretrained_dev_ppl,
                    "best_dev_ppl": result.best_dev_ppl,
                    "test_ppl": result.test_ppl,
                    "total_parameters": result.total_parameters,
                    "trainable_parameters": result.trainable_parameters,
                    "checkpoint_path": str(result.checkpoint_path),
                }
            )


def select_experiments(experiment: str, experiments: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return the requested rank/alpha configurations."""

    if experiment == "all":
        return experiments
    experiments_by_name = {config.name: config for config in experiments}
    return (experiments_by_name[experiment],)


def print_best_result(results: list[Any]) -> None:
    """Print the best rank/alpha setting by dev perplexity."""

    if not results:
        raise RuntimeError("No LoRA experiments were run.")

    best = min(results, key=lambda result: result.best_dev_ppl)
    print(
        "Best LoRA configuration: "
        f"name={best.name} rank={best.rank} alpha={best.alpha:g} "
        f"dev_ppl={best.best_dev_ppl:.2f} test_ppl={best.test_ppl:.2f}"
    )


def run_tuning(
    experiments: tuple[Any, ...],
    model_name: str,
    lora_targets: Any,
    run_experiment: Any,
) -> list[Any]:
    """Run the selected LoRA configurations."""

    return [
        run_experiment(
            config,
            model_name=model_name,
            lora_targets=lora_targets,
        )
        for config in experiments
    ]


def main() -> None:
    """Train selected LoRA configurations and report the best rank/alpha pair."""

    args = parse_args()
    experiment_config, resolve_lora_targets, run_experiment = load_training_modules()
    experiments = build_experiments(experiment_config)
    results = run_tuning(
        experiments=select_experiments(args.experiment, experiments),
        model_name=args.model_name,
        lora_targets=resolve_lora_targets(args.lora_targets),
        run_experiment=run_experiment,
    )
    write_results(results, path=args.output)
    print_best_result(results)
    print(f"Saved LoRA comparison metrics to: {args.output}")


if __name__ == "__main__":
    main()
