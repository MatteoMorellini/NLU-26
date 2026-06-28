"""Sequential multi-seed hyperparameter tuning for ATIS part A.

The search order is learning_rate -> d_model -> n_heads -> num_layers -> ff_dim.
Each sweep starts from the best configuration found by the previous sweep. Every
candidate value is run with several seeds, and the value with the highest average
development Slot F1 is carried into the next sweep.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from functions import (
    AggregateExperimentResult,
    ExperimentResult,
    summarize_repeated_runs,
    write_aggregate_results,
    write_experiment_results,
)
from model import ModelConfig
from tuning import format_lr, print_summary, run_repeated_experiment
from utils import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZER_NAME,
    build_label_vocab,
    default_bin_dir,
    default_dataset_dir,
    get_gpt2_tokenizer,
    load_atis_splits,
)


DEFAULT_ORDER = ("learning_rate", "d_model", "n_heads", "num_layers", "ff_dim")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=default_bin_dir())
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Seed used for the train/dev split.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--initialization", choices=["gpt2", "lm", "xavier"], default="gpt2")
    parser.add_argument("--lr-values", type=float, nargs="+", default=[1e-3, 5e-4, 1e-4])
    parser.add_argument("--d-model-values", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--n-head-values", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--num-layer-values", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--ff-dim-values", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base-learning-rate", type=float, default=5e-4)
    parser.add_argument("--base-d-model", type=int, default=128)
    parser.add_argument("--base-n-heads", type=int, default=4)
    parser.add_argument("--base-num-layers", type=int, default=2)
    parser.add_argument("--base-ff-dim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    """Run sequential tuning and save per-seed and aggregate results."""

    args = parse_args()
    tokenizer = get_gpt2_tokenizer(args.tokenizer)
    splits = load_atis_splits(args.dataset_dir, seed=args.seed)
    label_vocab = build_label_vocab(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current_config = ModelConfig(
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
    current_learning_rate = args.base_learning_rate

    all_results: list[ExperimentResult] = []
    all_summaries: list[AggregateExperimentResult] = []
    winners: list[AggregateExperimentResult] = []

    for step, sweep_name in enumerate(DEFAULT_ORDER, start=1):
        print(f"\nSequential tuning step {step}: {sweep_name}")
        print(f"Current config: {current_config}")
        print(f"Current learning rate: {current_learning_rate}")

        step_summaries: list[AggregateExperimentResult] = []
        for value in values_for_sweep(args, sweep_name):
            candidate_config = current_config
            candidate_learning_rate = current_learning_rate

            if sweep_name == "learning_rate":
                candidate_learning_rate = float(value)
                trial_name = f"seq_lr_{format_lr(candidate_learning_rate)}"
            else:
                candidate_config = replace(current_config, **{sweep_name: value})
                trial_name = f"seq_{sweep_name}_{value}"

            if candidate_config.d_model % candidate_config.n_heads != 0:
                print(
                    f"Skipping {trial_name}: d_model={candidate_config.d_model} "
                    f"is not divisible by n_heads={candidate_config.n_heads}"
                )
                continue

            repeated_results = run_repeated_experiment(
                name=trial_name,
                model_config=candidate_config,
                learning_rate=candidate_learning_rate,
                args=args,
                splits=splits,
                label_vocab=label_vocab,
                tokenizer=tokenizer,
            )
            summary = summarize_repeated_runs(trial_name, repeated_results)
            print_summary(summary)
            all_results.extend(repeated_results)
            all_summaries.append(summary)
            step_summaries.append(summary)

        if not step_summaries:
            raise RuntimeError(f"No valid trials completed for sweep {sweep_name!r}")

        winner = max(step_summaries, key=lambda result: result.dev_slot_f1_mean)
        winners.append(winner)
        current_learning_rate = winner.learning_rate
        current_config = replace(
            current_config,
            d_model=winner.d_model,
            n_heads=winner.n_heads,
            num_layers=winner.num_layers,
            ff_dim=winner.ff_dim,
            dropout=winner.dropout,
        )
        print(
            f"Best after {sweep_name}: {winner.name} "
            f"dev_slot_f1={winner.dev_slot_f1_mean:.4f} +- {winner.dev_slot_f1_std:.4f}"
        )

    write_experiment_results(all_results, args.output_dir / "sequential_tuning_results.csv")
    write_aggregate_results(all_summaries, args.output_dir / "sequential_tuning_all_summary.csv")
    write_aggregate_results(winners, args.output_dir / "sequential_tuning_winners.csv")

    final_winner = winners[-1]
    shutil.copyfile(final_winner.best_checkpoint_path, args.output_dir / "best_model.pt")
    print("\nFinal sequential best config:")
    print(current_config)
    print(f"Learning rate: {current_learning_rate}")
    print(f"Best checkpoint copied to: {args.output_dir / 'best_model.pt'}")
    print(f"Slot F1 {final_winner.test_slot_f1_mean:.3f} +- {final_winner.test_slot_f1_std:.3f}")
    print(
        f"Intent Acc {final_winner.test_intent_accuracy_mean:.3f} "
        f"+- {final_winner.test_intent_accuracy_std:.3f}"
    )


def values_for_sweep(args: argparse.Namespace, sweep_name: str) -> list[float | int]:
    """Return candidate values for a sequential sweep."""

    if sweep_name == "learning_rate":
        return list(args.lr_values)
    if sweep_name == "d_model":
        return list(args.d_model_values)
    if sweep_name == "n_heads":
        return list(args.n_head_values)
    if sweep_name == "num_layers":
        return list(args.num_layer_values)
    if sweep_name == "ff_dim":
        return list(args.ff_dim_values)
    raise ValueError(f"Unknown sweep: {sweep_name}")


if __name__ == "__main__":
    main()
