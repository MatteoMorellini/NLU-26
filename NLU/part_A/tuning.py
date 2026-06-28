"""Run baseline and incremental hyperparameter tuning for ATIS part A."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

from functions import (
    AggregateExperimentResult,
    ExperimentResult,
    TrainConfig,
    run_training,
    summarize_repeated_runs,
    write_aggregate_results,
    write_experiment_results,
)
from model import ModelConfig
from utils import (
    DEVICE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOKENIZER_NAME,
    build_dataloaders,
    build_label_vocab,
    default_bin_dir,
    default_dataset_dir,
    DatasetSplits,
    get_gpt2_tokenizer,
    LabelVocab,
    load_atis_splits,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Tune GPT-2 ATIS intent/slot model.")
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
    parser.add_argument("--base-d-model", type=int, default=128)
    parser.add_argument("--base-n-heads", type=int, default=4)
    parser.add_argument("--base-num-layers", type=int, default=2)
    parser.add_argument("--base-ff-dim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    """Run tuning and save checkpoints/results."""

    args = parse_args()
    tokenizer = get_gpt2_tokenizer(args.tokenizer)
    splits = load_atis_splits(args.dataset_dir, seed=args.seed)
    label_vocab = build_label_vocab(splits)

    base_model_config = ModelConfig(
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    summaries: list[AggregateExperimentResult] = []

    for lr in args.lr_values:
        name = f"baseline_lr_{format_lr(lr)}"
        repeated_results = run_repeated_experiment(
            name=name,
            model_config=base_model_config,
            learning_rate=lr,
            args=args,
            splits=splits,
            label_vocab=label_vocab,
            tokenizer=tokenizer,
        )
        summary = summarize_repeated_runs(name, repeated_results)
        print_summary(summary)
        results.extend(repeated_results)
        summaries.append(summary)

    best_summary = max(summaries, key=lambda result: result.dev_slot_f1_mean)
    best_lr = best_summary.learning_rate
    current_config = base_model_config
    tuning_specs = [
        ("d_model", args.d_model_values),
        ("n_heads", args.n_head_values),
        ("num_layers", args.num_layer_values),
        ("ff_dim", args.ff_dim_values),
    ]

    for field_name, values in tuning_specs:
        field_summaries: list[AggregateExperimentResult] = []
        for value in values:
            candidate_config = replace(current_config, **{field_name: value})
            if candidate_config.d_model % candidate_config.n_heads != 0:
                print(f"Skipping {field_name}={value}: d_model must be divisible by n_heads")
                continue

            name = f"tune_{field_name}_{value}"
            repeated_results = run_repeated_experiment(
                name=name,
                model_config=candidate_config,
                learning_rate=best_lr,
                args=args,
                splits=splits,
                label_vocab=label_vocab,
                tokenizer=tokenizer,
            )
            summary = summarize_repeated_runs(name, repeated_results)
            print_summary(summary)
            field_summaries.append(summary)
            results.extend(repeated_results)
            summaries.append(summary)

        if field_summaries:
            best_for_field = max(field_summaries, key=lambda result: result.dev_slot_f1_mean)
            current_config = replace(
                current_config,
                d_model=best_for_field.d_model,
                n_heads=best_for_field.n_heads,
                num_layers=best_for_field.num_layers,
                ff_dim=best_for_field.ff_dim,
            )
            best_summary = max([best_summary, best_for_field], key=lambda result: result.dev_slot_f1_mean)

    write_experiment_results(results, args.output_dir / "tuning_results.csv")
    write_aggregate_results(summaries, args.output_dir / "tuning_results_summary.csv")
    shutil.copyfile(best_summary.best_checkpoint_path, args.output_dir / "best_model.pt")
    print(f"Best repeated-run config: {best_summary.name}")
    print(f"Best checkpoint copied to: {args.output_dir / 'best_model.pt'}")
    print(f"Slot F1 {best_summary.test_slot_f1_mean:.3f} +- {best_summary.test_slot_f1_std:.3f}")
    print(f"Intent Acc {best_summary.test_intent_accuracy_mean:.3f} +- {best_summary.test_intent_accuracy_std:.3f}")


def run_repeated_experiment(
    name: str,
    model_config: ModelConfig,
    learning_rate: float,
    args: argparse.Namespace,
    splits: DatasetSplits,
    label_vocab: LabelVocab,
    tokenizer,
) -> list[ExperimentResult]:
    """Run one model configuration over all requested seeds."""

    repeated_results: list[ExperimentResult] = []
    for seed in args.seeds:
        train_loader, dev_loader, test_loader = build_dataloaders(
            splits=splits,
            label_vocab=label_vocab,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            max_length=args.max_length,
            device=DEVICE,
            seed=seed,
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
        f"Slot F1 {summary.test_slot_f1_mean:.3f} +- {summary.test_slot_f1_std:.3f}; "
        f"Intent Acc {summary.test_intent_accuracy_mean:.3f} +- {summary.test_intent_accuracy_std:.3f}"
    )


def format_lr(lr: float) -> str:
    """Format a learning rate for filenames."""

    return f"{lr:.0e}".replace("-", "m").replace("+", "")


if __name__ == "__main__":
    main()
