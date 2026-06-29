"""Train the final Part A GPT-2 configuration with last-subtoken slot labels."""

from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path


PART_DIR = Path(__file__).resolve().parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.chdir(PART_DIR)

from functions import TrainConfig, run_training, summarize_repeated_runs, write_experiment_results  # noqa: E402
from model import ModelConfig  # noqa: E402
from tuning import print_summary  # noqa: E402
from utils import (  # noqa: E402
    DEVICE,
    build_dataloaders,
    build_label_vocab,
    default_dataset_dir,
    get_gpt2_tokenizer,
    load_atis_splits,
)


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=PART_DIR / "bin")
    parser.add_argument("--tokenizer", type=str, default="openai-community/gpt2")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[42, 101, 27])
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--initialization", choices=["gpt2", "lm", "xavier"], default="gpt2")
    parser.add_argument("--slot-subtoken-strategy", choices=["first", "last"], default="last")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_gpt2_tokenizer(args.tokenizer)
    splits = load_atis_splits(args.dataset_dir, seed=42)
    label_vocab = build_label_vocab(splits)
    model_config = ModelConfig(
        vocab_size=len(tokenizer),
        slots_size=len(label_vocab.slot2id),
        n_intents=len(label_vocab.intent2id),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pos_emb_size=args.max_length,
        d_model=384,
        n_heads=2,
        num_layers=1,
        ff_dim=1024,
        dropout=0.0,
    )

    name = "final_complete_last_subtoken"
    results = []
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
                learning_rate=args.learning_rate,
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
        results.append(result)

    write_experiment_results(results, args.output_dir / f"{name}_results.csv")
    print_summary(summarize_repeated_runs(name, results))


if __name__ == "__main__":
    main()
