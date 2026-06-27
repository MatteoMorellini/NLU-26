"""Entry point for GPT-2 LoRA fine-tuning experiments."""

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path


PART_DIR = Path(__file__).resolve().parent
CACHE_DIR = PART_DIR / "hf_cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))

import torch
from torch import nn
from torch.optim import AdamW

from functions import (
    count_total_parameters,
    count_trainable_parameters,
    fit_model,
)
from model import GPT_LoRA
from utils import IGNORE_INDEX, build_dataloaders, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("bin")
DEFAULT_EXPERIMENT = "lora_r8_a16"


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for one GPT-2 LoRA experiment."""

    name: str
    rank: int
    alpha: float
    learning_rate: float
    dropout: float = 0.05
    batch_size: int = 4
    eval_batch_size: int = 8
    max_length: int = 128
    n_epochs: int = 3
    patience: int = 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig(
        name="lora_r4_a8",
        rank=4,
        alpha=8.0,
        learning_rate=5e-4,
    ),
    ExperimentConfig(
        name=DEFAULT_EXPERIMENT,
        rank=8,
        alpha=16.0,
        learning_rate=5e-4,
    ),
    ExperimentConfig(
        name="lora_r16_a32",
        rank=16,
        alpha=32.0,
        learning_rate=3e-4,
    ),
)
EXPERIMENTS_BY_NAME = {config.name: config for config in EXPERIMENTS}


def parse_args() -> Namespace:
    """Parse command-line options for selecting the experiment to run."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=(*EXPERIMENTS_BY_NAME.keys(), "all"),
        default=DEFAULT_EXPERIMENT,
        help="Named experiment to run. Use 'all' for the rank/alpha sweep.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="Hugging Face model name or local path to use as the GPT-2 base.",
    )
    return parser.parse_args()


def run_experiment(config: ExperimentConfig, model_name: str) -> None:
    """Fine-tune and evaluate one LoRA configuration."""

    set_seed(config.seed)
    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders(
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        device=DEVICE,
        model_name=model_name,
        max_length=config.max_length,
        cache_dir=CACHE_DIR,
        seed=config.seed,
    )
    model = GPT_LoRA(
        model_name=model_name,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        cache_dir=CACHE_DIR,
    ).to(DEVICE)
    model.gpt2.config.pad_token_id = tokenizer.pad_token_id

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = CHECKPOINT_DIR / f"{config.name}.pt"

    print(f"Experiment: {config.name}")
    print(f"Device: {DEVICE}")
    print(f"Base model: {model_name}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Total parameters: {count_total_parameters(model)}")
    print(f"Trainable LoRA parameters: {count_trainable_parameters(model)}")
    print(f"Hyperparameters: {config}")
    print(f"Seed: {config.seed}")

    result = fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_path=checkpoint_path,
        n_epochs=config.n_epochs,
        patience=config.patience,
        max_grad_norm=config.max_grad_norm,
    )

    print(f"Best dev ppl: {result.best_dev.perplexity:.2f}")
    print(f"Test ppl: {result.test.perplexity:.2f}")
    print(f"Saved best LoRA adapters to: {result.checkpoint_path}")


def main() -> None:
    """Train the selected LoRA experiment and print final metrics."""

    args = parse_args()
    selected = (
        EXPERIMENTS
        if args.experiment == "all"
        else (EXPERIMENTS_BY_NAME[args.experiment],)
    )

    for config in selected:
        run_experiment(config, model_name=args.model_name)


if __name__ == "__main__":
    main()
