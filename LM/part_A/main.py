"""Reusable Part A experiment helpers."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

from runtime_config import configure_runtime_environment

configure_runtime_environment()

import torch
from torch import nn
from torch.optim import AdamW, Optimizer

from functions import LRSchedule, TrainingResult, build_lr_scheduler, count_trainable_parameters, fit_model, initialize_weights
from model import GPT2
from utils import IGNORE_INDEX, build_dataloaders, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("bin")


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for one Part A language-modeling experiment."""

    name: str
    learning_rate: float
    pos_emb_size: int = 1024
    d_model: int = 512
    n_heads: int = 8
    num_layers: int = 10
    ff_dim: int = 3072
    dropout: float = 0.3
    tie_weights: bool = False
    weight_decay: float = 0.0
    lr_schedule: LRSchedule = "none"
    warmup_steps: int = 0
    gradient_clip: float | None = None
    batch_size: int = 64
    eval_batch_size: int = 128
    n_epochs: int = 100
    patience: int = 5
    seed: int = 42

def parse_args() -> Namespace:
    """Parse command-line options for one training run."""

    defaults = ExperimentConfig(name="single_run", learning_rate=3e-4)
    parser = ArgumentParser(description="Train and evaluate one Part A language model configuration.")
    parser.add_argument("--name", type=str, default=defaults.name)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--pos-emb-size", type=int, default=defaults.pos_emb_size)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--n-heads", type=int, default=defaults.n_heads)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--ff-dim", type=int, default=defaults.ff_dim)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--tie-weights", action="store_true", default=defaults.tie_weights)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "linear", "cosine", "inverse_sqrt"),
        default=defaults.lr_schedule,
    )
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--gradient-clip", type=float, default=defaults.gradient_clip)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=defaults.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.n_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    return parser.parse_args()


def config_from_args(args: Namespace) -> ExperimentConfig:
    """Build one experiment config from parsed command-line options."""

    return ExperimentConfig(
        name=args.name,
        learning_rate=args.learning_rate,
        pos_emb_size=args.pos_emb_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        n_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )




def run_experiment(config: ExperimentConfig) -> TrainingResult:
    """Train and evaluate one model configuration."""

    set_seed(config.seed)
    tokenizer, train_loader, valid_loader, test_loader = build_dataloaders(
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        device=DEVICE,
        max_length=config.pos_emb_size,
        seed=config.seed,
    )

    model = GPT2(
        vocab_size=len(tokenizer),
        pos_emb_size=config.pos_emb_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        tie_weights=config.tie_weights,
    ).to(DEVICE)
    initialize_weights(model)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.n_epochs * len(train_loader)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        schedule=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
    )
    checkpoint_path = CHECKPOINT_DIR / f"{config.name}.pt"

    print(f"Experiment: {config.name}")
    print(f"Device: {DEVICE}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
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
        scheduler=scheduler,
        gradient_clip=config.gradient_clip,
    )

    print(f"Best dev ppl: {result.best_dev.perplexity:.2f}")
    print(f"Test ppl: {result.test.perplexity:.2f}")
    print(f"Saved best model to: {result.checkpoint_path}")
    return result


def main() -> None:
    """Run one configured experiment from the command line."""

    run_experiment(config_from_args(parse_args()))


if __name__ == "__main__":
    main()
