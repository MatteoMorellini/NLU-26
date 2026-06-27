"""Reusable Part A experiment helpers."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from functions import TrainingResult, count_trainable_parameters, fit_model, initialize_weights
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
    d_model: int = 20
    n_heads: int = 1
    num_layers: int = 1
    ff_dim: int = 20
    dropout: float = 0.0
    tie_weights: bool = False
    batch_size: int = 64
    eval_batch_size: int = 128
    n_epochs: int = 100
    patience: int = 5
    seed: int = 42


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
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
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
    )

    print(f"Best dev ppl: {result.best_dev.perplexity:.2f}")
    print(f"Test ppl: {result.test.perplexity:.2f}")
    print(f"Saved best model to: {result.checkpoint_path}")
    return result
