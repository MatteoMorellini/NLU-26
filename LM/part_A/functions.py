"""Training, evaluation, and checkpoint helpers for the LM exercise."""

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import LanguageModelBatch


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics for a language model."""

    loss: float
    perplexity: float


@dataclass(frozen=True)
class TrainingResult:
    """Final training result and checkpoint location."""

    best_dev: EvaluationResult
    test: EvaluationResult
    checkpoint_path: Path


def compute_language_modeling_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute token-level LM loss on flattened logits."""

    vocab_size = logits.size(-1)
    return criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))


def train_loop(
    data: DataLoader[LanguageModelBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    model: nn.Module,
) -> float:
    """Run one training epoch and return token-normalized loss."""

    model.train()
    weighted_losses: list[float] = []
    token_counts: list[int] = []

    for batch in tqdm(data, desc="Training", unit="batch"):
        optimizer.zero_grad()
        output = model(batch.input_ids)
        loss = compute_language_modeling_loss(criterion, output, batch.labels)
        n_tokens = int(batch.n_tokens.item())

        weighted_losses.append(float(loss.item()) * n_tokens)
        token_counts.append(n_tokens)
        loss.backward()
        optimizer.step()

    return sum(weighted_losses) / sum(token_counts)


def eval_loop(
    data: DataLoader[LanguageModelBatch],
    criterion: nn.Module,
    model: nn.Module,
) -> EvaluationResult:
    """Evaluate a model and return loss plus perplexity."""

    model.eval()
    weighted_losses: list[float] = []
    token_counts: list[int] = []

    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating", unit="batch"):
            output = model(batch.input_ids)
            loss = compute_language_modeling_loss(criterion, output, batch.labels)
            n_tokens = int(batch.n_tokens.item())
            weighted_losses.append(float(loss.item()) * n_tokens)
            token_counts.append(n_tokens)

    loss_value = sum(weighted_losses) / sum(token_counts)
    return EvaluationResult(loss=loss_value, perplexity=math.exp(loss_value))


def initialize_weights(model: nn.Module) -> None:
    """Initialize train-from-scratch GPT-style model weights."""

    initialized_weights: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight_id = id(module.weight)
            if weight_id not in initialized_weights:
                nn.init.uniform_(module.weight, -0.01, 0.01)
                initialized_weights.add(weight_id)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Embedding):
            weight_id = id(module.weight)
            if weight_id not in initialized_weights:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                initialized_weights.add(weight_id)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader[LanguageModelBatch],
    valid_loader: DataLoader[LanguageModelBatch],
    test_loader: DataLoader[LanguageModelBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    checkpoint_path: Path,
    n_epochs: int,
    patience: int,
) -> TrainingResult:
    """Train with early stopping, save the best model, and evaluate on test data."""

    best_dev = EvaluationResult(loss=math.inf, perplexity=math.inf)
    best_state = deepcopy(model.state_dict())
    remaining_patience = patience

    for epoch in range(1, n_epochs + 1):
        train_loss = train_loop(train_loader, optimizer, criterion, model)
        dev_result = eval_loop(valid_loader, criterion, model)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_loss={dev_result.loss:.4f} dev_ppl={dev_result.perplexity:.2f}"
        )

        if dev_result.perplexity < best_dev.perplexity:
            best_dev = dev_result
            best_state = deepcopy(model.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience <= 0:
            break

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)
    test_result = eval_loop(test_loader, criterion, model)
    return TrainingResult(
        best_dev=best_dev,
        test=test_result,
        checkpoint_path=checkpoint_path,
    )
