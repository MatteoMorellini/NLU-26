"""Training, evaluation, and checkpoint helpers for the LM exercise."""

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from runtime_config import configure_runtime_environment

configure_runtime_environment()

import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import LanguageModelBatch

LRSchedule = Literal["none", "linear", "cosine", "inverse_sqrt"]


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


def build_lr_scheduler(
    optimizer: Optimizer,
    schedule: LRSchedule,
    warmup_steps: int,
    total_steps: int,
) -> lr_scheduler.LambdaLR | None:
    """Build a step-wise LR scheduler with optional warmup."""

    if schedule == "none":
        return None
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    warmup_steps = min(warmup_steps, total_steps)

    def warmup_factor(step: int) -> float:
        if warmup_steps == 0 or step >= warmup_steps:
            return 1.0
        return float(step + 1) / float(warmup_steps)

    def lr_lambda(step: int) -> float:
        if schedule == "linear":
            if step < warmup_steps:
                return warmup_factor(step)
            decay_steps = max(1, total_steps - warmup_steps)
            progress = float(step - warmup_steps) / float(decay_steps)
            return max(0.0, 1.0 - progress)

        if schedule == "cosine":
            if step < warmup_steps:
                return warmup_factor(step)
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, float(step - warmup_steps) / float(decay_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        if schedule == "inverse_sqrt":
            reference_step = max(1, warmup_steps)
            current_step = max(1, step + 1)
            if step < warmup_steps:
                return warmup_factor(step)
            return math.sqrt(reference_step / current_step)

        raise ValueError(f"Unsupported LR schedule: {schedule}")

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_loop(
    data: DataLoader[LanguageModelBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    model: nn.Module,
    scheduler: lr_scheduler.LambdaLR | None = None,
    gradient_clip: float | None = None,
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
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
    scheduler: lr_scheduler.LambdaLR | None = None,
    gradient_clip: float | None = None,
) -> TrainingResult:
    """Train with early stopping, save the best model, and evaluate on test data."""

    best_dev = EvaluationResult(loss=math.inf, perplexity=math.inf)
    best_state = deepcopy(model.state_dict())
    remaining_patience = patience

    for epoch in range(1, n_epochs + 1):
        train_loss = train_loop(
            train_loader,
            optimizer,
            criterion,
            model,
            scheduler=scheduler,
            gradient_clip=gradient_clip,
        )
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
