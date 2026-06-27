"""Training, evaluation, and checkpoint helpers for GPT-2 LoRA fine-tuning."""

import math
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
    """Evaluation metrics for a causal language model."""

    loss: float
    perplexity: float


@dataclass(frozen=True)
class TrainingResult:
    """Final training result and checkpoint location."""

    best_dev: EvaluationResult
    test: EvaluationResult
    checkpoint_path: Path


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Return the total number of model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def assert_only_lora_trainable(model: nn.Module) -> None:
    """Fail fast if a non-LoRA parameter is accidentally left trainable."""

    invalid_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    ]
    if invalid_names:
        raise RuntimeError(
            "Only LoRA adapter parameters should be trainable. "
            f"Found trainable base parameters: {invalid_names[:5]}"
        )


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone only the trainable parameters."""

    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load trainable parameters back into the current model."""

    parameters = dict(model.named_parameters())
    for name, saved_parameter in state_dict.items():
        parameters[name].data.copy_(saved_parameter.to(parameters[name].device))


def train_loop(
    data: DataLoader[LanguageModelBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    model: nn.Module,
    max_grad_norm: float | None = 1.0,
) -> float:
    """Run one training epoch and return token-normalized loss."""

    model.train()
    weighted_losses: list[float] = []
    token_counts: list[int] = []

    for batch in tqdm(data, desc="Training", unit="batch"):
        optimizer.zero_grad()
        logits = model(batch.input_ids, attention_mask=batch.attention_mask)
        loss = criterion(logits.permute(0, 2, 1), batch.labels)
        n_tokens = int(batch.n_tokens.item())

        weighted_losses.append(float(loss.item()) * n_tokens)
        token_counts.append(n_tokens)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
            logits = model(batch.input_ids, attention_mask=batch.attention_mask)
            loss = criterion(logits.permute(0, 2, 1), batch.labels)
            n_tokens = int(batch.n_tokens.item())
            weighted_losses.append(float(loss.item()) * n_tokens)
            token_counts.append(n_tokens)

    loss_value = sum(weighted_losses) / sum(token_counts)
    return EvaluationResult(loss=loss_value, perplexity=math.exp(loss_value))


def fit_model(
    model: nn.Module,
    train_loader: DataLoader[LanguageModelBatch],
    valid_loader: DataLoader[LanguageModelBatch],
    test_loader: DataLoader[LanguageModelBatch],
    optimizer: Optimizer,
    criterion: nn.Module,
    checkpoint_path: Path | str,
    n_epochs: int,
    patience: int,
    max_grad_norm: float | None = 1.0,
) -> TrainingResult:
    """Train with early stopping, save the best LoRA weights, and test them."""

    checkpoint_path = Path(checkpoint_path)
    assert_only_lora_trainable(model)
    best_dev = EvaluationResult(loss=math.inf, perplexity=math.inf)
    best_state = trainable_state_dict(model)
    remaining_patience = patience

    for epoch in range(1, n_epochs + 1):
        train_loss = train_loop(
            train_loader,
            optimizer,
            criterion,
            model,
            max_grad_norm=max_grad_norm,
        )
        dev_result = eval_loop(valid_loader, criterion, model)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_loss={dev_result.loss:.4f} dev_ppl={dev_result.perplexity:.2f}"
        )

        if dev_result.perplexity < best_dev.perplexity:
            best_dev = dev_result
            best_state = trainable_state_dict(model)
            remaining_patience = patience
        else:
            remaining_patience -= 1

        if remaining_patience <= 0:
            break

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    load_trainable_state_dict(model, best_state)
    test_result = eval_loop(test_loader, criterion, model)
    return TrainingResult(
        best_dev=best_dev,
        test=test_result,
        checkpoint_path=checkpoint_path,
    )
