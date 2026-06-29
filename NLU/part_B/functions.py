"""Training, evaluation, and experiment helpers for Part B."""

from __future__ import annotations

import copy
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from conll import evaluate
from model import ModelConfig, build_model
from utils import Batch, DEVICE, IGNORE_SLOT_ID, LabelVocab, ModelType, set_seed


@dataclass(frozen=True)
class TrainConfig:
    """Fine-tuning hyperparameters."""

    learning_rate: float = 5e-5
    n_epochs: int = 5
    patience: int = 2
    seed: int = 42
    intent_loss_weight: float = 1.0
    slot_loss_weight: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and average loss."""

    slot_f1: float
    intent_accuracy: float
    loss: float


@dataclass(frozen=True)
class ExperimentResult:
    """One completed fine-tuning run."""

    model_type: ModelType
    pretrained_model_name: str
    seed: int
    learning_rate: float
    best_dev_slot_f1: float
    best_dev_intent_accuracy: float
    test_slot_f1: float
    test_intent_accuracy: float
    epochs_ran: int
    checkpoint_path: str


def compute_slot_loss(criterion_slots: nn.Module, slot_logits: torch.Tensor, slot_labels: torch.Tensor) -> torch.Tensor:
    """Compute token-level slot loss while ignoring masked subtokens."""

    return criterion_slots(
        slot_logits.reshape(-1, slot_logits.shape[-1]),
        slot_labels.reshape(-1),
    )


def compute_multitask_loss(
    slot_logits: torch.Tensor,
    intent_logits: torch.Tensor,
    batch: Batch,
    criterion_slots: nn.Module,
    criterion_intents: nn.Module,
    train_config: TrainConfig,
) -> torch.Tensor:
    """Combine slot and intent losses for multi-task learning."""

    loss_slot = compute_slot_loss(criterion_slots, slot_logits, batch["slot_labels"])
    loss_intent = criterion_intents(intent_logits, batch["intent_labels"])
    return train_config.slot_loss_weight * loss_slot + train_config.intent_loss_weight * loss_intent


def train_loop(
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion_slots: nn.Module,
    criterion_intents: nn.Module,
    model: nn.Module,
    train_config: TrainConfig,
) -> list[float]:
    """Run one fine-tuning epoch."""

    model.train()
    losses: list[float] = []

    for batch in data:
        optimizer.zero_grad()
        slot_logits, intent_logits = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_multitask_loss(
            slot_logits,
            intent_logits,
            batch,
            criterion_slots,
            criterion_intents,
            train_config,
        )
        losses.append(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return losses


def eval_loop(
    data: DataLoader,
    criterion_slots: nn.Module,
    criterion_intents: nn.Module,
    model: nn.Module,
    label_vocab: LabelVocab,
    train_config: TrainConfig,
) -> EvaluationResult:
    """Evaluate slot F1 with CoNLL and intent accuracy."""

    model.eval()
    losses: list[float] = []
    ref_slots: list[list[tuple[str, str]]] = []
    hyp_slots: list[list[tuple[str, str]]] = []
    intent_correct = 0
    intent_total = 0
    id2slot = label_vocab.id2slot

    with torch.no_grad():
        for batch in data:
            slot_logits, intent_logits = model(batch["input_ids"], batch["attention_mask"])
            loss = compute_multitask_loss(
                slot_logits,
                intent_logits,
                batch,
                criterion_slots,
                criterion_intents,
                train_config,
            )
            losses.append(float(loss.item()))

            pred_intents = torch.argmax(intent_logits, dim=1)
            intent_correct += int((pred_intents == batch["intent_labels"]).sum().item())
            intent_total += int(batch["intent_labels"].numel())

            pred_slots = torch.argmax(slot_logits, dim=2).detach().cpu()
            first_masks = batch["first_subtoken_mask"].detach().cpu()
            _extend_slot_predictions(batch, pred_slots, first_masks, id2slot, ref_slots, hyp_slots)

    slot_scores = evaluate(ref_slots, hyp_slots)
    return EvaluationResult(
        slot_f1=slot_scores["total"]["f"],
        intent_accuracy=intent_correct / intent_total if intent_total else 0.0,
        loss=float(np.mean(losses)) if losses else 0.0,
    )


def _extend_slot_predictions(
    batch: Batch,
    pred_slots: torch.Tensor,
    first_masks: torch.Tensor,
    id2slot: dict[int, str],
    ref_slots: list[list[tuple[str, str]]],
    hyp_slots: list[list[tuple[str, str]]],
) -> None:
    """Convert first-subtoken predictions back to word-level CoNLL sequences."""

    for row, words in enumerate(batch["words"]):
        refs = batch["word_slot_labels"][row]
        pred_ids = pred_slots[row][first_masks[row]].tolist()
        usable_len = min(len(words), len(refs), len(pred_ids))
        ref_slots.append([(words[i], refs[i]) for i in range(usable_len)])
        hyp_slots.append([(words[i], id2slot[int(pred_ids[i])]) for i in range(usable_len)])


def run_training(
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    label_vocab: LabelVocab,
    checkpoint_path: Path,
) -> ExperimentResult:
    """Fine-tune one model, keep the best dev checkpoint, and evaluate test."""

    set_seed(train_config.seed)
    model = build_model(model_config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * train_config.n_epochs)
    warmup_steps = int(total_steps * train_config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_SLOT_ID)
    criterion_intents = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_dev = EvaluationResult(slot_f1=0.0, intent_accuracy=0.0, loss=0.0)
    patience_left = train_config.patience
    epochs_ran = 0

    description = f"{model_config.model_type}:{model_config.pretrained_model_name}"
    pbar = tqdm(range(train_config.n_epochs), desc=description)
    for epoch in pbar:
        epochs_ran = epoch + 1
        train_losses = train_loop(
            train_loader,
            optimizer,
            scheduler,
            criterion_slots,
            criterion_intents,
            model,
            train_config,
        )
        dev_result = eval_loop(dev_loader, criterion_slots, criterion_intents, model, label_vocab, train_config)
        pbar.set_postfix(
            train_loss=f"{np.mean(train_losses):.4f}",
            dev_slot_f1=f"{dev_result.slot_f1:.4f}",
            dev_int_acc=f"{dev_result.intent_accuracy:.4f}",
        )

        if _is_better(dev_result, best_dev):
            best_dev = dev_result
            best_state = copy.deepcopy(model.state_dict())
            patience_left = train_config.patience
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model_config": model_config.to_dict(),
            "train_config": train_config.__dict__,
            "label_vocab": {
                "slot2id": label_vocab.slot2id,
                "intent2id": label_vocab.intent2id,
            },
        },
        checkpoint_path,
    )

    test_result = eval_loop(test_loader, criterion_slots, criterion_intents, model, label_vocab, train_config)
    return ExperimentResult(
        model_type=model_config.model_type,
        pretrained_model_name=model_config.pretrained_model_name,
        seed=train_config.seed,
        learning_rate=train_config.learning_rate,
        best_dev_slot_f1=best_dev.slot_f1,
        best_dev_intent_accuracy=best_dev.intent_accuracy,
        test_slot_f1=test_result.slot_f1,
        test_intent_accuracy=test_result.intent_accuracy,
        epochs_ran=epochs_ran,
        checkpoint_path=str(checkpoint_path),
    )


def _is_better(current: EvaluationResult, best: EvaluationResult) -> bool:
    """Rank checkpoints by slot F1 first, then intent accuracy."""

    if current.slot_f1 > best.slot_f1:
        return True
    return current.slot_f1 == best.slot_f1 and current.intent_accuracy > best.intent_accuracy


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device = DEVICE) -> tuple[nn.Module, LabelVocab]:
    """Load a saved Part B model and label vocabulary."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    label_vocab = LabelVocab(
        slot2id=checkpoint["label_vocab"]["slot2id"],
        intent2id=checkpoint["label_vocab"]["intent2id"],
    )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, label_vocab


def write_experiment_results(results: Iterable[ExperimentResult], output_path: str | Path) -> None:
    """Write experiment results to CSV."""

    rows = [result.__dict__ for result in results]
    if not rows:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
