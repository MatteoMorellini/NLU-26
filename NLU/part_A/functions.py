"""Training, evaluation, and experiment helpers for ATIS intent/slot filling."""

from __future__ import annotations

import copy
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model import GPT2IntentSlotModel, ModelConfig, build_model
from utils import Batch, DEVICE, LabelVocab, set_seed


InitializationStrategy = Literal["gpt2", "lm", "xavier"]


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    learning_rate: float = 5e-4
    n_epochs: int = 30
    patience: int = 5
    seed: int = 42
    initialization: InitializationStrategy = "gpt2"


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics and average loss."""

    slot_f1: float
    intent_accuracy: float
    loss: float


@dataclass(frozen=True)
class ExperimentResult:
    """One completed training run."""

    name: str
    seed: int
    learning_rate: float
    d_model: int
    n_heads: int
    num_layers: int
    ff_dim: int
    dropout: float
    best_dev_slot_f1: float
    best_dev_intent_accuracy: float
    test_slot_f1: float
    test_intent_accuracy: float
    epochs_ran: int
    checkpoint_path: str


@dataclass(frozen=True)
class AggregateExperimentResult:
    """Mean and standard deviation over repeated runs."""

    name: str
    learning_rate: float
    d_model: int
    n_heads: int
    num_layers: int
    ff_dim: int
    dropout: float
    runs: int
    seeds: str
    dev_slot_f1_mean: float
    dev_slot_f1_std: float
    dev_intent_accuracy_mean: float
    dev_intent_accuracy_std: float
    test_slot_f1_mean: float
    test_slot_f1_std: float
    test_intent_accuracy_mean: float
    test_intent_accuracy_std: float
    best_checkpoint_path: str


def initialize_weights(model: nn.Module, strategy: InitializationStrategy = "gpt2") -> None:
    """Initialize train-from-scratch GPT-2 weights.

    `gpt2` follows the standard GPT-2/Hugging Face convention: normal weights with
    std=0.02, zero biases, and LayerNorm scale set to 1.
    `lm` mirrors LM/part_A: linear weights uniform in [-0.01, 0.01], linear biases
    0.01, and embeddings normal with std=0.02.
    `xavier` is a common alternative for linear projections and output heads.
    """

    initialized_weights: set[int] = set()

    for module in model.modules():
        if isinstance(module, nn.Embedding):
            _init_once(module.weight, initialized_weights, lambda weight: nn.init.normal_(weight, mean=0.0, std=0.02))
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif _is_linear_like(module):
            _initialize_linear_like(module, initialized_weights, strategy)


def _init_once(parameter: torch.Tensor, initialized_weights: set[int], initializer) -> None:
    """Initialize a tensor once, even if weights are tied/shared."""

    weight_id = id(parameter)
    if weight_id in initialized_weights:
        return
    initializer(parameter)
    initialized_weights.add(weight_id)


def _is_linear_like(module: nn.Module) -> bool:
    """Return whether a module has linear-projection-style weights."""

    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"


def _initialize_linear_like(
    module: nn.Module,
    initialized_weights: set[int],
    strategy: InitializationStrategy,
) -> None:
    """Initialize Linear and GPT-2 Conv1D modules."""

    weight = getattr(module, "weight")
    bias = getattr(module, "bias", None)

    if strategy == "gpt2":
        _init_once(weight, initialized_weights, lambda tensor: nn.init.normal_(tensor, mean=0.0, std=0.02))
        if bias is not None:
            nn.init.zeros_(bias)
        return

    if strategy == "lm":
        _init_once(weight, initialized_weights, lambda tensor: nn.init.uniform_(tensor, -0.01, 0.01))
        if bias is not None:
            nn.init.constant_(bias, 0.01)
        return

    if strategy == "xavier":
        _init_once(weight, initialized_weights, nn.init.xavier_uniform_)
        if bias is not None:
            nn.init.zeros_(bias)
        return

    raise ValueError(f"Unknown initialization strategy: {strategy}")


def train_loop(
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_slots: nn.Module,
    criterion_intents: nn.Module,
    model: GPT2IntentSlotModel,
) -> list[float]:
    """Run one training epoch."""

    model.train()
    losses: list[float] = []

    for batch in data:
        optimizer.zero_grad()
        slots, intents = model(batch["input_ids"], batch["attention_mask"])
        loss_intent = criterion_intents(intents, batch["intent_labels"])
        loss_slot = criterion_slots(slots.permute(0, 2, 1), batch["slot_labels"])
        loss = loss_intent + loss_slot
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()

    return losses


def eval_loop(
    data: DataLoader,
    criterion_slots: nn.Module,
    criterion_intents: nn.Module,
    model: GPT2IntentSlotModel,
    label_vocab: LabelVocab,
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
            slots, intents = model(batch["input_ids"], batch["attention_mask"])
            loss_intent = criterion_intents(intents, batch["intent_labels"])
            loss_slot = criterion_slots(slots.permute(0, 2, 1), batch["slot_labels"])
            losses.append(float((loss_intent + loss_slot).item()))

            pred_intents = torch.argmax(intents, dim=1)
            intent_correct += int((pred_intents == batch["intent_labels"]).sum().item())
            intent_total += int(batch["intent_labels"].numel())

            pred_slots = torch.argmax(slots, dim=2).detach().cpu()
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
    """Convert subword predictions back to word-level sequences."""

    for row, words in enumerate(batch["words"]):
        refs = batch["word_slot_labels"][row]
        pred_ids = pred_slots[row][first_masks[row]].tolist()
        usable_len = min(len(words), len(refs), len(pred_ids))
        ref_slots.append([(words[i], refs[i]) for i in range(usable_len)])
        hyp_slots.append([(words[i], id2slot[int(pred_ids[i])]) for i in range(usable_len)])


def run_training(
    name: str,
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    test_loader: DataLoader,
    label_vocab: LabelVocab,
    checkpoint_path: Path,
) -> ExperimentResult:
    """Train one model, keep the best dev checkpoint, and evaluate it on test."""

    set_seed(train_config.seed)
    model = build_model(model_config).to(DEVICE)
    initialize_weights(model, train_config.initialization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_intents = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_dev = EvaluationResult(slot_f1=0.0, intent_accuracy=0.0, loss=0.0)
    patience_left = train_config.patience
    epochs_ran = 0

    pbar = tqdm(range(train_config.n_epochs), desc=name)
    for epoch in pbar:
        epochs_ran = epoch + 1
        train_losses = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        dev_result = eval_loop(dev_loader, criterion_slots, criterion_intents, model, label_vocab)
        pbar.set_postfix(
            train_loss=f"{np.mean(train_losses):.4f}",
            dev_slot_f1=f"{dev_result.slot_f1:.4f}",
            dev_int_acc=f"{dev_result.intent_accuracy:.4f}",
        )

        if dev_result.slot_f1 > best_dev.slot_f1:
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
            "seed": train_config.seed,
            "initialization": train_config.initialization,
            "label_vocab": {
                "slot2id": label_vocab.slot2id,
                "intent2id": label_vocab.intent2id,
            },
        },
        checkpoint_path,
    )

    test_result = eval_loop(test_loader, criterion_slots, criterion_intents, model, label_vocab)
    return ExperimentResult(
        name=name,
        seed=train_config.seed,
        learning_rate=train_config.learning_rate,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        num_layers=model_config.num_layers,
        ff_dim=model_config.ff_dim,
        dropout=model_config.dropout,
        best_dev_slot_f1=best_dev.slot_f1,
        best_dev_intent_accuracy=best_dev.intent_accuracy,
        test_slot_f1=test_result.slot_f1,
        test_intent_accuracy=test_result.intent_accuracy,
        epochs_ran=epochs_ran,
        checkpoint_path=str(checkpoint_path),
    )


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device = DEVICE) -> tuple[GPT2IntentSlotModel, LabelVocab]:
    """Load a saved model and label vocabulary."""

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


def summarize_repeated_runs(name: str, results: list[ExperimentResult]) -> AggregateExperimentResult:
    """Compute mean and standard deviation across repeated seeds."""

    if not results:
        raise ValueError("Cannot summarize an empty result list")

    best = max(results, key=lambda result: result.best_dev_slot_f1)
    first = results[0]
    return AggregateExperimentResult(
        name=name,
        learning_rate=first.learning_rate,
        d_model=first.d_model,
        n_heads=first.n_heads,
        num_layers=first.num_layers,
        ff_dim=first.ff_dim,
        dropout=first.dropout,
        runs=len(results),
        seeds=" ".join(str(result.seed) for result in results),
        dev_slot_f1_mean=float(np.mean([result.best_dev_slot_f1 for result in results])),
        dev_slot_f1_std=float(np.std([result.best_dev_slot_f1 for result in results])),
        dev_intent_accuracy_mean=float(np.mean([result.best_dev_intent_accuracy for result in results])),
        dev_intent_accuracy_std=float(np.std([result.best_dev_intent_accuracy for result in results])),
        test_slot_f1_mean=float(np.mean([result.test_slot_f1 for result in results])),
        test_slot_f1_std=float(np.std([result.test_slot_f1 for result in results])),
        test_intent_accuracy_mean=float(np.mean([result.test_intent_accuracy for result in results])),
        test_intent_accuracy_std=float(np.std([result.test_intent_accuracy for result in results])),
        best_checkpoint_path=best.checkpoint_path,
    )


def write_aggregate_results(results: Iterable[AggregateExperimentResult], output_path: str | Path) -> None:
    """Write repeated-run aggregate results to CSV."""

    rows = [result.__dict__ for result in results]
    if not rows:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stats() -> dict[str, int]:
    """Create a CoNLL counter."""

    return {"cor": 0, "hyp": 0, "ref": 0}


def evaluate(ref: list[list[tuple[str, str]]], hyp: list[list[tuple[str, str]]], otag: str = "O"):
    """Evaluate slot chunks with CoNLL-style span F1."""

    return conlleval(align_hyp(ref, hyp), otag=otag)


def align_hyp(ref: list[list[tuple[str, str]]], hyp: list[list[tuple[str, str]]]):
    """Align reference and hypothesis labels."""

    if len(ref) != len(hyp):
        raise ValueError(f"Size mismatch: ref={len(ref)} hyp={len(hyp)}")

    out = []
    for idx in range(len(ref)):
        if len(ref[idx]) != len(hyp[idx]):
            raise ValueError(f"Size mismatch at sentence {idx}: ref={len(ref[idx])} hyp={len(hyp[idx])}")
        out.append([(*ref[idx][j], hyp[idx][j][-1]) for j in range(len(ref[idx]))])
    return out


def conlleval(data, otag: str = "O"):
    """Compute CoNLL chunk-level scores."""

    seg = stats()
    cls: dict[str | None, dict[str, int]] = {}

    for sent in data:
        prev_ref = otag
        prev_hyp = otag
        prev_ref_iob = None
        prev_hyp_iob = None
        in_correct = False

        for token in sent:
            hyp_iob, hyp = parse_iob(token[-1])
            ref_iob, ref = parse_iob(token[-2])
            ref_e = is_eoc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_e = is_eoc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)
            ref_b = is_boc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_b = is_boc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            if ref not in cls and ref:
                cls[ref] = stats()
            if hyp not in cls and hyp:
                cls[hyp] = stats()

            if in_correct:
                if ref_e and hyp_e and prev_hyp == prev_ref:
                    in_correct = False
                    seg["cor"] += 1
                    cls[prev_ref]["cor"] += 1
                elif ref_e != hyp_e or hyp != ref:
                    in_correct = False

            if ref_b and hyp_b and hyp == ref:
                in_correct = True
            if ref_b:
                seg["ref"] += 1
                cls[ref]["ref"] += 1
            if hyp_b:
                seg["hyp"] += 1
                cls[hyp]["hyp"] += 1

            prev_ref = ref
            prev_hyp = hyp
            prev_ref_iob = ref_iob
            prev_hyp_iob = hyp_iob

        if in_correct:
            seg["cor"] += 1
            cls[prev_ref]["cor"] += 1

    return summarize(seg, cls)


def parse_iob(tag: str) -> tuple[str, str | None]:
    """Split an IOB tag into prefix and chunk label."""

    match = re.match(r"^([^-]*)-(.*)$", tag)
    return match.groups() if match else (tag, None)


def is_boc(lbl: str | None, iob: str, prev_lbl: str | None, prev_iob: str | None, otag: str = "O") -> bool:
    """Return whether the current tag starts a chunk."""

    return (
        iob in ["B", "S", "U"]
        or (iob in ["E", "L"] and prev_iob in ["E", "L", "S", otag])
        or (iob == "I" and prev_iob in ["S", "L", "E", otag])
        or (lbl != prev_lbl and iob != otag and iob != ".")
        or iob in ["[", "]"]
    )


def is_eoc(lbl: str | None, iob: str, prev_lbl: str | None, prev_iob: str | None, otag: str = "O") -> bool:
    """Return whether the current tag ends the previous chunk."""

    return (
        iob in ["E", "L", "S", "U"]
        or (iob == "B" and prev_iob in ["B", "I"])
        or (iob in ["S", "U"] and prev_iob in ["B", "I"])
        or (iob == otag and prev_iob in ["B", "I"])
        or (lbl != prev_lbl and iob != otag and prev_iob != ".")
        or iob in ["[", "]"]
    )


def score(cor_cnt: int, hyp_cnt: int, ref_cnt: int) -> dict[str, float | int]:
    """Compute precision, recall, F1, and support."""

    precision = 1.0 if hyp_cnt == 0 else cor_cnt / hyp_cnt
    recall = 0.0 if ref_cnt == 0 else cor_cnt / ref_cnt
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {"p": precision, "r": recall, "f": f1, "s": ref_cnt}


def summarize(seg: dict[str, int], cls: dict[str | None, dict[str, int]]):
    """Summarize CoNLL scores."""

    res = {label: score(cls[label]["cor"], cls[label]["hyp"], cls[label]["ref"]) for label in cls}
    res.update({"total": score(seg.get("cor", 0), seg.get("hyp", 0), seg.get("ref", 0))})
    return res
