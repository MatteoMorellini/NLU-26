"""Reusable Part B GPT-2 LoRA fine-tuning helpers."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias


PART_DIR = Path(__file__).resolve().parent
CACHE_DIR = PART_DIR / "hf_cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))

import torch
from torch.optim import AdamW

from functions import (
    count_total_parameters,
    count_trainable_parameters,
    eval_loop,
    fit_model,
)
from model import LoRATargets, load_gpt2_lora
from utils import build_dataloaders, set_seed

LoRATargetMode: TypeAlias = Literal["lab", "paper"]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = PART_DIR / "bin"
LORA_INIT_STD = 0.02


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters for one GPT-2 LoRA experiment."""

    name: str
    rank: int
    alpha: float
    learning_rate: float
    batch_size: int = 4
    eval_batch_size: int = 8
    max_length: int = 128
    n_epochs: int = 3
    patience: int = 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class ExperimentRunResult:
    """Recorded metrics for one pretrained GPT-2 plus LoRA fine-tuning run."""

    name: str
    lora_targets: LoRATargets
    rank: int
    alpha: float
    learning_rate: float
    pretrained_dev_ppl: float
    best_dev_ppl: float
    test_ppl: float
    total_parameters: int
    trainable_parameters: int
    checkpoint_path: Path


def resolve_lora_targets(mode: LoRATargetMode) -> LoRATargets:
    """Map the CLI target mode to concrete GPT-2 attention projections."""

    if mode == "paper":
        return "query_value"
    return "query_key_value"


def run_experiment(
    config: ExperimentConfig,
    model_name: str,
    lora_targets: LoRATargets,
) -> ExperimentRunResult:
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
    model = load_gpt2_lora(
        model_name=model_name,
        rank=config.rank,
        alpha=config.alpha,
        lora_init_std=LORA_INIT_STD,
        lora_targets=lora_targets,
        cache_dir=CACHE_DIR,
    ).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    total_parameters = count_total_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

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
    print(f"Total parameters: {total_parameters}")
    print(f"Trainable LoRA parameters: {trainable_parameters}")
    print(f"LoRA targets: {lora_targets}")
    print(f"Hyperparameters: {config}")
    print(f"LoRA init: A ~ N(0, {LORA_INIT_STD}), B = 0")
    print(f"Seed: {config.seed}")

    pretrained_dev = eval_loop(valid_loader, model)
    print(f"Pretrained dev ppl before LoRA updates: {pretrained_dev.perplexity:.2f}")

    result = fit_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        checkpoint_path=checkpoint_path,
        n_epochs=config.n_epochs,
        patience=config.patience,
        max_grad_norm=config.max_grad_norm,
    )

    print(f"Best dev ppl: {result.best_dev.perplexity:.2f}")
    print(f"Test ppl: {result.test.perplexity:.2f}")
    print(f"Saved best LoRA adapters to: {result.checkpoint_path}")
    return ExperimentRunResult(
        name=config.name,
        lora_targets=lora_targets,
        rank=config.rank,
        alpha=config.alpha,
        learning_rate=config.learning_rate,
        pretrained_dev_ppl=pretrained_dev.perplexity,
        best_dev_ppl=result.best_dev.perplexity,
        test_ppl=result.test.perplexity,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        checkpoint_path=result.checkpoint_path,
    )
