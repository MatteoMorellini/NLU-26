"""Entry point for GPT-2 LoRA fine-tuning experiments."""

import csv
import os
from argparse import ArgumentParser, Namespace
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
RESULTS_PATH = CHECKPOINT_DIR / "lora_sweep_results.csv"
DEFAULT_EXPERIMENT = "lora_r8_a8"
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


RANK_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16)
EXPERIMENTS: tuple[ExperimentConfig, ...] = tuple(
    ExperimentConfig(
        name=f"lora_r{rank}_a{rank}",
        rank=rank,
        alpha=float(rank),
        learning_rate=5e-4,
    )
    for rank in RANK_SWEEP
) + tuple(
    ExperimentConfig(
        name=f"lora_r{rank}_a{2 * rank}",
        rank=rank,
        alpha=float(2 * rank),
        learning_rate=5e-4,
    )
    for rank in RANK_SWEEP
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
        default="openai-community/gpt2",
        help="Hugging Face model name or local path to use as the GPT-2 base.",
    )
    parser.add_argument(
        "--lora-targets",
        choices=("lab", "paper"),
        default="lab",
        help=(
            "Use 'lab' for query/key/value adapters, or 'paper' for the "
            "query/value setup used in the LoRA GPT experiments."
        ),
    )
    return parser.parse_args()


def resolve_lora_targets(mode: LoRATargetMode) -> LoRATargets:
    """Map the CLI target mode to concrete GPT-2 attention projections."""

    if mode == "paper":
        return "query_value"
    return "query_key_value"


def write_results(results: list[ExperimentRunResult], path: Path = RESULTS_PATH) -> None:
    """Write sweep metrics for report-ready comparison across LoRA hyperparameters."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "lora_targets",
                "rank",
                "alpha",
                "learning_rate",
                "pretrained_dev_ppl",
                "best_dev_ppl",
                "test_ppl",
                "total_parameters",
                "trainable_parameters",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "name": result.name,
                    "lora_targets": result.lora_targets,
                    "rank": result.rank,
                    "alpha": result.alpha,
                    "learning_rate": result.learning_rate,
                    "pretrained_dev_ppl": result.pretrained_dev_ppl,
                    "best_dev_ppl": result.best_dev_ppl,
                    "test_ppl": result.test_ppl,
                    "total_parameters": result.total_parameters,
                    "trainable_parameters": result.trainable_parameters,
                    "checkpoint_path": str(result.checkpoint_path),
                }
            )


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


def main() -> None:
    """Train the selected LoRA experiment and print final metrics."""

    args = parse_args()
    lora_targets = resolve_lora_targets(args.lora_targets)
    selected = (
        EXPERIMENTS
        if args.experiment == "all"
        else (EXPERIMENTS_BY_NAME[args.experiment],)
    )

    results = [
        run_experiment(
            config,
            model_name=args.model_name,
            lora_targets=lora_targets,
        )
        for config in selected
    ]
    write_results(results)
    print(f"Saved LoRA comparison metrics to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
