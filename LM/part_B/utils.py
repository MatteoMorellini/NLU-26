"""Dataset loading utilities for GPT-2 LoRA fine-tuning on Penn Treebank."""

import os
from dataclasses import dataclass
from pathlib import Path

PART_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PART_DIR.parent / "part_A" / "dataset" / "PennTreeBank"
DEFAULT_CACHE_DIR = PART_DIR / "hf_cache"
IGNORE_INDEX = -100

os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_DIR / "hub"))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast


@dataclass(frozen=True)
class DatasetPaths:
    """Paths for the Penn Treebank train, validation, and test splits."""

    train: Path
    valid: Path
    test: Path


@dataclass(frozen=True)
class LanguageModelBatch:
    """A padded causal language-modeling batch."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    n_tokens: torch.Tensor


class GPT2PennTreeBankDataset(Dataset[list[int]]):
    """Penn Treebank split encoded as contiguous GPT-2 token blocks."""

    def __init__(
        self,
        corpus: list[str],
        tokenizer: GPT2TokenizerFast,
        max_length: int,
    ) -> None:
        self.examples: list[list[int]] = []
        token_stream: list[int] = []
        for sentence in corpus:
            text = f"{sentence.strip()} {tokenizer.eos_token}"
            token_stream.extend(tokenizer.encode(text, add_special_tokens=False))

        for start in range(0, len(token_stream), max_length):
            token_ids = token_stream[start : start + max_length]
            if len(token_ids) >= 2:
                self.examples.append(token_ids)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> list[int]:
        return self.examples[idx]


def get_dataset_paths(dataset_dir: Path = DEFAULT_DATASET_DIR) -> DatasetPaths:
    """Return the expected Penn Treebank split paths."""

    return DatasetPaths(
        train=dataset_dir / "ptb.train.txt",
        valid=dataset_dir / "ptb.valid.txt",
        test=dataset_dir / "ptb.test.txt",
    )


def read_corpus(path: Path) -> list[str]:
    """Read non-empty PTB lines."""

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_gpt2_tokenizer(
    model_name: str = "gpt2",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> GPT2TokenizerFast:
    """Load GPT-2 tokenizer and set EOS as the padding token."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def collate_language_model_batch(
    batch: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> LanguageModelBatch:
    """Pad encoded sentences and create shifted GPT-style inputs and labels."""

    lengths = [len(sentence) for sentence in batch]
    max_len = max(lengths)
    padded = [
        sentence + [pad_token_id] * (max_len - len(sentence))
        for sentence in batch
    ]
    token_ids = torch.tensor(padded, dtype=torch.long, device=device)
    full_attention_mask = torch.zeros_like(token_ids)
    for row, length in enumerate(lengths):
        full_attention_mask[row, :length] = 1

    input_ids = token_ids[:, :-1]
    labels = token_ids[:, 1:]
    attention_mask = full_attention_mask[:, :-1]
    label_mask = full_attention_mask[:, 1:]
    labels = labels.masked_fill(label_mask == 0, IGNORE_INDEX)
    n_tokens = torch.sum(label_mask)
    return LanguageModelBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        n_tokens=n_tokens,
    )


def build_dataloaders(
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
    model_name: str = "gpt2",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    max_length: int = 128,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[GPT2TokenizerFast, DataLoader[LanguageModelBatch], DataLoader[LanguageModelBatch], DataLoader[LanguageModelBatch]]:
    """Load PTB splits and return tokenizer plus train/dev/test loaders."""

    paths = get_dataset_paths(dataset_dir)
    tokenizer = load_gpt2_tokenizer(model_name=model_name, cache_dir=cache_dir)

    train_dataset = GPT2PennTreeBankDataset(
        read_corpus(paths.train),
        tokenizer,
        max_length=max_length,
    )
    valid_dataset = GPT2PennTreeBankDataset(
        read_corpus(paths.valid),
        tokenizer,
        max_length=max_length,
    )
    test_dataset = GPT2PennTreeBankDataset(
        read_corpus(paths.test),
        tokenizer,
        max_length=max_length,
    )

    def collate(batch: list[list[int]]) -> LanguageModelBatch:
        return collate_language_model_batch(batch, tokenizer.pad_token_id, device)

    return (
        tokenizer,
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate),
        DataLoader(valid_dataset, batch_size=eval_batch_size, collate_fn=collate),
        DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate),
    )
