"""Dataset loading and preprocessing utilities for the Penn Treebank LM task."""

import random
from dataclasses import dataclass
from pathlib import Path

from runtime_config import DEFAULT_CACHE_DIR

IGNORE_INDEX = -100
PTB_EOS_TOKEN = "<eos>"
PART_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PART_DIR / "dataset" / "PennTreeBank"

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(frozen=True)
class DatasetPaths:
    """Paths for the Penn Treebank train, validation, and test splits."""

    train: Path
    valid: Path
    test: Path


@dataclass(frozen=True)
class LanguageModelBatch:
    """A padded language-modeling batch."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    n_tokens: torch.Tensor


class PennTreeBankDataset(Dataset[list[int]]):
    """Penn Treebank split encoded with the GPT-2 BPE tokenizer."""

    def __init__(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.sentences = [
            tokenizer.encode(
                f"{sentence} {PTB_EOS_TOKEN}",
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
            for sentence in corpus
        ]
        self.sentences = [sentence for sentence in self.sentences if len(sentence) >= 2]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> list[int]:
        return self.sentences[idx]


def get_dataset_paths(dataset_dir: Path = DEFAULT_DATASET_DIR) -> DatasetPaths:
    """Return the expected Penn Treebank split paths."""

    return DatasetPaths(
        train=dataset_dir / "ptb.train.txt",
        valid=dataset_dir / "ptb.valid.txt",
        test=dataset_dir / "ptb.test.txt",
    )


def read_corpus(path: Path) -> list[str]:
    """Read non-empty text lines."""

    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    """Seed random number generators used by training and dataloading."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the required GPT-2 BPE tokenizer."""

    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def collate_language_model_batch(
    batch: list[list[int]],
    pad_token_id: int,
    device: torch.device,
) -> LanguageModelBatch:
    """Pad encoded sentences and create shifted inputs and labels."""

    max_len = max(len(sentence) for sentence in batch)
    padded = [
        sentence + [pad_token_id] * (max_len - len(sentence))
        for sentence in batch
    ]
    token_ids = torch.tensor(padded, dtype=torch.long, device=device)
    label_mask = torch.zeros_like(token_ids)
    for row, sentence in enumerate(batch):
        label_mask[row, : len(sentence)] = 1

    input_ids = token_ids[:, :-1]
    labels = token_ids[:, 1:]
    label_mask = label_mask[:, 1:]
    labels = labels.masked_fill(label_mask == 0, IGNORE_INDEX)
    n_tokens = torch.sum(label_mask)
    return LanguageModelBatch(input_ids=input_ids, labels=labels, n_tokens=n_tokens)


def build_dataloaders(
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    max_length: int = 1024,
    seed: int = 42,
) -> tuple[PreTrainedTokenizerBase, DataLoader[LanguageModelBatch], DataLoader[LanguageModelBatch], DataLoader[LanguageModelBatch]]:
    """Load Penn Treebank splits and return tokenizer plus train/dev/test loaders."""

    paths = get_dataset_paths(dataset_dir)
    train_corpus = read_corpus(paths.train)
    valid_corpus = read_corpus(paths.valid)
    test_corpus = read_corpus(paths.test)
    tokenizer = load_tokenizer()

    train_dataset = PennTreeBankDataset(train_corpus, tokenizer, max_length=max_length)
    valid_dataset = PennTreeBankDataset(valid_corpus, tokenizer, max_length=max_length)
    test_dataset = PennTreeBankDataset(test_corpus, tokenizer, max_length=max_length)

    def collate(batch: list[list[int]]) -> LanguageModelBatch:
        return collate_language_model_batch(batch, tokenizer.pad_token_id, device)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return (
        tokenizer,
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=generator),
        DataLoader(valid_dataset, batch_size=eval_batch_size, collate_fn=collate),
        DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate),
    )
