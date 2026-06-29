"""Dataset loading and preprocessing utilities for ATIS transformer fine-tuning."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypedDict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IGNORE_SLOT_ID = -100
DEFAULT_MAX_LENGTH = 128
DEFAULT_BERT_MODEL = "bert-base-uncased"
DEFAULT_GPT2_MODEL = "gpt2"
ModelType = Literal["bert", "gpt2"]


class ATISExample(TypedDict):
    """One ATIS sample."""

    utterance: str
    slots: str
    intent: str


class EncodedSample(TypedDict):
    """One tokenized sample before batching."""

    input_ids: list[int]
    attention_mask: list[int]
    slot_labels: list[int]
    first_subtoken_mask: list[bool]
    intent_label: int
    words: list[str]
    word_slot_labels: list[str]


class Batch(TypedDict):
    """One padded transformer batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    slot_labels: torch.Tensor
    first_subtoken_mask: torch.Tensor
    intent_labels: torch.Tensor
    words: list[list[str]]
    word_slot_labels: list[list[str]]


@dataclass(frozen=True)
class DatasetSplits:
    """Train, development, and test data."""

    train: list[ATISExample]
    dev: list[ATISExample]
    test: list[ATISExample]


@dataclass(frozen=True)
class LabelVocab:
    """Mappings for slot and intent labels."""

    slot2id: dict[str, int]
    intent2id: dict[str, int]

    @property
    def id2slot(self) -> dict[int, str]:
        return {idx: label for label, idx in self.slot2id.items()}

    @property
    def id2intent(self) -> dict[int, str]:
        return {idx: label for label, idx in self.intent2id.items()}


def set_seed(seed: int) -> None:
    """Seed random number generators used by training and dataloading."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def part_b_dir() -> Path:
    """Return the Part B directory."""

    return Path(__file__).resolve().parent


def default_dataset_dir() -> Path:
    """Return the ATIS dataset directory used by Part B.

    The repository stores ATIS under part_A/dataset/ATIS. Part B reuses it unless
    a local part_B/dataset/ATIS folder is added later.
    """

    local_dataset = part_b_dir() / "dataset" / "ATIS"
    if local_dataset.exists():
        return local_dataset
    return part_b_dir().parent / "part_A" / "dataset" / "ATIS"


def default_bin_dir() -> Path:
    """Return the output directory for Part B checkpoints and reports."""

    return part_b_dir() / "bin"


def load_data(path: str | Path) -> list[ATISExample]:
    """Load an ATIS JSON file."""

    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_atis_splits(dataset_dir: str | Path, dev_size: float = 0.10, seed: int = 42) -> DatasetSplits:
    """Load ATIS train/test files and create a stratified dev split."""

    dataset_path = Path(dataset_dir)
    tmp_train_raw = load_data(dataset_path / "train.json")
    test_raw = load_data(dataset_path / "test.json")

    intents = [sample["intent"] for sample in tmp_train_raw]
    intent_counts = Counter(intents)
    stratified_inputs: list[ATISExample] = []
    stratified_labels: list[str] = []
    singletons: list[ATISExample] = []

    for sample, intent in zip(tmp_train_raw, intents):
        if intent_counts[intent] > 1:
            stratified_inputs.append(sample)
            stratified_labels.append(intent)
        else:
            singletons.append(sample)

    train_raw, dev_raw = train_test_split(
        stratified_inputs,
        test_size=dev_size,
        random_state=seed,
        shuffle=True,
        stratify=stratified_labels,
    )
    train_raw = list(train_raw)
    train_raw.extend(singletons)
    return DatasetSplits(train=train_raw, dev=list(dev_raw), test=test_raw)


def _labels_in_first_seen_order(examples: list[ATISExample], key: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for sample in examples:
        values = sample[key].split() if key == "slots" else [sample[key]]
        for value in values:
            if value not in seen:
                seen.add(value)
                labels.append(value)
    return labels


def build_label_vocab(splits: DatasetSplits) -> LabelVocab:
    """Build slot and intent vocabularies."""

    corpus = splits.train + splits.dev + splits.test
    slot_labels = _labels_in_first_seen_order(corpus, "slots")
    intent_labels = _labels_in_first_seen_order(corpus, "intent")
    return LabelVocab(
        slot2id={label: idx for idx, label in enumerate(slot_labels)},
        intent2id={label: idx for idx, label in enumerate(intent_labels)},
    )


def get_tokenizer(model_type: ModelType, model_name: str | None = None):
    """Load and configure the tokenizer for BERT or GPT-2."""

    if model_type == "bert":
        return AutoTokenizer.from_pretrained(model_name or DEFAULT_BERT_MODEL)
    if model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(model_name or DEFAULT_GPT2_MODEL, add_prefix_space=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    raise ValueError(f"Unsupported model type: {model_type}")


class IntentsAndSlotsDataset(Dataset[EncodedSample]):
    """ATIS dataset with word-level slot labels aligned to transformer subtokens."""

    def __init__(
        self,
        dataset: list[ATISExample],
        label_vocab: LabelVocab,
        tokenizer,
        model_type: ModelType,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.samples = dataset
        self.label_vocab = label_vocab
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EncodedSample:
        sample = self.samples[idx]
        words = sample["utterance"].split()
        word_slot_labels = sample["slots"].split()
        if len(words) != len(word_slot_labels):
            raise ValueError(
                f"Utterance/slot length mismatch at sample {idx}: "
                f"{len(words)} words and {len(word_slot_labels)} slots"
            )

        add_special_tokens = self.model_type == "bert"
        max_length = self.max_length if add_special_tokens else self.max_length - 1
        encoded = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=max_length,
        )

        input_ids = list(encoded["input_ids"])
        attention_mask = list(encoded["attention_mask"])
        slot_labels, first_subtoken_mask = self._align_slot_labels(encoded.word_ids(), word_slot_labels)

        if self.model_type == "gpt2":
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)
            slot_labels.append(IGNORE_SLOT_ID)
            first_subtoken_mask.append(False)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_labels": slot_labels,
            "first_subtoken_mask": first_subtoken_mask,
            "intent_label": self.label_vocab.intent2id[sample["intent"]],
            "words": words,
            "word_slot_labels": word_slot_labels,
        }

    def _align_slot_labels(
        self,
        word_ids: list[int | None],
        word_slot_labels: list[str],
    ) -> tuple[list[int], list[bool]]:
        """Assign slot labels only to first subtokens and mask the rest."""

        slot_labels: list[int] = []
        first_subtoken_mask: list[bool] = []
        previous_word_id: int | None = None

        for word_id in word_ids:
            is_first_subtoken = word_id is not None and word_id != previous_word_id
            first_subtoken_mask.append(is_first_subtoken)
            if is_first_subtoken:
                slot_labels.append(self.label_vocab.slot2id[word_slot_labels[word_id]])
            else:
                slot_labels.append(IGNORE_SLOT_ID)
            previous_word_id = word_id

        return slot_labels, first_subtoken_mask


def make_collate_fn(pad_token_id: int, device: torch.device = DEVICE) -> Callable[[list[EncodedSample]], Batch]:
    """Create a collate function that pads tokenized samples."""

    def collate_fn(samples: list[EncodedSample]) -> Batch:
        max_len = max(len(sample["input_ids"]) for sample in samples)
        batch_size = len(samples)

        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        slot_labels = torch.full((batch_size, max_len), IGNORE_SLOT_ID, dtype=torch.long)
        first_subtoken_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        intent_labels = torch.tensor([sample["intent_label"] for sample in samples], dtype=torch.long)

        for row, sample in enumerate(samples):
            length = len(sample["input_ids"])
            input_ids[row, :length] = torch.tensor(sample["input_ids"], dtype=torch.long)
            attention_mask[row, :length] = torch.tensor(sample["attention_mask"], dtype=torch.long)
            slot_labels[row, :length] = torch.tensor(sample["slot_labels"], dtype=torch.long)
            first_subtoken_mask[row, :length] = torch.tensor(sample["first_subtoken_mask"], dtype=torch.bool)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "slot_labels": slot_labels.to(device),
            "first_subtoken_mask": first_subtoken_mask.to(device),
            "intent_labels": intent_labels.to(device),
            "words": [sample["words"] for sample in samples],
            "word_slot_labels": [sample["word_slot_labels"] for sample in samples],
        }

    return collate_fn


def build_dataloaders(
    splits: DatasetSplits,
    label_vocab: LabelVocab,
    tokenizer,
    model_type: ModelType,
    batch_size: int,
    eval_batch_size: int,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: torch.device = DEVICE,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/dev/test dataloaders."""

    train_dataset = IntentsAndSlotsDataset(splits.train, label_vocab, tokenizer, model_type, max_length)
    dev_dataset = IntentsAndSlotsDataset(splits.dev, label_vocab, tokenizer, model_type, max_length)
    test_dataset = IntentsAndSlotsDataset(splits.test, label_vocab, tokenizer, model_type, max_length)
    collate_fn = make_collate_fn(tokenizer.pad_token_id, device)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, dev_loader, test_loader
