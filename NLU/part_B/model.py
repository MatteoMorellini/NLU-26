"""Pretrained BERT and GPT-2 models for joint ATIS intent/slot fine-tuning."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Literal

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


ModelType = Literal["bert", "gpt2"]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a pretrained joint intent/slot model."""

    model_type: ModelType
    pretrained_model_name: str
    slots_size: int
    n_intents: int
    pad_token_id: int
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class BERTIntentSlotModel(nn.Module):
    """BERT encoder with token-level slot and CLS-level intent classifiers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config_values = config
        transformer_config = AutoConfig.from_pretrained(
            config.pretrained_model_name,
            num_labels=config.n_intents,
        )
        self.transformer = AutoModel.from_pretrained(config.pretrained_model_name, config=transformer_config)
        hidden_size = transformer_config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.slot_classifier = nn.Linear(hidden_size, config.slots_size)
        self.intent_classifier = nn.Linear(hidden_size, config.n_intents)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        slot_logits = self.slot_classifier(self.dropout(hidden_states))
        cls_state = hidden_states[:, 0, :]
        intent_logits = self.intent_classifier(self.dropout(cls_state))
        return slot_logits, intent_logits


class GPT2IntentSlotModel(nn.Module):
    """GPT-2 decoder with token-level slot and final-token intent classifiers."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config_values = config
        transformer_config = AutoConfig.from_pretrained(
            config.pretrained_model_name,
            pad_token_id=config.pad_token_id,
        )
        self.transformer = AutoModel.from_pretrained(config.pretrained_model_name, config=transformer_config)
        hidden_size = transformer_config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.slot_classifier = nn.Linear(hidden_size, config.slots_size)
        self.intent_classifier = nn.Linear(hidden_size, config.n_intents)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        slot_logits = self.slot_classifier(self.dropout(hidden_states))

        last_token_positions = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        final_states = hidden_states[batch_indices, last_token_positions]
        intent_logits = self.intent_classifier(self.dropout(final_states))
        return slot_logits, intent_logits


def build_model(config: ModelConfig) -> nn.Module:
    """Create the requested pretrained model."""

    if config.model_type == "bert":
        return BERTIntentSlotModel(config)
    if config.model_type == "gpt2":
        return GPT2IntentSlotModel(config)
    raise ValueError(f"Unsupported model type: {config.model_type}")
