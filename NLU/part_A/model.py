"""GPT-2 model for joint ATIS slot filling and intent classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the GPT-2 encoder used in this exercise."""

    vocab_size: int
    slots_size: int
    n_intents: int
    pad_token_id: int
    eos_token_id: int
    pos_emb_size: int = 1024
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        """Serialize the config for checkpoints."""

        return asdict(self)


class GPT2IntentSlotModel(nn.Module):
    """GPT-2 architecture with slot and intent heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.config_values = config
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.pos_emb_size,
            n_embd=config.d_model,
            n_head=config.n_heads,
            n_layer=config.num_layers,
            n_inner=config.ff_dim,
            resid_pdrop=config.dropout,
            embd_pdrop=config.dropout,
            attn_pdrop=config.dropout,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            bos_token_id=config.eos_token_id,
        )
        self.transformer = GPT2Model(gpt2_config)
        self.output_dropout = nn.Dropout(config.dropout)
        self.slot_out = nn.Linear(config.d_model, config.slots_size)
        self.intent_out = nn.Linear(config.d_model, config.n_intents)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return slot logits for every token and intent logits for each sequence."""

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        dropped_states = self.output_dropout(hidden_states)
        slot_logits = self.slot_out(dropped_states)

        last_token_positions = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        final_states = hidden_states[batch_indices, last_token_positions]
        intent_logits = self.intent_out(self.output_dropout(final_states))
        return slot_logits, intent_logits


def build_model(config: ModelConfig) -> GPT2IntentSlotModel:
    """Create a GPT-2 intent/slot model."""

    return GPT2IntentSlotModel(config)
