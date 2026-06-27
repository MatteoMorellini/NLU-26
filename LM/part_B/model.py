"""GPT-2 with manually implemented LoRA attention adapters."""

import math
import os
from pathlib import Path
from typing import Optional, Tuple, Union

PART_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PART_DIR / "hf_cache"

os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_DIR / "hub"))

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


class CustomGPT2Attention(GPT2Attention):
    """GPT-2 attention layer with trainable LoRA adapters on query, key, and value."""

    def __init__(
        self,
        config: GPT2Config,
        rank: int = 8,
        alpha: float = 16.0,
        lora_dropout: float = 0.0,
        is_cross_attention: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(
            config=config,
            is_cross_attention=is_cross_attention,
            layer_idx=layer_idx,
        )
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.lora_query_a = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_query_b = nn.Linear(rank, self.embed_dim, bias=False)
        self.lora_key_a = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_key_b = nn.Linear(rank, self.embed_dim, bias=False)
        self.lora_value_a = nn.Linear(self.embed_dim, rank, bias=False)
        self.lora_value_b = nn.Linear(rank, self.embed_dim, bias=False)
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        """Initialize LoRA with random down-projections and zero up-projections."""

        for down_projection in (
            self.lora_query_a,
            self.lora_key_a,
            self.lora_value_a,
        ):
            nn.init.kaiming_uniform_(down_projection.weight, a=math.sqrt(5))
        for up_projection in (
            self.lora_query_b,
            self.lora_key_b,
            self.lora_value_b,
        ):
            nn.init.zeros_(up_projection.weight)

    def enable_lora_training(self) -> None:
        """Freeze pretrained attention weights and leave only LoRA adapters trainable."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (
            self.lora_query_a,
            self.lora_query_b,
            self.lora_key_a,
            self.lora_key_b,
            self.lora_value_a,
            self.lora_value_b,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _lora_update(
        self,
        hidden_states: torch.Tensor,
        down_projection: nn.Linear,
        up_projection: nn.Linear,
    ) -> torch.Tensor:
        """Return the scaled low-rank update for one attention projection."""

        dropped = self.lora_dropout(hidden_states)
        return up_projection(down_projection(dropped)) * self.scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], ...]:
        """Run GPT-2 attention with LoRA updates added to Q, K, and V."""

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have "
                    "to be defined. Please instantiate with is_cross_attention=True."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            query = query + self._lora_update(
                hidden_states,
                self.lora_query_a,
                self.lora_query_b,
            )
            key = key + self._lora_update(
                encoder_hidden_states,
                self.lora_key_a,
                self.lora_key_b,
            )
            value = value + self._lora_update(
                encoder_hidden_states,
                self.lora_value_a,
                self.lora_value_b,
            )
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            query = query + self._lora_update(
                hidden_states,
                self.lora_query_a,
                self.lora_query_b,
            )
            key = key + self._lora_update(
                hidden_states,
                self.lora_key_a,
                self.lora_key_b,
            )
            value = value + self._lora_update(
                hidden_states,
                self.lora_value_a,
                self.lora_value_b,
            )

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value) if use_cache is True else None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query,
                key,
                value,
                attention_mask,
                head_mask,
            )
        else:
            attn_output, attn_weights = self._attn(
                query,
                key,
                value,
                attention_mask,
                head_mask,
            )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class GPT2_LoRA(GPT2LMHeadModel):
    """Pretrained GPT-2 with manually added LoRA adapters in every attention block."""

    def __init__(
        self,
        config: GPT2Config,
        rank: int = 8,
        alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        self.lora_dropout = lora_dropout
        self._replace_attention_layers(
            config=config,
            rank=rank,
            alpha=alpha,
            lora_dropout=lora_dropout,
        )
        self._freeze_base_model()

    def _replace_attention_layers(
        self,
        config: GPT2Config,
        rank: int,
        alpha: float,
        lora_dropout: float,
    ) -> None:
        """Replace each pretrained attention module with a LoRA-aware subclass."""

        for block in self.transformer.h:
            old_attention = block.attn
            new_attention = CustomGPT2Attention(
                config=config,
                rank=rank,
                alpha=alpha,
                lora_dropout=lora_dropout,
                is_cross_attention=getattr(old_attention, "is_cross_attention", False),
                layer_idx=getattr(old_attention, "layer_idx", None),
            )
            new_attention.load_state_dict(old_attention.state_dict(), strict=False)
            block.attn = new_attention

    def _freeze_base_model(self) -> None:
        """Freeze GPT-2 and make only LoRA parameters trainable."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in self.modules():
            if isinstance(module, CustomGPT2Attention):
                module.enable_lora_training()

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only trainable LoRA parameters for compact checkpointing."""

        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }


def load_gpt2_lora(
    model_name: str = "openai-community/gpt2",
    rank: int = 8,
    alpha: float = 16.0,
    lora_dropout: float = 0.0,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> GPT2_LoRA:
    """Load pretrained GPT-2 weights into the manual LoRA model."""

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    model = GPT2_LoRA.from_pretrained(
        model_name,
        rank=rank,
        alpha=alpha,
        lora_dropout=lora_dropout,
        cache_dir=cache_path,
    )
    model.config.use_cache = False
    return model


GPT_LoRA = GPT2_LoRA
