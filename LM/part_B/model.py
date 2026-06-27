"""GPT-2 with manually implemented LoRA attention adapters."""

from pathlib import Path
from typing import Literal, Optional, Tuple, TypeAlias, Union

from runtime_config import DEFAULT_CACHE_DIR

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

LoRATargets: TypeAlias = Literal["query_value", "query_key_value"]


class LoRALinearUpdate(nn.Module):
    """Low-rank update BA from the LoRA paper, scaled by alpha / rank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, init_std: float) -> None:
        """Initialize A with a Gaussian and B with zeros so BA starts as zero."""

        nn.init.normal_(self.lora_a.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the scaled LoRA update for one frozen projection."""

        return self.lora_b(self.lora_a(x)) * self.scaling


class CustomGPT2Attention(GPT2Attention):
    """GPT-2 attention layer with trainable LoRA adapters on query, key, and value."""

    def __init__(
        self,
        config: GPT2Config,
        rank: int = 8,
        alpha: float = 16.0,
        lora_init_std: float = 0.02,
        lora_targets: LoRATargets = "query_key_value",
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
        self.lora_init_std = lora_init_std
        self.lora_targets = lora_targets
        self.lora_query = LoRALinearUpdate(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            rank=rank,
            alpha=alpha,
            init_std=lora_init_std,
        )
        self.lora_key: LoRALinearUpdate | None = None
        if lora_targets == "query_key_value":
            self.lora_key = LoRALinearUpdate(
                in_features=self.embed_dim,
                out_features=self.embed_dim,
                rank=rank,
                alpha=alpha,
                init_std=lora_init_std,
            )
        self.lora_value = LoRALinearUpdate(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            rank=rank,
            alpha=alpha,
            init_std=lora_init_std,
        )

    def reset_lora_parameters(self) -> None:
        """Reset all LoRA adapters to the paper's zero-start update."""

        modules = [self.lora_query, self.lora_value]
        if self.lora_key is not None:
            modules.append(self.lora_key)

        for module in modules:
            module.reset_parameters(init_std=self.lora_init_std)

    def enable_lora_training(self) -> None:
        """Freeze pretrained attention weights and leave only LoRA adapters trainable."""

        for parameter in self.parameters():
            parameter.requires_grad = False
        trainable_modules = [self.lora_query, self.lora_value]
        if self.lora_key is not None:
            trainable_modules.append(self.lora_key)

        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

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
            query = query + self.lora_query(hidden_states)
            if self.lora_key is not None:
                key = key + self.lora_key(encoder_hidden_states)
            value = value + self.lora_value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            query = query + self.lora_query(hidden_states)
            if self.lora_key is not None:
                key = key + self.lora_key(hidden_states)
            value = value + self.lora_value(hidden_states)

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
        lora_init_std: float = 0.02,
        lora_targets: LoRATargets = "query_key_value",
    ) -> None:
        super().__init__(config)
        self.rank = rank
        self.alpha = alpha
        self.lora_init_std = lora_init_std
        self.lora_targets = lora_targets
        self._replace_attention_layers(
            config=config,
            rank=rank,
            alpha=alpha,
            lora_init_std=lora_init_std,
            lora_targets=lora_targets,
        )
        self._freeze_base_model()

    def _replace_attention_layers(
        self,
        config: GPT2Config,
        rank: int,
        alpha: float,
        lora_init_std: float,
        lora_targets: LoRATargets,
    ) -> None:
        """Replace each pretrained attention module with a LoRA-aware subclass."""

        for block in self.transformer.h:
            old_attention = block.attn
            new_attention = CustomGPT2Attention(
                config=config,
                rank=rank,
                alpha=alpha,
                lora_init_std=lora_init_std,
                lora_targets=lora_targets,
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

    def reset_lora_parameters(self) -> None:
        """Reset all LoRA adapters after loading pretrained GPT-2 weights."""

        for module in self.modules():
            if isinstance(module, CustomGPT2Attention):
                module.reset_lora_parameters()
        self._freeze_base_model()

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
    lora_init_std: float = 0.02,
    lora_targets: LoRATargets = "query_key_value",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> GPT2_LoRA:
    """Load pretrained GPT-2 weights into the manual LoRA model."""

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    model = GPT2_LoRA.from_pretrained(
        model_name,
        rank=rank,
        alpha=alpha,
        lora_init_std=lora_init_std,
        lora_targets=lora_targets,
        cache_dir=cache_path,
    )
    model.config.use_cache = False
    model.reset_lora_parameters()
    return model


GPT_LoRA = GPT2_LoRA
