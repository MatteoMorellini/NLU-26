"""GPT-2 with manually implemented LoRA adapters."""

import math
import os
from pathlib import Path

PART_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = PART_DIR / "hf_cache"

os.environ.setdefault("HF_HOME", str(DEFAULT_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_CACHE_DIR / "hub"))

import torch
from torch import nn
from transformers import GPT2LMHeadModel


class LoRAConv1D(nn.Module):
    """LoRA adapter for GPT-2's combined QKV Conv1D projection."""

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        in_features, out_features_total = base_layer.weight.shape
        if out_features_total % 3 != 0:
            raise ValueError("GPT-2 c_attn output size must contain Q, K, and V")
        self.out_features = out_features_total // 3

        self.lora_A = nn.ModuleDict(
            {
                name: nn.Linear(in_features, rank, bias=False)
                for name in ("query", "key", "value")
            }
        )
        self.lora_B = nn.ModuleDict(
            {
                name: nn.Linear(rank, self.out_features, bias=False)
                for name in ("query", "key", "value")
            }
        )
        self.reset_parameters()

        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def reset_parameters(self) -> None:
        """Use the standard LoRA initialization: random A, zero B."""

        for adapter_a in self.lora_A.values():
            nn.init.kaiming_uniform_(adapter_a.weight, a=math.sqrt(5))
        for adapter_b in self.lora_B.values():
            nn.init.zeros_(adapter_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        dropped = self.dropout(x)
        lora_output = torch.cat(
            [
                self.lora_B[name](self.lora_A[name](dropped))
                for name in ("query", "key", "value")
            ],
            dim=-1,
        )
        return base_output + (self.scaling * lora_output)


class GPT_LoRA(nn.Module):
    """Frozen Hugging Face GPT-2 with trainable LoRA adapters on Q, K, and V."""

    def __init__(
        self,
        model_name: str = "gpt2",
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
    ) -> None:
        super().__init__()
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.rank = rank
        self.alpha = alpha
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            model_name,
            cache_dir=cache_path,
        )
        self.gpt2.config.use_cache = False

        for parameter in self.gpt2.parameters():
            parameter.requires_grad = False

        self._add_lora_adapters(rank=rank, alpha=alpha, dropout=dropout)

    def _add_lora_adapters(self, rank: int, alpha: float, dropout: float) -> None:
        for block in self.gpt2.transformer.h:
            block.attn.c_attn = LoRAConv1D(
                base_layer=block.attn.c_attn,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only the trainable LoRA parameters for compact checkpoints."""

        return {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
