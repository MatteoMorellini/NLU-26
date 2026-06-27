"""PyTorch model definitions for the language-modeling exercise."""

from runtime_config import configure_runtime_environment

configure_runtime_environment()

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.h_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.h_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / torch.sqrt(torch.tensor(self.h_dim, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        attention = self.attention_dropout(attention)

        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.output_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    """Token-wise feed-forward network used inside each Transformer block."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer decoder block."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff(self.ln2(x))


class GPT2(nn.Module):
    """Small GPT-style causal language model."""

    def __init__(
        self,
        vocab_size: int,
        pos_emb_size: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ) -> None:
        super().__init__()
        self.pos_emb_size = pos_emb_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.pos_emb_size:
            raise ValueError(f"Sequence length {seq_len} exceeds {self.pos_emb_size}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embedding_dropout(x)
        mask = self.mask[:, :, :seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, mask)

        return self.lm_head(self.ln_f(x))
