#!/usr/bin/env python3
"""
NOAH LLM - Clean Transformer Architecture
No legacy code, no MAYA references. Pure NOAH.
"""

import math
import json
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer


@dataclass
class NoahConfig:
    """NOAH model configuration."""
    name: str
    vocab_size: int = 16000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1

    @property
    def params_count(self) -> int:
        embed = self.vocab_size * self.d_model
        pos_embed = self.max_seq_len * self.d_model
        layer_params = 4 * self.d_model * self.d_model
        layer_params += self.d_model * self.d_ff + self.d_ff * self.d_model
        layer_params += 2 * self.d_model * 2
        total = embed + pos_embed + self.n_layers * layer_params + self.vocab_size * self.d_model
        return total


NOAH_CONFIGS = {
    "small": NoahConfig("small", d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=512),
    "medium": NoahConfig("medium", d_model=512, n_layers=8, n_heads=8, d_ff=2048, max_seq_len=1024),
    "large": NoahConfig("large", d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_seq_len=2048),
}


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(context)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class NoahLLM(nn.Module):
    """NOAH - Clean Transformer Language Model."""

    def __init__(self, config: NoahConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len: int, device: torch.device):
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        device = x.device

        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        mask = self.create_causal_mask(seq_len, device)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.output(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        self.eval()
        for _ in range(max_tokens):
            logits = self(input_ids[:, -self.config.max_seq_len:])
            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == 2:
                break
        return input_ids


class NoahTokenizer:
    """Wrapper for trained BPE tokenizer."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.unk_token_id = self.tokenizer.token_to_id("[UNK]") or 0
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]") or 1
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]") or 2
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]") or 3

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)


class TextDataset(Dataset):
    """Dataset for training NOAH."""

    def __init__(self, texts: List[str], tokenizer: NoahTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        print("Tokenizing training data...")
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.data.append(chunk)
        print(f"Created {len(self.data)} training chunks")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y