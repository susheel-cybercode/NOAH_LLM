#!/usr/bin/env python3
"""
MAYA with BPE Tokenizer - Real Transformer LLM
Uses trained BPE tokenizer for subword tokenization.
"""

import math
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tokenizers import Tokenizer


@dataclass
class ModelConfig:
    name: str
    vocab_size: int = 16000
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
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


MODEL_CONFIGS = {
    "small": ModelConfig("small", d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=512),
    "medium": ModelConfig("medium", d_model=512, n_layers=8, n_heads=8, d_ff=2048, max_seq_len=1024),
    "large": ModelConfig("large", d_model=768, n_layers=12, n_heads=12, d_ff=3072, max_seq_len=2048),
}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class MAYA_LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.token_embed.weight
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
        x = self.token_embed(x) + self.pos_embed(positions)
        x = self.dropout(x)
        mask = self.create_causal_mask(seq_len, device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        return self.output(x)

    def generate(self, input_ids: torch.Tensor, max_tokens: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
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


class BPETokenizerWrapper:
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
    def __init__(self, texts: List[str], tokenizer: BPETokenizerWrapper, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        print("Tokenizing data...")
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


def train_maya(model_size: str = "small",
               data_dir: str = "training_data",
               epochs: int = 3,
               batch_size: int = 4,
               learning_rate: float = 3e-4):

    config = MODEL_CONFIGS[model_size]
    print(f"\n{'='*60}")
    print(f"  Training MAYA-{model_size.upper()} with BPE Tokenizer")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Parameters: ~{config.params_count:,}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAYA_LLM(config).to(device)
    print(f"Model loaded on {device}")
    print(f"Actual params: {sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer
    tokenizer_path = Path(data_dir) / "maya_bpe_tokenizer.json"
    if not tokenizer_path.exists():
        print("ERROR: BPE tokenizer not found. Run train_bpe_tokenizer.py first.")
        return

    tokenizer = BPETokenizerWrapper(str(tokenizer_path))
    config.vocab_size = tokenizer.vocab_size
    model.config.vocab_size = tokenizer.vocab_size
    # Resize embeddings
    model.token_embed = nn.Embedding(tokenizer.vocab_size, config.d_model).to(device)
    model.output = nn.Linear(config.d_model, tokenizer.vocab_size, bias=False).to(device)
    model.output.weight = model.token_embed.weight

    # Load all text files
    texts = []
    for f in Path(data_dir).glob("*.txt"):
        if "tokenizer" not in f.name and "all_text" not in f.name:
            texts.append(f.read_text(encoding='utf-8', errors='ignore'))

    print(f"Loaded {len(texts)} text files")

    dataset = TextDataset(texts, tokenizer, max_length=min(config.max_seq_len, 512))
    if len(dataset) == 0:
        print("No training data created!")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} complete! Average Loss: {avg_loss:.4f}\n")

    save_path = f"maya_bpe_{model_size}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': tokenizer.vocab_size,
    }, save_path)
    print(f"Model saved to {save_path}")
    return model


def chat_with_maya(model_size: str = "small"):
    config = MODEL_CONFIGS[model_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = Path("training_data") / "maya_bpe_tokenizer.json"
    tokenizer = BPETokenizerWrapper(str(tokenizer_path))
    config.vocab_size = tokenizer.vocab_size

    model = MAYA_LLM(config).to(device)
    model_path = f"maya_bpe_{model_size}.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded!")
    else:
        print("WARNING: No trained model found.")

    print("=" * 60)
    print(f"  MAYA BPE - Transformer LLM")
    print(f"  Size: {model_size.upper()} | Device: {device}")
    print(f"  Vocab: {tokenizer.vocab_size} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print("  Type 'exit' to quit")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\nMAYA: Goodbye!")
                break

            print("\nMAYA: ", end="", flush=True)
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            with torch.no_grad():
                generated = model.generate(input_ids, max_tokens=100, temperature=0.8)
                response_ids = generated[0][len(input_ids[0]):].tolist()
                response = tokenizer.decode(response_ids)
                print(response, flush=True)
        except KeyboardInterrupt:
            print("\n\nMAYA: Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAYA BPE - Real LLM")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", type=str, default="training_data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if args.train:
        train_maya(args.size, args.data, args.epochs, args.batch_size)
    else:
        chat_with_maya(args.size)