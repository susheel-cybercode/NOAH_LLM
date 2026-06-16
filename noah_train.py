#!/usr/bin/env python3
"""
NOAH Training Script - Trains NOAH LLM on custom data.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from noah_model import NoahLLM, NoahConfig, NOAH_CONFIGS, NoahTokenizer, TextDataset


def train_noah(
    model_size: str = "small",
    data_dir: str = "training_data",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    device: str = "auto"
):
    """Train NOAH model."""

    config = NOAH_CONFIGS[model_size]

    print(f"\n{'='*60}")
    print(f"  Training NOAH-{model_size.upper()}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Parameters: ~{config.params_count:,}")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}")
    print(f"{'='*60}\n")

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer_path = Path(data_dir) / "noah_tokenizer.json"
    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Run: python noah_tokenizer.py")
        return

    tokenizer = NoahTokenizer(str(tokenizer_path))
    config.vocab_size = tokenizer.vocab_size

    model = NoahLLM(config).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

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

    save_path = f"noah_{model_size}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': tokenizer.vocab_size,
        'model_size': model_size,
    }, save_path)
    print(f"Model saved to {save_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="NOAH LLM Training")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--data", type=str, default="training_data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    train_noah(args.size, args.data, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()