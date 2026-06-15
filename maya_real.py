#!/usr/bin/env python3
"""
MAYA - Real Transformer-based LLM
A fully functional, self-hosted language model with no censorship hardcoded.
"""

import math
import json
import time
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =====================================================================
# Configuration: Small/Medium/Large model sizes
# =====================================================================

@dataclass
class ModelConfig:
    """Configuration for different model sizes."""
    name: str
    vocab_size: int = 50257          # GPT-2 vocab size
    d_model: int = 128               # Embedding dimension
    n_layers: int = 4                # Number of transformer layers
    n_heads: int = 4                 # Number of attention heads
    d_ff: int = 512                  # Feed-forward hidden dimension
    max_seq_len: int = 512           # Maximum sequence length
    dropout: float = 0.1
    
    @property
    def params_count(self) -> int:
        """Estimate parameter count."""
        embed = self.vocab_size * self.d_model
        pos_embed = self.max_seq_len * self.d_model
        
        layer_params = 0
        # Attention: 4 linear layers (Q, K, V, output projection)
        layer_params += 4 * self.d_model * self.d_model
        # FF: 2 linear layers
        layer_params += self.d_model * self.d_ff + self.d_ff * self.d_model
        # Layer norms (2 per layer)
        layer_params += 2 * self.d_model * 2
        
        total = embed + pos_embed + self.n_layers * layer_params + self.vocab_size * self.d_model
        return total

# Define model sizes
MODEL_CONFIGS = {
    "small": ModelConfig(
        name="small",
        d_model=256, n_layers=4, n_heads=4, d_ff=1024,
        max_seq_len=512, dropout=0.1
    ),
    "medium": ModelConfig(
        name="medium",
        d_model=512, n_layers=8, n_heads=8, d_ff=2048,
        max_seq_len=1024, dropout=0.1
    ),
    "large": ModelConfig(
        name="large",
        d_model=768, n_layers=12, n_heads=12, d_ff=3072,
        max_seq_len=2048, dropout=0.1
    ),
}

# =====================================================================
# Real Transformer Components
# =====================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        
        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, attn


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention and FF."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection
        attn_out, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class MAYA_LLM(nn.Module):
    """
    MAYA - Real Transformer Language Model.
    No hardcoded censorship. Behavior determined entirely by training data.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embeddings
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm and output projection
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights with token embedding
        self.output.weight = self.token_embed.weight
        
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
                
    def create_causal_mask(self, seq_len: int, device: torch.device):
        """Create causal (look-ahead) mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask
    
    def forward(self, x: torch.Tensor):
        """
        x: [batch_size, seq_len] token IDs.
        Returns: [batch_size, seq_len, vocab_size] logits.
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Token + positional embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        x = self.token_embed(x) + self.pos_embed(positions)
        x = self.dropout(x)
        
        # Causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm and output
        x = self.norm(x)
        logits = self.output(x)
        
        return logits
    
    def generate(self, prompt: str, tokenizer, 
                 max_tokens: int = 100, 
                 temperature: float = 0.8,
                 top_k: int = 50,
                 device: str = "cpu") -> str:
        """Generate text from a prompt."""
        self.eval()
        
        # Encode prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                logits = self(input_ids)
                
                # Get logits for next token (last position)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for end token (approximate)
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode
        generated = tokenizer.decode(input_ids[0].tolist())
        return generated[len(prompt):] if generated.startswith(prompt) else generated


# =====================================================================
# Simple BPE Tokenizer
# =====================================================================

class SimpleBPETokenizer:
    """
    Simplified BPE tokenizer.
    In production, use tiktoken or transformers GPT2Tokenizer.
    """
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.eos_token_id = 50256  # <|endoftext|>
        self.pad_token_id = 0
        
        # For a real model, load from tiktoken/GPT-2 vocab
        # Here we'll use a simple character-level fallback
        self.char_to_id = {}
        self.id_to_char = {}
        self._build_char_vocab()
        
    def _build_char_vocab(self):
        """Build simple character vocabulary."""
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>?`~\n\t")
        for i, char in enumerate(chars):
            self.char_to_id[char] = i + 1
            self.id_to_char[i + 1] = char
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_id.get(c, 1) for c in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return "".join([self.id_to_char.get(t, "") for t in tokens])


# =====================================================================
# Training Code (Placeholder for now)
# =====================================================================

class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length]
                if len(chunk) == max_length:
                    self.data.append(chunk)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def train_maya(model_size: str = "small", 
               data_path: Optional[str] = None,
               epochs: int = 10,
               batch_size: int = 8,
               learning_rate: float = 3e-4):
    """Train MAYA model."""
    
    # Get config
    config = MODEL_CONFIGS[model_size]
    print(f"\n{'='*60}")
    print(f"  Training MAYA-{model_size.upper()}")
    print(f"  Parameters: ~{config.params_count:,}")
    print(f"{'='*60}\n")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MAYA_LLM(config).to(device)
    
    print(f"Model loaded on {device}")
    print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Dummy data for demonstration
    if data_path is None or not os.path.exists(data_path):
        print("WARNING: No training data provided. Using dummy data.")
        print("For real training, provide a text file.\n")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was nothing.",
            "Artificial intelligence is transforming the world.",
        ] * 100  # Repeat for more data
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        texts = [text[i:i+1000] for i in range(0, len(text)-1000, 500)]
    
    # Setup
    tokenizer = SimpleBPETokenizer()
    dataset = TextDataset(texts, tokenizer, max_length=min(config.max_seq_len, 512))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x)
            loss = criterion(logits.view(-1, config.vocab_size), y.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} complete! Average Loss: {avg_loss:.4f}\n")
    
    # Save model
    save_path = f"maya_{model_size}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, save_path)
    print(f"Model saved to {save_path}")
    
    return model


# =====================================================================
# Interactive Terminal Chat
# =====================================================================

def chat_with_maya(model_size: str = "small"):
    """Interactive chat with MAYA."""
    
    config = MODEL_CONFIGS[model_size]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load or initialize model
    model = MAYA_LLM(config).to(device)
    
    # Try to load pretrained weights
    model_path = f"maya_{model_size}.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!\n")
    else:
        print("WARNING: No trained model found. Using random weights.")
        print("The model will generate nonsense until trained.\n")
    
    tokenizer = SimpleBPETokenizer()
    
    print("=" * 60)
    print("  MAYA AI - Real Transformer LLM")
    print(f"  Size: {model_size.upper()} | Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("  Type 'exit' to quit, 'train' to start training")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\nMAYA: Goodbye!")
                break
            
            if prompt.lower() == 'train':
                print("\nStarting training...")
                train_maya(model_size)
                continue
            
            if prompt.lower() == 'stats':
                print(f"\nModel: MAYA-{model_size.upper()}")
                print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
                print(f"Layers: {config.n_layers}")
                print(f"Heads: {config.n_heads}")
                print(f"Dimension: {config.d_model}")
                continue
            
            # Generate response
            print("\nMAYA: ", end="", flush=True)
            
            model.eval()
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], device=device)
            
            with torch.no_grad():
                max_new = 50
                for _ in range(max_new):
                    logits = model(input_ids)
                    next_token_logits = logits[:, -1, :] / 0.8
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    token = next_token.item()
                    if token == tokenizer.eos_token_id:
                        break
                    
                    # Decode and print
                    char = tokenizer.id_to_char.get(token, "")
                    print(char, end="", flush=True)
            
            print()  # New line
            
        except KeyboardInterrupt:
            print("\n\nMAYA: Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAYA - Real LLM")
    parser.add_argument("--size", choices=["small", "medium", "large"], 
                       default="small", help="Model size")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data", type=str, help="Path to training data")
    
    args = parser.parse_args()
    
    if args.train:
        train_maya(args.size, args.data)
    else:
        chat_with_maya(args.size)