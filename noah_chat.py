#!/usr/bin/env python3
"""
NOAH Chat Interface - Interactive chat with trained NOAH model.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from noah_model import NoahLLM, NoahConfig, NOAH_CONFIGS, NoahTokenizer


def chat_with_noah(model_size: str = "small", device: str = "auto"):
    """Interactive chat with NOAH."""

    config = NOAH_CONFIGS[model_size]

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    tokenizer_path = Path("training_data") / "noah_tokenizer.json"
    if not tokenizer_path.exists():
        print("ERROR: Tokenizer not found. Run: python noah_tokenizer.py")
        return

    tokenizer = NoahTokenizer(str(tokenizer_path))
    config.vocab_size = tokenizer.vocab_size

    model = NoahLLM(config).to(device)

    model_path = f"noah_{model_size}.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print("WARNING: No trained model found. Using random weights.")
        print("Train first: python noah_train.py --train --size small")

    print("=" * 60)
    print(f"  NOAH LLM - Interactive Chat")
    print(f"  Size: {model_size.upper()} | Device: {device}")
    print(f"  Vocab: {tokenizer.vocab_size} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print("  Commands: 'exit' to quit, 'clear' to reset context")
    print("=" * 60)

    conversation_history = []

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if not prompt:
                continue

            if prompt.lower() in ['exit', 'quit', 'q']:
                print("\nNOAH: Goodbye!")
                break

            if prompt.lower() == 'clear':
                conversation_history = []
                print("NOAH: Conversation history cleared.")
                continue

            conversation_history.append({"role": "user", "content": prompt})

            # Build context
            context_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_history])
            context_text += "\nassistant:"

            input_ids = torch.tensor([[tokenizer.cls_token_id] + tokenizer.encode(context_text)], device=device)

            print("\nNOAH: ", end="", flush=True)

            model.eval()
            with torch.no_grad():
                generated = model.generate(input_ids, max_tokens=200, temperature=0.8)
                response_ids = generated[0][len(input_ids[0]):].tolist()
                response = tokenizer.decode(response_ids)
                print(response, flush=True)

            conversation_history.append({"role": "assistant", "content": response})

            # Limit history
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        except KeyboardInterrupt:
            print("\n\nNOAH: Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="NOAH LLM Chat")
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    chat_with_noah(args.size, args.device)


if __name__ == "__main__":
    main()