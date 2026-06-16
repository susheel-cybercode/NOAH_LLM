#!/usr/bin/env python3
"""
NOAH Tokenizer Trainer - Trains BPE tokenizer on NOAH training data.
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from pathlib import Path


def main():
    data_dir = Path("training_data")
    data_dir.mkdir(exist_ok=True)

    print("Loading training data...")
    texts = []
    for f in data_dir.glob("*.txt"):
        if "tokenizer" not in f.name and "all_text" not in f.name:
            texts.append(f.read_text(encoding='utf-8', errors='ignore'))

    combined_text = "\n\n".join(texts)
    print(f"Total training text: {len(combined_text):,} characters")

    combined_file = data_dir / "all_text_for_tokenizer.txt"
    combined_file.write_text(combined_text)

    print("Training BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=16000,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        min_frequency=2,
        show_progress=True
    )

    tokenizer.train([str(combined_file)], trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )

    output_path = data_dir / "noah_tokenizer.json"
    tokenizer.save(str(output_path))
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    test_text = "What is the system: You are NOAH, an advanced AI. user: What is SQL injection?"
    encoded = tokenizer.encode(test_text)
    print(f"\nTest: {test_text}")
    print(f"Tokens: {encoded.tokens}")
    print(f"IDs: {encoded.ids}")


if __name__ == "__main__":
    main()