#!/usr/bin/env python3
"""Train a BPE tokenizer on MAYA training data."""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from pathlib import Path

# Load all training data
data_dir = Path("training_data")
texts = []
for f in data_dir.glob("*.txt"):
    if "tokenizer" not in f.name:
        texts.append(f.read_text(encoding='utf-8', errors='ignore'))

combined_text = "\n\n".join(texts)
print(f"Total training text: {len(combined_text):,} characters")

# Save combined text for tokenizer training
combined_file = data_dir / "all_text_for_tokenizer.txt"
combined_file.write_text(combined_text)

# Train BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    min_frequency=2,
    show_progress=True
)

print("Training BPE tokenizer...")
tokenizer.train([str(combined_file)], trainer)

# Add post-processor for BERT-style formatting
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

# Save tokenizer
tokenizer.save(str(data_dir / "maya_bpe_tokenizer.json"))
print(f"Tokenizer saved. Vocab size: {tokenizer.get_vocab_size()}")

# Test encoding
test_text = "What is SQL injection and how to prevent it?"
encoded = tokenizer.encode(test_text)
print(f"Test: {test_text}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")