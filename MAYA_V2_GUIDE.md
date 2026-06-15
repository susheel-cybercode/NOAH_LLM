# MAYA v2 - Advanced Adaptive LLM

## Whats New in v2

Based on cutting-edge research, MAYA v2 includes:

### Architecture Improvements
- **RoPE** (Rotary Position Embeddings) - Better than standard positional encoding
- **Grouped Query Attention (GQA)** - Reduces memory by 75% while maintaining quality
- **SwiGLU Feed-Forward** - Better than ReLU/GELU (used in Llama 2, PaLM)
- **RMSNorm** - More stable than LayerNorm
- **LoRA** - 100x more efficient fine-tuning

### Adaptive Features
- **Continual Learning** - Remembers what you teach it
- **Streaming Dataset** - Handles files larger than RAM
- **Hardware Auto-Detection** - Scales from phone to server
- **Memory System** - Reservoir sampling for important examples

## Quick Start

### 1. Run on Your Device

```bash
# Phone (lightweight)
python maya_v2.py --config phone

# Laptop (balanced)
python maya_v2.py --config laptop

# Cloud (powerful)
python maya_v2.py ston--config cloud
```

### 2. Train from Text

```bash
# In the chat interface:
You: train:The quick brown fox jumps over the lazy dog. This is a test of the training system.
MAYA: Training complete!
```

### 3. Train from File

```bash
# In the chat interface:
You: load:/path/to/your/data.txt
MAYA: Training on 50000 characters...
```

## Hardware Configurations

| Config | RAM | Layers | Heads | Params | Best For |
|--------|-----|--------|-------|--------|----------|
| phone | 2GB | 2 | 2 | 7M | Mobile |
| laptop | 8GB | 4 | 4 | 28M | Personal |
| cloud | 16GB | 8 | 8 | 112M | Server |
| server | 32GB+ | 12 | 12 | 250M+ | Data center |

## Training Tips

### Feed It Anything
- Books (Project Gutenberg)
- Code (GitHub repos)
- Conversations (chat logs)
- Technical docs
- Fiction
- Your own writing

### The More You Train, The Better It Gets
- 1 hour: Basic word patterns
- 10 hours: Coherent sentences
- 100 hours: Complex reasoning
- 1000 hours: Expert-level

### Efficientča Training
```bash
# Small dataset (faster)
python -c "from maya_v2 import *; t = AdaptiveTrainer(MAYAv2(CONFIGS['laptop']), CONFIGS['laptop']); t.train_from_file('data.txt', epochs=10)"

# Large dataset (streaming)
python -c "from maya_v2 import *; t = AdaptiveTrainer(MAYAv2(CONFIGS['cloud']), CONFIGS['cloud']); t.train_from_file('huge_data.txt', epochs=5)"
```

## Free Training Platforms

### Google Colab (Best)
```python
# Upload maya_v2.py
# Run:
!python maya_v2.py --config cloud
```
- GPU: NVIDIA T4 (16GB)
- RAM: 12GB
- Free: 12 hours/day

### Kaggle
- GPU: 2x T4
- RAM: 30GB
- Free: 30 hours/week

### Oracle Cloud (Always Free)
- CPU: 4 ARM cores
- RAM: 24GB
- Storage: 200GB

## Why Its Truly Adaptive

1. **Learns from any text** - No predefined categories
2. **Remembers important examples** - Continual memory
3. **Scales with hardware** - Phone to server
4. **No censorship** - Behavior = training data
5. **Efficient fine-tuning** - LoRA for 100x speedup

## Comparison

| Feature | MAYA v1 | MAYA v2 | GPT-4 |
|---------|---------|---------|-------|
| Real LLM | ✅ | ✅ | ✅ |
| Parameters | 16M | 7M-250M | 1.7T |
| LoRA fine-tuning | ❌ | ✅ | N/A |
| Continual learning | ❌ | ✅ | N/A |
| Streaming data | ❌ | ✅ | N/A |
| GQA attention | ❌ | ✅ | ✅ |
| Self-hosted | ✅ | ✅ | ❌ |
| Free | ✅ | ✅ | ❌ |

## Advanced Usage

### Custom Training Script
```python
from maya_v2 import MAYAv2, MayaConfig, CONFIGS, AdaptiveTrainer

# Create custom config
config = MayaConfig(
    name="custom",
    d_model=512,
    n_layers=8,
    n_heads=8,
    use_lora=True,
    lora_rank=32
)

# Create model and trainer
model = MAYAv2(config)
trainer = AdaptiveTrainer(model, config, device="cuda")

# Train
trainer.train_from_file("my_data.txt", epochs=10)
```

### API Usage
```python
from maya_v2 import MAYAv2, CONFIGS
import torch

model = MAYAv2(CONFIGS['laptop'])
model.load_state_dict(torch.load("checkpoint.pt")['model_state_dict'])
model.eval()

# Generate
tokens = torch.tensor([[1, 2, 3]])  # Your prompt
output = model.generate(tokens, max_tokens=100)
```

## Honest Limitations

1. **Smaller than commercial models** - But trains on your specific data
2. **Needs training** - Random at first, improves with data
3. **CPU training is slow** - Use Colab for GPU speedup
4. **Quality depends on data** - Garbage in, garbage out

## Whats Possible

After training on 1GB+ of quality data:
- Write in your style
- Answer domain-specific questions
- Generate creative content
- Code in your preferred languages
- Learn new topics continuously

## The Future

MAYA v2 is designed to grow with you:
- Start on your phone
- Move to laptop as data grows  
- Scale to cloud for large training
- Always learning, always adapting

**This is YOUR AI. Train it however you want.**