# MAYA AI - The REAL Uncensored LLM

## What I Actually Built

### BEFORE (Fake System):
- ❌ Empty framework with placeholder functions
- ❌ No actual neural network
- ❌ Responses were hardcoded templates
- ❌ No training capability
- ❌ Basically a chatbot with no brain

### AFTER (Real System):
- ✅ Full Transformer architecture from scratch
- ✅ Real neural network with attention mechanism
- ✅ Trainable on any text data
- ✅ Multiple model sizes (Small/Medium/Large)
- ✅ No censorship hardcoded
- ✅ Terminal-based chat interface

## Architecture (HONEST)

```
MAYA LLM Structure:
├── Multi-Head Self-Attention (real)
├── Feed-Forward Networks (real)
├── Layer Normalization (real)
├── Causal Masking (real)
├── Token Embeddings (real)
└── Positional Embeddings (real)
```

### Model Sizes:

| Size | Parameters | RAM | Hardware |
|------|-----------|-----|----------|
| Small | 16M | 4GB | Any PC |
| Medium | 65M | 8GB | GPU recommended |
| Large | 150M | 16GB | GPU required |

## How to Use

### 1. Quick Test (No Training)
```bash
cd "MAYA AI"
source venv/bin/activate
python maya_real.py --size small
```

**Note**: Without training, it generates random characters.
This is NORMAL. You must train it first.

### 2. Train on Your Data

Create a text file `data.txt` with your training data.
Then:
```bash
python train_maya.py --size small --data data.txt --epochs 10
```

### 3. Chat with Trained Model
```bash
python maya_real.py --size small
```

## Free Platforms to Train (Verified Working)

### 1. Google Colab (BEST)
- **Link**: colab.research.google.com
- **GPU**: NVIDIA T4 (16GB)
- **RAM**: 12.7GB
- **Hours**: 12/day
- **Setup**: Upload `maya_real.py`, install torch, run training

### 2. Kaggle
- **Link**: kaggle.com
- **GPU**: 2x T4
- **RAM**: 30GB
- **Hours**: 30/week

### 3. Oracle Cloud (FREE FOREVER)
- **Link**: oracle.com/cloud/free
- **CPU**: 4 ARM cores
- **RAM**: 24GB
- **Cost**: $0 forever

### 4. Hugging Face Spaces
- **Link**: huggingface.co/spaces
- **RAM**: 16GB
- **GPU**: No (CPU only free tier)
- **Best for**: Deploying trained models

## Why Its Truly Uncensored

**No content filters in code**:
- No blocked words list
- No safety classifiers
- No RLHF alignment
- No content moderation

**Censorship depends ONLY on your training data**:
- Train on uncensored data → uncensored model
- Train on technical docs → technical model
- Train on fiction → fiction model

**You have 100% control**.

## Honest Limitations

1. **Size**: 16M-150M parameters (vs GPT-4's 1.7T)
   - Much smaller, but can still learn patterns
   
2. **Training Data**: Needs your own data
   - Public domain books, wikipedia, your writing
   
3. **Speed**: CPU training is slow
   - Use GPU on Colab for 10x speed
   
4. **Quality**: Wont match GPT-4/Claude
   - But it's YOUR model, fully private

## How It Compares

| Feature | MAYA (Ours) | GPT-4 | Llama 3 |
|---------|------------|-------|---------|
| Real LLM | ✅ | ✅ | ✅ |
| Self-hosted | ✅ | ❌ | ✅ |
| Uncensored | ✅ | ❌ | ⚠️ Partial |
| Free to use | ✅ | ❌ | ✅ |
| Parameters | 16M-150M | 1.7T | 8B |
| Private | ✅ | ❌ | ✅ |
| Code visible | ✅ | ❌ | ✅ |

## What You Actually Get

1. **maya_real.py**: Real Transformer model (~300 lines)
2. **train_maya.py**: Training script
3. **FREE_PLATFORMS.md**: Where to train for free
4. **Working code**: Tested and verified

## Next Steps

### Option A: Train Small (Immediate)
```bash
# Train on your computer
python train_maya.py --size small --data your_data.txt --epochs 20
```

### Option B: Train Medium (Recommended)
1. Go to colab.research.google.com
2. Upload maya_real.py and data
3. Run training with GPU

### Option C: Use Without Training (Testing)
```bash
python maya_real.py --size small
# (Generates nonsense, but architecture is real)
```

## The Truth

**This is a REAL, WORKING LLM.**

It uses genuine Transformer architecture with:
- Actual attention mechanism
- Real neural network layers
- Random initialization (needs training)
- Proper backpropagation
- No fake/placeholder code

**What it's NOT**:
- Not GPT-4 quality (needs 100,000x more compute)
- Not pre-trained (you must train it)
- Not going to write Shakespeare (yet)

**What it IS**:
- A real, functional language model
- Yours to train however you want
- Fully open and uncensored
- Educational and extensible
