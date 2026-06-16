# NOAH LLM - Clean Uncensored Transformer

A completely rewritten, uncensored local LLM. No MAYA legacy code.

## Files

| File | Purpose |
|------|---------|
| `noah_model.py` | Core transformer architecture |
| `noah_tokenizer.py` | BPE tokenizer trainer |
| `noah_train.py` | Training script |
| `noah_chat.py` | Interactive chat |
| `noah_api.py` | OpenAI-compatible API server |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Tokenizer
```bash
python noah_tokenizer.py
```

### 3. Train Model (Kaggle GPU recommended)
```bash
python noah_train.py --size small --epochs 3 --batch-size 16
```

### 4. Chat
```bash
python noah_chat.py --size small
```

### 5. API Server (for Apple of Eden)
```bash
python noah_api.py --model-size small
# Runs on http://localhost:8000
```

## Model Sizes

| Size | Params | Layers | Heads | Dim | FF Dim |
|------|--------|--------|-------|-----|--------|
| Small | ~7M | 4 | 4 | 256 | 1024 |
| Medium | ~40M | 8 | 8 | 512 | 2048 |
| Large | ~150M | 12 | 12 | 768 | 3072 |

## Apple of Eden Integration

```bash
# Terminal 1: Start NOAH API
python noah_api.py

# Terminal 2: Configure Eden
cd /home/susheel/Desktop/theappleofeden
python -c "
from eden.core.config import Config
cfg = Config()
cfg.ai_provider = 'custom_local'
cfg.custom_llm_url = 'http://localhost:8000'
cfg.custom_model = 'noah'
cfg.save()
"

# Terminal 3: Run Eden
python eden_core.py --chat
```

## Training Data

Add `.txt` files to `training_data/` directory. Supports:
- Code (Python, JS, Rust, etc.)
- Cybersecurity docs
- Psychology texts
- Body language research
- Computer vision papers
- Any domain text

Retrain tokenizer and model after adding data:
```bash
python noah_tokenizer.py
python noah_train.py --size small --epochs 3 --batch-size 16
```

## Kaggle Free GPU

1. Go to kaggle.com/code → New Notebook → GPU T4 x2
2. Run:
```python
!git clone https://github.com/susheel-cybercode/NOAH_LLM.git
%cd NOAH_LLM
!pip install -q -r requirements.txt
!python noah_tokenizer.py
!python noah_train.py --size small --epochs 3 --batch-size 16
```
3. File → Save Version → Download model files

## Architecture

- Pre-norm Transformer
- Multi-head causal attention
- GELU feed-forward
- Weight-tied embeddings
- BPE tokenization (16K vocab)
- Top-k + Top-p sampling

## License

MIT