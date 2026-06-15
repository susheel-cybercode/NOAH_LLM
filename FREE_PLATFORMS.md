# MAYA AI - Free Platforms Guide
## Where to Train and Run Your LLM for FREE

## 1. Google Colab (RECOMMENDED)
**Best for**: Training medium/large models
**Free Tier**: 
- GPU: NVIDIA T4 (16GB VRAM)
- RAM: 12.7 GB
- Runtime: 12 hours continuous

```bash
# Upload maya_real.py to Colab
# Run:
!pip install torch
from maya_real import train_maya
model = train_maya("medium", "data.txt", epochs=10)
```

**Link**: https://colab.research.google.com

## 2. Kaggle Kernels
**Best for**: GPU training, competitions
**Free Tier**:
- GPU: NVIDIA T4 x2 (30 hours/week)
- RAM: 30 GB
- Storage: 20 GB

**Link**: https://www.kaggle.com/code

## 3. Hugging Face Spaces
**Best for**: Deploying trained models
**Free Tier**:
- CPU: 2 cores
- RAM: 16 GB
- Persistent storage

**Deploy command**:
```bash
pip install huggingface-hub
git push https://huggingface.co/spaces/YOURNAME/maya
```

## 4. GitHub Codespaces
**Best for**: Development, small models
**Free Tier**:
- CPU: 2 cores
- RAM: 4 GB
- 120 hours/month

## 5. Oracle Cloud Free Tier
**Best for**: Long-running training
**Free Forever**:
- CPU: 4 ARM cores
- RAM: 24 GB
- Storage: 200 GB

**Link**: https://www.oracle.com/cloud/free/

## 6. Paperspace Gradient
**Best for**: GPU access
**Free Tier**:
- GPU: NVIDIA M4000
- 6 hours continuous

## 7. RunPod
**Best for**: Serverless GPU
**Pay-as-you-go**: ~$0.20/hour for RTX A5000

## 8. Lambda Cloud
**Best for**: Research
**Free Tier**: $30 credits
- GPU: NVIDIA A10 (24GB)

## Platform Comparison

| Platform | GPU | RAM | Free Hours | Setup |
|----------|-----|-----|-----------|-------|
| Google Colab | T4 | 12GB | 12h/day | Easy |
| Kaggle | T4x2 | 30GB | 30h/week | Medium |
| Hugging Face | No | 16GB | Unlimited | Easy |
| Oracle Cloud | No | 24GB | Unlimited | Hard |
| Paperspace | M4000 | - | 6h | Easy |
| Gitpod | No | 8GB | 50h/month | Easy |

## Quick Start for Colab

1. Upload `maya_real.py`
2. Upload your `training_data.txt`
3. Run:
```python
from maya_real import train_maya
model = train_maya("medium", "training_data.txt", epochs=5)
```

## Model Size vs Resources

| Size | Parameters | RAM Needed | GPU Needed | Best Platform |
|------|-----------|-----------|-----------|-------|
| Small | 16M | 4GB | None | Any |
| Medium | 65M | 8GB | T4 | Colab/Kaggle |
| Large | 150M | 16GB | A10/V100 | Colab Pro/Paperspace |

## Training Data Sources (Free)

1. **Project Gutenberg**: https://www.gutenberg.org
2. **Wikipedia Dumps**: https://dumps.wikimedia.org
3. **Common Crawl**: https://commoncrawl.org
4. **OpenWebText**: https://skylion007.github.io/OpenWebTextCorpus/
5. **Your own data**: Books, articles, code
