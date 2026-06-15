#!/usr/bin/env python3
"""
MAYA LLM - Complete Colab Training Script
Run this entire file in ONE Colab cell.
Mounts Drive, clones repo, trains tokenizer, trains model, saves to Drive.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run(cmd, desc=""):
    """Run command with description."""
    if desc:
        print(f"\n{'='*60}")
        print(f"🔄 {desc}")
        print(f"{'='*60}")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result

def main():
    print("🚀 MAYA LLM - Complete Colab Training")
    print("=" * 60)
    
    # 1. Mount Google Drive
    print("\n📁 Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Drive mounted")
    
    # 2. Setup directories
    work_dir = Path("/content/MAYA_AI")
    drive_save_dir = Path("/content/drive/MyDrive/MAYA_models")
    drive_save_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Clone repo
    if not work_dir.exists():
        run(["git", "clone", "https://github.com/susheel-cybercode/MAYA_AI.git", str(work_dir)], "Cloning repository")
    else:
        print("📂 Repo exists, pulling latest...")
        run(["git", "-C", str(work_dir), "pull"], "Updating repository")
    
    os.chdir(work_dir)
    
    # 4. Install dependencies
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "tokenizers"], "Installing dependencies")
    
    # 5. Verify GPU
    run(["nvidia-smi"], "Checking GPU")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 6. Train BPE tokenizer
    tokenizer_path = work_dir / "training_data" / "maya_bpe_tokenizer.json"
    if not tokenizer_path.exists():
        run([sys.executable, "train_bpe_tokenizer.py"], "Training BPE tokenizer")
    else:
        print("✅ Tokenizer already exists, skipping")
    
    # 7. Train model
    model_size = "small"  # Change to "medium" if you have time
    epochs = 3
    batch_size = 8
    
    model_file = work_dir / f"maya_bpe_{model_size}.pt"
    if not model_file.exists():
        run([
            sys.executable, "maya_bpe.py",
            "--train",
            "--size", model_size,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size)
        ], f"Training {model_size} model for {epochs} epochs")
    else:
        print(f"✅ Model {model_file.name} exists, skipping training")
    
    # 8. Save to Google Drive
    print("\n💾 Saving to Google Drive...")
    files_to_save = [
        f"maya_bpe_{model_size}.pt",
        "training_data/maya_bpe_tokenizer.json"
    ]
    
    for f in files_to_save:
        src = work_dir / f
        dst = drive_save_dir / f
        if src.exists():
            shutil.copy2(src, dst)
            print(f"✅ Saved: {dst}")
        else:
            print(f"⚠️ Not found: {src}")
    
    # 9. Quick test
    print("\n🧪 Quick generation test...")
    test_result = run([
        sys.executable, "-c", f"""
import torch
from maya_bpe import MAYA_LLM, ModelConfig, BPETokenizerWrapper

tokenizer = BPETokenizerWrapper('training_data/maya_bpe_tokenizer.json')
config = ModelConfig('test', vocab_size=tokenizer.vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024, max_seq_len=512)
model = MAYA_LLM(config).to('{device}')

if Path('{model_file}').exists():
    ckpt = torch.load('{model_file}', map_location='{device}')
    model.load_state_dict(ckpt['model_state_dict'])
    print('Model loaded')

model.eval()
with torch.no_grad():
    ids = torch.tensor([[tokenizer.cls_token_id] + tokenizer.encode('Hello')], device='{device}')
    out = model.generate(ids, max_tokens=30, temperature=0.8)
    print('Prompt: Hello')
    print('Generated:', tokenizer.decode(out[0].tolist()))
"""
    ], "Testing generation")
    
    print("\n" + "=" * 60)
    print("🎉 ALL DONE!")
    print("=" * 60)
    print(f"📁 Model saved to: {drive_save_dir}")
    print(f"   - maya_bpe_{model_size}.pt")
    print(f"   - maya_bpe_tokenizer.json")
    print("\n🔄 To resume later:")
    print(f"   1. Copy files back: cp {drive_save_dir}/* /content/MAYA_AI/")
    print(f"   2. Run training again with more epochs")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)