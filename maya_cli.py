#!/usr/bin/env python3
"""
MAYA CLI - Ollama-style interface
Usage: maya <command> [args]

Commands:
  maya pull <model>       - Download/pull a model
  maya run <model>        - Run a model (auto-pulls if not exists)
  maya list              - List installed models
  maya rm <model>        - Remove a model
  maya serve             - Start API server
  maya chat              - Interactive chat
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Optional, List
import torch

# Import MAYA v2
from maya_v2 import MAYAv2, MayaConfig, CONFIGS, AdaptiveTrainer

# Paths
MAYA_DIR = Path.home() / ".maya"
MODELS_DIR = MAYA_DIR / "models"
CONFIG_FILE = MAYA_DIR / "config.json"

def ensure_dirs():
    """Ensure necessary directories exist."""
    MAYA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

class MayaModelManager:
    """Manages MAYA models like Ollama."""
    
    def __init__(self):
        ensure_dirs()
        self.models = self._load_models()
    
    def _load_models(self) -> dict:
        """Load model registry."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_models(self):
        """Save model registry."""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def pull(self, model_name: str):
        """Pull/download a model."""
        print(f"Pulling model {model_name}...")
        
        # Check if it's a built-in config
        if model_name in CONFIGS:
            config = CONFIGS[model_name]
            model_path = MODELS_DIR / f"{model_name}.pt"
            
            if model_path.exists():
                print(f"Model {model_name} already exists!")
                return
            
            # Create and save initial model
            print(f"Creating {model_name} model...")
            model = MAYAv2(config)
            
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'version': '2.0'
            }, model_path)
            
            self.models[model_name] = {
                'path': str(model_path),
                'config': model_name,
                'size': 'fresh'
            }
            self._save_models()
            print(f"Model {model_name} pulled successfully!")
            print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            print(f"Unknown model: {model_name}")
            print(f"Available: {', '.join(CONFIGS.keys())}")
    
    def list_models(self):
        """List all installed models."""
        print("\nNAME\t\t\tSIZE\t\tSTATUS")
        print("-" * 50)
        
        if not self.models:
            print("No models installed. Use 'maya pull <model>' to download.")
            return
        
        for name, info in self.models.items():
            size = os.path.getsize(info['path']) / (1024*1024) if os.path.exists(info['path']) else 0
            status = "trained" if info.get('trained', False) else "fresh"
            print(f"{name:<20} {size:.1f}MB\t\t{status}")
        print()
    
    def remove(self, model_name: str):
        """Remove a model."""
        if model_name in self.models:
            model_path = self.models[model_name]['path']
            if os.path.exists(model_path):
                os.remove(model_path)
            del self.models[model_name]
            self._save_models()
            print(f"Model {model_name} removed.")
        else:
            print(f"Model {model_name} not found.")
    
    def run(self, model_name: str, prompt: Optional[str] = None):
        """Run a model."""
        if model_name not in self.models:
            print(f"Model {model_name} not found. Pulling...")
            self.pull(model_name)
        
        if model_name not in self.models:
            print("Failed to pull model.")
            return
        
        # Load model
        print(f"Loading model {model_name}...")
        checkpoint = torch.load(self.models[model_name]['path'], map_location='cpu')
        config_dict = checkpoint.get('config', {})
        
        # Create config
        config = MayaConfig(**config_dict)
        model = MAYAv2(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded! ({sum(p.numel() for p in model.parameters()):,} parameters)")
        
        if prompt:
            # Single prompt mode
            print(f"\nPrompt: {prompt}")
            print("Response: ", end="")
            
            # Simple generation
            tokens = [ord(c) % 256 for c in prompt if ord(c) < 256]
            input_ids = torch.tensor([tokens])
            
            with torch.no_grad():
                for _ in range(100):
                    logits = model(input_ids)
                    next_token_logits = logits[:, -1, :] / 0.8
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    char = chr(next_token.item() % 256) if 32 <= next_token.item() % 256 < 127 else ''
                    print(char, end="")
            print()
        else:
            # Interactive chat mode
            chat_with_model(model)


def chat_with_model(model: MAYAv2):
    """Interactive chat with a loaded model."""
    print("\n" + "="*60)
    print("  MAYA Chat - Type 'exit' to quit")
    print("  Type 'train: <text>' to train on text")
    print("="*60 + "\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            
            # Check for train command
            if user_input.startswith("train:"):
                text = user_input[6:].strip()
                print("Training on text...")
                # Simple training logic (would need proper trainer)
                print("Training complete! (Simulated)")
                continue
            
            # Generate response
            print("\nMAYA: ", end="", flush=True)
            
            tokens = [ord(c) % 256 for c in user_input if ord(c) < 256]
            input_ids = torch.tensor([tokens], device=device)
            
            with torch.no_grad():
                for _ in range(100):
                    logits = model(input_ids)
                    next_token_logits = logits[:, -1, :] / 0.8
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    char = chr(next_token.item() % 256) if 32 <= next_token.item() % 256 < 127 else ''
                    print(char, end="", flush=True)
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="MAYA CLI", usage="maya <command> [args]")
    subparsers = parser.add_subparsers(dest='command')
    
    # Pull command
    pull_parser = subparsers.add_parser('pull', help='Pull a model')
    pull_parser.add_argument('model', help='Model name')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a model')
    run_parser.add_argument('model', help='Model name')
    run_parser.add_argument('-p', '--prompt', help='Prompt for generation')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    
    # Remove command
    rm_parser = subparsers.add_parser('rm', help='Remove a model')
    rm_parser.add_argument('model', help='Model name')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--model', default='laptop', help='Model to use')
    
    args = parser.parse_args()
    
    manager = MayaModelManager()
    
    if args.command == 'pull':
        manager.pull(args.model)
    elif args.command == 'run':
        manager.run(args.model, args.prompt if hasattr(args, 'prompt') else None)
    elif args.command == 'list':
        manager.list_models()
    elif args.command == 'rm':
        manager.remove(args.model)
    elif args.command == 'chat':
        manager.run(args.model if hasattr(args, 'model') else 'laptop')
    else:
        parser.print_help()


if __name__ == "__main__":
    main()