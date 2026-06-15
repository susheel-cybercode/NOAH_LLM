#!/usr/bin/env python3
"""
MAYA Training Script
Train the real LLM on your data.
"""
import sys
import argparse
from maya_real import train_maya

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "medium", "large"], default="small")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    train_maya(
        model_size=args.size,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
