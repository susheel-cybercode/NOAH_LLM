#!/usr/bin/env python3
"""
Dataset collector for MAYA training.
Downloads and prepares uncensored data for: coding, cybersecurity, psychology, body language, computer vision.
"""

import os
import json
import requests
import subprocess
from pathlib import Path
from typing import List

DATA_DIR = Path("training_data")
DATA_DIR.mkdir(exist_ok=True)

DOMAINS = {
    "coding": [
        "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore",
    ],
    "cybersecurity": [
        "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-100.txt",
    ],
    "psychology": [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    ],
    "body_language": [],
    "computer_vision": [
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/python/tutorial_code/Introduction/BasicLinearTransforms/basic_linear_transforms_tutorial.py",
    ]
}

def download_file(url: str, dest: Path):
    """Download a file."""
    try:
        print(f"Downloading {url}...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_text(r.text)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  Failed: {e}")

def clone_github_repos():
    """Clone relevant GitHub repos for training data."""
    repos = {
        "coding": [
            "https://github.com/public-apis/public-apis",
            "https://github.com/vinta/awesome-python",
        ],
        "cybersecurity": [
            "https://github.com/danielmiessler/SecLists",
            "https://github.com/OWASP/CheatSheetSeries",
        ],
        "computer_vision": [
            "https://github.com/opencv/opencv",
            "https://github.com/pytorch/vision",
        ]
    }
    
    for domain, urls in repos.items():
        domain_dir = DATA_DIR / domain
        domain_dir.mkdir(exist_ok=True)
        
        for url in urls:
            repo_name = url.split("/")[-1].replace(".git", "")
            repo_path = domain_dir / repo_name
            
            if not repo_path.exists():
                print(f"Cloning {url}...")
                try:
                    subprocess.run(["git", "clone", "--depth", "1", url, str(repo_path)], 
                                 check=True, capture_output=True)
                    print(f"  Cloned to {repo_path}")
                except subprocess.CalledProcessError as e:
                    print(f"  Failed: {e}")
            else:
                print(f"  Already exists: {repo_path}")

def extract_text_from_repos():
    """Extract text/code from cloned repos."""
    output_file = DATA_DIR / "combined_training_data.txt"
    
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as out:
        for domain_dir in DATA_DIR.iterdir():
            if not domain_dir.is_dir():
                continue
            
            print(f"Processing {domain_dir.name}...")
            for ext in ['.py', '.js', '.ts', '.cpp', '.c', '.h', '.md', '.txt', '.rst']:
                for file in domain_dir.rglob(f"*{ext}"):
                    try:
                        text = file.read_text(encoding='utf-8', errors='ignore')
                        if len(text) > 100:  # Skip tiny files
                            out.write(f"\n=== {domain_dir.name}/{file.relative_to(DATA_DIR)} ===\n")
                            out.write(text)
                            out.write("\n")
                    except:
                        pass
    
    print(f"Combined data saved to {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

def download_arxiv_papers():
    """Download arXiv papers for domains."""
    # Using arXiv API
    categories = {
        "cybersecurity": "cs.CR",
        "computer_vision": "cs.CV", 
        "coding": "cs.SE",
        "psychology": "physics.soc-ph",
    }
    
    for domain, cat in categories.items():
        print(f"Fetching arXiv papers for {domain} ({cat})...")
        url = f"http://export.arxiv.org/api/query?search_query=cat:{cat}&start=0&max_results=100"
        try:
            r = requests.get(url, timeout=30)
            dest = DATA_DIR / domain / f"arxiv_{cat.replace('.', '_')}.xml"
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(r.text)
        except Exception as e:
            print(f"  Failed: {e}")

def main():
    print("=" * 60)
    print("MAYA Dataset Collector")
    print("Domains: coding, cybersecurity, psychology, body_language, computer_vision")
    print("=" * 60)
    
    # Download individual files
    for domain, urls in DOMAINS.items():
        if urls:
            domain_dir = DATA_DIR / domain
            domain_dir.mkdir(exist_ok=True)
            for i, url in enumerate(urls):
                download_file(url, domain_dir / f"source_{i}.txt")
    
    # Clone GitHub repos (optional - requires git)
    print("\nCloning GitHub repos (requires git)...")
    clone_github_repos()
    
    # Download arXiv papers
    print("\nDownloading arXiv papers...")
    download_arxiv_papers()
    
    # Extract and combine
    print("\nExtracting text from repos...")
    extract_text_from_repos()
    
    print("\nDone! Training data in training_data/combined_training_data.txt")

if __name__ == "__main__":
    main()