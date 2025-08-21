#
#

import os
import math
import time
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')  # for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.isotonic import IsotonicRegression
    SK_ISO_AVAILABLE = True
except Exception:
    SK_ISO_AVAILABLE = False

from train import train_consistency_head
from evaluate import evaluate_system
from preprocess import generate_synthetic_data


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_pdf(fig, filename: str):
    if not filename.endswith('.pdf'):
        filename = filename + '.pdf'
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


ESCALATION_STEPS = [
    {"dim": 64,  "k": 10, "hops": 0, "sources": {"local"},                  "ens": 1, "level": 0},
    {"dim": 128, "k": 20, "hops": 0, "sources": {"local"},                  "ens": 1, "level": 1},
    {"dim": 256, "k": 20, "hops": 1, "sources": {"local", "web"},          "ens": 1, "level": 2},
    {"dim": 256, "k": 40, "hops": 1, "sources": {"local", "web"},          "ens": 2, "level": 3},
    {"dim": 512, "k": 50, "hops": 2, "sources": {"local", "web", "kg"},    "ens": 3, "level": 4},
]


@dataclass(frozen=True)
class PlanKey:
    dim: int
    k: int
    hops: int
    sources: Tuple[str, ...]
    ens: int

    def bucket(self) -> Tuple[int, int, int, Tuple[str, ...], int]:
        return (self.dim, self.k, self.hops, tuple(sorted(self.sources)), self.ens)


def main():
    """Main experimental pipeline orchestrating all components."""
    print("=== ReaLiTy-RAG Experimental Pipeline ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    set_seeds(42)
    
    print("\n1. Generating synthetic data...")
    claims, retriever = generate_synthetic_data()
    
    print("\n2. Training consistency head...")
    head, conformal_g = train_consistency_head(claims, device=device)
    
    print("\n3. Evaluating system...")
    results = evaluate_system(head, conformal_g, claims, retriever, device=device)
    
    print("\n4. Updating status...")
    update_status_to_stopped()
    
    print("\n=== Experiments completed successfully! ===")
    return results


def update_status_to_stopped():
    """Update the status_enum to 'stopped' in research_history.json"""
    history_path = ".research/research_history.json"
    
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        if isinstance(history, dict):
            history['status_enum'] = 'stopped'
        elif isinstance(history, list) and len(history) > 0:
            history[-1]['status_enum'] = 'stopped'
        else:
            history = {'status_enum': 'stopped'}
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
        print("Status updated to 'stopped'")
        
    except Exception as e:
        print(f"Warning: Could not update status in research_history.json: {e}")


def test():
    """Quick test function for verification on T4 GPU."""
    print("=== Quick Test Mode ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")
    
    set_seeds(123)
    
    print("Running quick test with reduced parameters...")
    
    try:
        claims, retriever = generate_synthetic_data(n_claims=50, n_docs=100)
        print(f"Generated {len(claims)} claims and retriever with {retriever.docs.shape[0] if retriever.docs is not None else 0} docs")
        
        head, conformal_g = train_consistency_head(claims, device=device, epochs=5, batch_size=16)
        print("Consistency head trained successfully")
        
        results = evaluate_system(head, conformal_g, claims, retriever, device=device, n_test=20)
        print("System evaluation completed")
        
        images_dir = ".research/iteration1/images"
        if os.path.exists(images_dir):
            pdf_files = [f for f in os.listdir(images_dir) if f.endswith('.pdf')]
            print(f"Generated {len(pdf_files)} PDF files: {pdf_files}")
        
        update_status_to_stopped()
        
        print("✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        main()
