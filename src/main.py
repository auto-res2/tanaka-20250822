#

import os
import sys
import json
import time
import random
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from preprocess import set_seed, SyntheticPromptGenerator, DecodeConfig
from train import TinyLM, AlphaEstimator, train_acceptance_estimator
from evaluate import run_experiment_1, run_experiment_2, run_experiment_3

def main():
    """Main orchestration script for CAVES experiments."""
    print("=" * 80)
    print("CAVES: Constrained, Acceptance-aware, Verifier-Efficient Speculation")
    print("=" * 80)
    
    set_seed(1337)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    output_dir = Path(".research/iteration1/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    vocab_size = 256
    seq_len = 32
    n_prompts = 100  # Small for quick testing
    
    print(f"\nExperiment Parameters:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Number of prompts: {n_prompts}")
    
    print("\nInitializing models...")
    mp = TinyLM(vocab_size=vocab_size, d_model=64, n_layers=6, n_heads=4, d_ff=128).to(device)
    mq = TinyLM(vocab_size=vocab_size, d_model=48, n_layers=4, n_heads=4, d_ff=96).to(device)
    
    print(f"Mp parameters: {sum(p.numel() for p in mp.parameters()):,}")
    print(f"Mq parameters: {sum(p.numel() for p in mq.parameters()):,}")
    
    print("\nGenerating synthetic data...")
    gen_chat = SyntheticPromptGenerator(vocab_size, 'chat', seq_len)
    gen_code = SyntheticPromptGenerator(vocab_size, 'code', seq_len)
    
    prompts_chat = gen_chat.generate(n_prompts // 2)
    prompts_code = gen_code.generate(n_prompts // 2)
    all_prompts = prompts_chat + prompts_code
    
    print(f"Generated {len(all_prompts)} prompts")
    
    try:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Acceptance Estimator Training & Calibration")
        print("=" * 60)
        run_experiment_1(mp, mq, all_prompts, vocab_size, output_dir, device)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Cascaded Block Verification")
        print("=" * 60)
        run_experiment_2(mp, mq, all_prompts, vocab_size, output_dir, device)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Alpha-SLA Controller")
        print("=" * 60)
        run_experiment_3(mp, mq, all_prompts, vocab_size, output_dir, device)
        
        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        pdf_files = list(output_dir.glob("*.pdf"))
        print(f"\nGenerated {len(pdf_files)} PDF files:")
        for pdf_file in sorted(pdf_files):
            print(f"  - {pdf_file.name}")
            
    except Exception as e:
        print(f"\nERROR during experiments: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
