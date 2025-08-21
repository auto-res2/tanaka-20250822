"""
Training module for DuoStream-Mem experiments.

Since DuoStream-Mem is training-free, this module focuses on:
- Model initialization and setup
- Hyperparameter calibration (optional grid search)
- Baseline model preparation
"""

import os
import json
import time
import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

try:
    from .duostream_mem import set_seed, ModelCfg
    from .streaming_llm import StreamingLLM
    from .preprocess import StreamingDataPreprocessor
except ImportError:
    from duostream_mem import set_seed, ModelCfg
    from streaming_llm import StreamingLLM
    from preprocess import StreamingDataPreprocessor


class DuoStreamMemTrainer:
    """Trainer for DuoStream-Mem method (training-free setup and calibration)."""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256, n_layers: int = 6, 
                 n_heads: int = 8, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device
        
        self.models = {}
        self.setup_models()
        
    def setup_models(self):
        """Initialize models for different memory strategies."""
        strategies = ["streaming_llm", "duostream_mem"]
        
        for strategy in strategies:
            model = StreamingLLM(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                device=self.device,
                strategy=strategy
            )
            
            self._init_weights(model)
            model.eval()  # Always in eval mode for our experiments
            
            self.models[strategy] = model
            
    def _init_weights(self, model: nn.Module):
        """Initialize model weights."""
        for name, param in model.named_parameters():
            if 'embed' in name or 'lm_head' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'proj' in name:
                nn.init.xavier_uniform_(param)
            elif 'ln' in name or 'norm' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 1.0)
                    
    def calibrate_hyperparameters(self, calibration_sequences: List[List[int]], 
                                 budget_bytes: int = 64 * 1024 * 1024) -> Dict[str, float]:
        """Optional grid search calibration for DuoAttention hyperparameters."""
        print("Starting hyperparameter calibration...")
        
        model = self.models["duostream_mem"]
        model.set_budget(budget_bytes)
        
        alpha_values = [4.0, 6.0, 8.0]
        beta_values = [0.5, 1.0, 1.5]
        gamma_values = [-4.0, -3.0, -2.0]
        cap_values = [0.3, 0.4, 0.5]
        
        best_params = None
        best_perplexity = float('inf')
        
        calib_sequences = calibration_sequences[:min(20, len(calibration_sequences))]
        
        for alpha in alpha_values:
            for beta in beta_values:
                for gamma in gamma_values:
                    for cap in cap_values:
                        for router in model.duo_routers:
                            router.alpha = alpha
                            router.beta = beta
                            router.gamma = gamma
                            router.cap = cap
                            
                        total_perplexity = 0.0
                        n_evaluated = 0
                        
                        for seq in calib_sequences:
                            if len(seq) < 100:  # Skip very short sequences
                                continue
                                
                            model.reset_cache()
                            perplexity = self._compute_sequence_perplexity(model, seq[:500])  # Truncate for speed
                            
                            if not np.isnan(perplexity) and not np.isinf(perplexity):
                                total_perplexity += perplexity
                                n_evaluated += 1
                                
                        if n_evaluated > 0:
                            avg_perplexity = total_perplexity / n_evaluated
                            if avg_perplexity < best_perplexity:
                                best_perplexity = avg_perplexity
                                best_params = {
                                    "alpha": alpha,
                                    "beta": beta, 
                                    "gamma": gamma,
                                    "cap": cap
                                }
                                
        if best_params is not None:
            print(f"Best calibrated parameters: {best_params}")
            print(f"Best perplexity: {best_perplexity:.4f}")
            
            for router in model.duo_routers:
                router.alpha = best_params["alpha"]
                router.beta = best_params["beta"]
                router.gamma = best_params["gamma"]
                router.cap = best_params["cap"]
        else:
            print("Calibration failed, using default parameters")
            best_params = {"alpha": 6.0, "beta": 1.0, "gamma": -3.0, "cap": 0.4}
            
        return best_params
        
    def _compute_sequence_perplexity(self, model: StreamingLLM, sequence: List[int]) -> float:
        """Compute perplexity for a single sequence."""
        if len(sequence) < 2:
            return float('inf')
            
        total_log_prob = 0.0
        n_tokens = 0
        
        chunk_size = 32
        for i in range(0, len(sequence) - 1, chunk_size):
            end_idx = min(i + chunk_size, len(sequence) - 1)
            input_chunk = torch.tensor(sequence[i:end_idx], device=self.device).unsqueeze(0)
            target_chunk = torch.tensor(sequence[i+1:end_idx+1], device=self.device).unsqueeze(0)
            
            try:
                with torch.no_grad():
                    logits = model(input_chunk)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    for j in range(target_chunk.size(1)):
                        target_token = target_chunk[0, j].item()
                        if 0 <= target_token < self.vocab_size:
                            total_log_prob += log_probs[0, j, target_token].item()
                            n_tokens += 1
                            
            except Exception as e:
                print(f"Error computing perplexity: {e}")
                return float('inf')
                
        if n_tokens == 0:
            return float('inf')
            
        avg_log_prob = total_log_prob / n_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
        
    def setup_byte_budgets(self, budgets_mb: List[int]):
        """Setup byte budgets for all models."""
        for budget_mb in budgets_mb:
            budget_bytes = budget_mb * 1024 * 1024
            print(f"Setting up {budget_mb}MB budget...")
            
            for strategy, model in self.models.items():
                model.set_budget(budget_bytes)
                usage = model.get_memory_usage()
                print(f"  {strategy}: {usage['total_bytes'] / (1024*1024):.1f}MB allocated")
                
    def save_model_configs(self, output_dir: str):
        """Save model configurations and calibrated parameters."""
        os.makedirs(output_dir, exist_ok=True)
        
        config = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "device": self.device,
            "strategies": list(self.models.keys())
        }
        
        if "duostream_mem" in self.models:
            router = self.models["duostream_mem"].duo_routers[0]
            config["duoattention_params"] = {
                "alpha": router.alpha,
                "beta": router.beta,
                "gamma": router.gamma,
                "zeta": router.zeta,
                "delta": router.delta,
                "cap": router.cap,
                "b_proto": router.b_proto
            }
            
        with open(f"{output_dir}/model_config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"Model configuration saved to {output_dir}/model_config.json")


def train_duostream_mem(config_path: str = "config/train_config.json", 
                       data_path: str = "data/streaming_datasets.json",
                       output_dir: str = "models/",
                       device: str = "cpu"):
    """Main training function (setup and calibration)."""
    set_seed(1234)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "vocab_size": 1000,
            "d_model": 256,
            "n_layers": 6,
            "n_heads": 8,
            "budgets_mb": [64, 128, 256],
            "calibrate": True
        }
        
    trainer = DuoStreamMemTrainer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        device=device
    )
    
    trainer.setup_byte_budgets(config["budgets_mb"])
    
    if config.get("calibrate", False) and os.path.exists(data_path):
        with open(data_path, 'r') as f:
            datasets = json.load(f)
            
        calib_sequences = datasets["perplexity"]["val"][:50]  # Subset for speed
        best_params = trainer.calibrate_hyperparameters(calib_sequences)
        
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model_configs(output_dir)
    
    print("Training (setup) completed successfully!")
    return trainer


if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = train_duostream_mem(device=device)
