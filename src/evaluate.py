"""
Evaluation module for DuoStream-Mem experiments.

This module implements comprehensive evaluation including:
- Streaming perplexity under fixed byte budgets
- Long-range factual recall (Needle-in-a-Haystack)
- Efficiency and stability metrics
- Comparison with baselines
"""

import os
import json
import time
import math
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    from .duostream_mem import set_seed, SyntheticStreamingTask
    from .streaming_llm import StreamingLLM
    from .train import DuoStreamMemTrainer
except ImportError:
    from duostream_mem import set_seed, SyntheticStreamingTask
    from streaming_llm import StreamingLLM
    from train import DuoStreamMemTrainer


class DuoStreamMemEvaluator:
    """Comprehensive evaluator for DuoStream-Mem method."""
    
    def __init__(self, trainer: DuoStreamMemTrainer, output_dir: str = ".research/iteration1/images/"):
        self.trainer = trainer
        self.output_dir = output_dir
        self.results = {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
        
    def evaluate_streaming_perplexity(self, sequences: List[List[int]], 
                                    budgets_mb: List[int] = [64, 128, 256]) -> Dict[str, Any]:
        """Evaluate streaming perplexity under fixed byte budgets."""
        print("Evaluating streaming perplexity...")
        
        results = {
            "budgets_mb": budgets_mb,
            "strategies": list(self.trainer.models.keys()),
            "perplexities": {},
            "memory_usage": {},
            "processing_times": {}
        }
        
        for budget_mb in budgets_mb:
            budget_bytes = budget_mb * 1024 * 1024
            print(f"\nEvaluating {budget_mb}MB budget...")
            
            results["perplexities"][budget_mb] = {}
            results["memory_usage"][budget_mb] = {}
            results["processing_times"][budget_mb] = {}
            
            for strategy, model in self.trainer.models.items():
                print(f"  Strategy: {strategy}")
                
                model.set_budget(budget_bytes)
                
                perplexities = []
                processing_times = []
                memory_usages = []
                
                eval_sequences = sequences[:min(50, len(sequences))]
                
                for seq_idx, sequence in enumerate(tqdm(eval_sequences, desc=f"{strategy}")):
                    if len(sequence) < 100:  # Skip very short sequences
                        continue
                        
                    model.reset_cache()
                    
                    start_time = time.time()
                    perplexity = self._compute_streaming_perplexity(model, sequence)
                    end_time = time.time()
                    
                    if not np.isnan(perplexity) and not np.isinf(perplexity):
                        perplexities.append(perplexity)
                        processing_times.append(end_time - start_time)
                        
                        usage = model.get_memory_usage()
                        memory_usages.append(usage["total_bytes"])
                        
                if perplexities:
                    results["perplexities"][budget_mb][strategy] = {
                        "mean": float(np.mean(perplexities)),
                        "std": float(np.std(perplexities)),
                        "median": float(np.median(perplexities)),
                        "values": perplexities[:10]  # Store first 10 for debugging
                    }
                    results["processing_times"][budget_mb][strategy] = {
                        "mean": float(np.mean(processing_times)),
                        "std": float(np.std(processing_times))
                    }
                    results["memory_usage"][budget_mb][strategy] = {
                        "mean": float(np.mean(memory_usages)),
                        "max": float(np.max(memory_usages)),
                        "budget_bytes": budget_bytes
                    }
                    
                    print(f"    Perplexity: {np.mean(perplexities):.3f} ± {np.std(perplexities):.3f}")
                    print(f"    Memory: {np.mean(memory_usages)/(1024*1024):.1f}MB")
                    print(f"    Time: {np.mean(processing_times):.3f}s per sequence")
                    
        self.results["streaming_perplexity"] = results
        return results
        
    def evaluate_needle_in_haystack(self, needle_sequences: List[Dict]) -> Dict[str, Any]:
        """Evaluate long-range factual recall using needle-in-a-haystack."""
        print("Evaluating needle-in-a-haystack recall...")
        
        results = {
            "position_types": ["start", "middle", "end"],
            "strategies": list(self.trainer.models.keys()),
            "accuracies": {},
            "response_times": {}
        }
        
        sequences_by_pos = {}
        for seq_data in needle_sequences:
            pos_type = seq_data["position_type"]
            if pos_type not in sequences_by_pos:
                sequences_by_pos[pos_type] = []
            sequences_by_pos[pos_type].append(seq_data)
            
        for strategy, model in self.trainer.models.items():
            print(f"\nStrategy: {strategy}")
            
            model.set_budget(128 * 1024 * 1024)
            
            results["accuracies"][strategy] = {}
            results["response_times"][strategy] = {}
            
            for pos_type in results["position_types"]:
                if pos_type not in sequences_by_pos:
                    continue
                    
                correct = 0
                total = 0
                response_times = []
                
                sequences = sequences_by_pos[pos_type][:min(20, len(sequences_by_pos[pos_type]))]
                
                for seq_data in tqdm(sequences, desc=f"{strategy}-{pos_type}"):
                    model.reset_cache()
                    
                    sequence = seq_data["sequence"]
                    needle_value = seq_data["needle_value"]
                    query_pos = seq_data["query_position"]
                    
                    start_time = time.time()
                    
                    input_seq = sequence[:query_pos]
                    predicted_token = self._predict_next_token(model, input_seq)
                    
                    end_time = time.time()
                    
                    if predicted_token == needle_value:
                        correct += 1
                    total += 1
                    response_times.append(end_time - start_time)
                    
                if total > 0:
                    accuracy = correct / total
                    results["accuracies"][strategy][pos_type] = {
                        "accuracy": float(accuracy),
                        "correct": correct,
                        "total": total
                    }
                    results["response_times"][strategy][pos_type] = {
                        "mean": float(np.mean(response_times)),
                        "std": float(np.std(response_times))
                    }
                    
                    print(f"  {pos_type}: {accuracy:.3f} ({correct}/{total})")
                    
        self.results["needle_in_haystack"] = results
        return results
        
    def evaluate_efficiency_scaling(self, ultra_long_sequences: List[List[int]]) -> Dict[str, Any]:
        """Evaluate efficiency and stability on ultra-long sequences."""
        print("Evaluating efficiency and scaling...")
        
        results = {
            "sequence_lengths": [],
            "strategies": list(self.trainer.models.keys()),
            "throughput": {},
            "memory_growth": {},
            "stability_metrics": {}
        }
        
        test_lengths = [1000, 2000, 4000, 8000]
        
        for strategy, model in self.trainer.models.items():
            print(f"\nStrategy: {strategy}")
            
            model.set_budget(256 * 1024 * 1024)
            
            results["throughput"][strategy] = {}
            results["memory_growth"][strategy] = {}
            results["stability_metrics"][strategy] = {}
            
            for target_length in test_lengths:
                print(f"  Testing length: {target_length}")
                
                suitable_sequences = [seq for seq in ultra_long_sequences if len(seq) >= target_length]
                if not suitable_sequences:
                    continue
                    
                test_sequences = suitable_sequences[:min(5, len(suitable_sequences))]
                
                throughputs = []
                memory_traces = []
                perplexity_traces = []
                
                for seq in test_sequences:
                    model.reset_cache()
                    
                    sequence = seq[:target_length]
                    
                    start_time = time.time()
                    perplexity_trace = self._compute_streaming_perplexity_with_trace(model, sequence)
                    end_time = time.time()
                    
                    throughput = len(sequence) / (end_time - start_time)  # tokens/second
                    throughputs.append(throughput)
                    perplexity_traces.append(perplexity_trace)
                    
                    memory_trace = []
                    for i in range(0, len(sequence), 100):
                        usage = model.get_memory_usage()
                        memory_trace.append(usage["total_bytes"])
                    memory_traces.append(memory_trace)
                    
                if throughputs:
                    results["throughput"][strategy][target_length] = {
                        "mean": float(np.mean(throughputs)),
                        "std": float(np.std(throughputs))
                    }
                    
                    if memory_traces:
                        avg_memory_trace = np.mean(memory_traces, axis=0)
                        memory_growth = (avg_memory_trace[-1] - avg_memory_trace[0]) / avg_memory_trace[0]
                        results["memory_growth"][strategy][target_length] = {
                            "growth_ratio": float(memory_growth),
                            "final_mb": float(avg_memory_trace[-1] / (1024*1024))
                        }
                        
                    if perplexity_traces:
                        avg_perplexity_trace = np.mean(perplexity_traces, axis=0)
                        perplexity_variance = np.var(avg_perplexity_trace)
                        results["stability_metrics"][strategy][target_length] = {
                            "perplexity_variance": float(perplexity_variance),
                            "final_perplexity": float(avg_perplexity_trace[-1])
                        }
                        
                    print(f"    Throughput: {np.mean(throughputs):.1f} tokens/s")
                    print(f"    Memory growth: {memory_growth:.3f}")
                    
        results["sequence_lengths"] = test_lengths
        self.results["efficiency_scaling"] = results
        return results
        
    def _compute_streaming_perplexity(self, model: StreamingLLM, sequence: List[int]) -> float:
        """Compute perplexity for streaming sequence."""
        if len(sequence) < 2:
            return float('inf')
            
        total_log_prob = 0.0
        n_tokens = 0
        
        chunk_size = 16
        for i in range(0, len(sequence) - 1, chunk_size):
            end_idx = min(i + chunk_size, len(sequence) - 1)
            input_chunk = torch.tensor(sequence[i:end_idx], device=model.device).unsqueeze(0)
            target_chunk = torch.tensor(sequence[i+1:end_idx+1], device=model.device).unsqueeze(0)
            
            try:
                with torch.no_grad():
                    logits = model(input_chunk)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    for j in range(target_chunk.size(1)):
                        target_token = target_chunk[0, j].item()
                        if 0 <= target_token < model.vocab_size:
                            total_log_prob += log_probs[0, j, target_token].item()
                            n_tokens += 1
                            
            except Exception as e:
                print(f"Error in perplexity computation: {e}")
                return float('inf')
                
        if n_tokens == 0:
            return float('inf')
            
        avg_log_prob = total_log_prob / n_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
        
    def _compute_streaming_perplexity_with_trace(self, model: StreamingLLM, sequence: List[int]) -> List[float]:
        """Compute perplexity trace for stability analysis."""
        if len(sequence) < 2:
            return [float('inf')]
            
        perplexity_trace = []
        chunk_size = 16
        window_size = 100  # Compute perplexity over sliding window
        
        for i in range(0, len(sequence) - 1, chunk_size):
            end_idx = min(i + chunk_size, len(sequence) - 1)
            input_chunk = torch.tensor(sequence[i:end_idx], device=model.device).unsqueeze(0)
            
            try:
                with torch.no_grad():
                    logits = model(input_chunk)
                    
                if i >= window_size:
                    window_start = max(0, i - window_size)
                    window_perplexity = self._compute_streaming_perplexity(model, sequence[window_start:i+chunk_size])
                    perplexity_trace.append(window_perplexity)
                    
            except Exception:
                perplexity_trace.append(float('inf'))
                
        return perplexity_trace
        
    def _predict_next_token(self, model: StreamingLLM, sequence: List[int]) -> int:
        """Predict next token given sequence."""
        if not sequence:
            return 0
            
        chunk_size = 32
        for i in range(0, len(sequence), chunk_size):
            end_idx = min(i + chunk_size, len(sequence))
            input_chunk = torch.tensor(sequence[i:end_idx], device=model.device).unsqueeze(0)
            
            with torch.no_grad():
                logits = model(input_chunk)
                
        next_token_logits = logits[0, -1, :]
        predicted_token = torch.argmax(next_token_logits).item()
        
        return predicted_token
        
    def generate_plots(self):
        """Generate all evaluation plots as high-quality PDFs."""
        print("Generating evaluation plots...")
        
        if "streaming_perplexity" in self.results:
            self._plot_perplexity_vs_budget()
            
        if "needle_in_haystack" in self.results:
            self._plot_needle_accuracy()
            
        if "efficiency_scaling" in self.results:
            self._plot_efficiency_scaling()
            
        self._plot_memory_usage()
        
        print(f"All plots saved to {self.output_dir}")
        
    def _plot_perplexity_vs_budget(self):
        """Plot streaming perplexity vs byte budget."""
        results = self.results["streaming_perplexity"]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        budgets = results["budgets_mb"]
        strategies = results["strategies"]
        
        for strategy in strategies:
            perplexities = []
            errors = []
            
            for budget in budgets:
                if budget in results["perplexities"] and strategy in results["perplexities"][budget]:
                    data = results["perplexities"][budget][strategy]
                    perplexities.append(data["mean"])
                    errors.append(data["std"])
                else:
                    perplexities.append(np.nan)
                    errors.append(0)
                    
            ax.errorbar(budgets, perplexities, yerr=errors, marker='o', linewidth=2, 
                       markersize=8, label=strategy.replace('_', ' ').title(), capsize=5)
            
        ax.set_xlabel('Memory Budget (MB)', fontsize=12)
        ax.set_ylabel('Streaming Perplexity', fontsize=12)
        ax.set_title('Streaming Perplexity vs Memory Budget', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/perplexity_vs_budget.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_needle_accuracy(self):
        """Plot needle-in-a-haystack accuracy by position."""
        results = self.results["needle_in_haystack"]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        position_types = results["position_types"]
        strategies = results["strategies"]
        
        x = np.arange(len(position_types))
        width = 0.35
        
        for i, strategy in enumerate(strategies):
            accuracies = []
            for pos_type in position_types:
                if strategy in results["accuracies"] and pos_type in results["accuracies"][strategy]:
                    accuracies.append(results["accuracies"][strategy][pos_type]["accuracy"])
                else:
                    accuracies.append(0)
                    
            ax.bar(x + i * width, accuracies, width, label=strategy.replace('_', ' ').title(), alpha=0.8)
            
        ax.set_xlabel('Needle Position', fontsize=12)
        ax.set_ylabel('Recall Accuracy', fontsize=12)
        ax.set_title('Long-Range Factual Recall (Needle-in-a-Haystack)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([pos.title() for pos in position_types])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/needle_accuracy.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_efficiency_scaling(self):
        """Plot efficiency scaling with sequence length."""
        results = self.results["efficiency_scaling"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        lengths = results["sequence_lengths"]
        strategies = results["strategies"]
        
        for strategy in strategies:
            throughputs = []
            for length in lengths:
                if strategy in results["throughput"] and length in results["throughput"][strategy]:
                    throughputs.append(results["throughput"][strategy][length]["mean"])
                else:
                    throughputs.append(np.nan)
                    
            ax1.plot(lengths, throughputs, marker='o', linewidth=2, markersize=8, 
                    label=strategy.replace('_', ' ').title())
            
        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Throughput (tokens/s)', fontsize=12)
        ax1.set_title('Processing Throughput vs Sequence Length', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        for strategy in strategies:
            memory_growth = []
            for length in lengths:
                if strategy in results["memory_growth"] and length in results["memory_growth"][strategy]:
                    memory_growth.append(results["memory_growth"][strategy][length]["final_mb"])
                else:
                    memory_growth.append(np.nan)
                    
            ax2.plot(lengths, memory_growth, marker='s', linewidth=2, markersize=8,
                    label=strategy.replace('_', ' ').title())
            
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax2.set_title('Memory Usage vs Sequence Length', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/efficiency_scaling.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_memory_usage(self):
        """Plot memory usage breakdown."""
        if "streaming_perplexity" not in self.results:
            return
            
        results = self.results["streaming_perplexity"]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        budgets = results["budgets_mb"]
        strategies = results["strategies"]
        
        x = np.arange(len(budgets))
        width = 0.35
        
        for i, strategy in enumerate(strategies):
            memory_usage = []
            budget_limits = []
            
            for budget in budgets:
                if budget in results["memory_usage"] and strategy in results["memory_usage"][budget]:
                    usage_mb = results["memory_usage"][budget][strategy]["mean"] / (1024 * 1024)
                    memory_usage.append(usage_mb)
                    budget_limits.append(budget)
                else:
                    memory_usage.append(0)
                    budget_limits.append(budget)
                    
            bars = ax.bar(x + i * width, memory_usage, width, 
                         label=f'{strategy.replace("_", " ").title()} (Actual)', alpha=0.8)
            
            if i == 0:  # Only add once
                ax.plot(x + width/2, budget_limits, 'r--', linewidth=2, 
                       label='Budget Limit', alpha=0.7)
                
        ax.set_xlabel('Memory Budget (MB)', fontsize=12)
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Usage vs Budget Allocation', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f'{b}MB' for b in budgets])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/memory_usage.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, filepath: Optional[str] = None):
        """Save all evaluation results to JSON."""
        if filepath is None:
            filepath = f"{self.output_dir}/evaluation_results.json"
            
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results saved to {filepath}")
        
    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("DUOSTREAM-MEM EVALUATION SUMMARY")
        print("="*60)
        
        if "streaming_perplexity" in self.results:
            print("\n1. STREAMING PERPLEXITY RESULTS:")
            results = self.results["streaming_perplexity"]
            for budget in results["budgets_mb"]:
                print(f"\n  Budget: {budget}MB")
                for strategy in results["strategies"]:
                    if budget in results["perplexities"] and strategy in results["perplexities"][budget]:
                        data = results["perplexities"][budget][strategy]
                        print(f"    {strategy:15}: {data['mean']:.3f} ± {data['std']:.3f}")
                        
        if "needle_in_haystack" in self.results:
            print("\n2. NEEDLE-IN-A-HAYSTACK RESULTS:")
            results = self.results["needle_in_haystack"]
            for strategy in results["strategies"]:
                print(f"\n  {strategy}:")
                for pos_type in results["position_types"]:
                    if strategy in results["accuracies"] and pos_type in results["accuracies"][strategy]:
                        data = results["accuracies"][strategy][pos_type]
                        print(f"    {pos_type:8}: {data['accuracy']:.3f} ({data['correct']}/{data['total']})")
                        
        if "efficiency_scaling" in self.results:
            print("\n3. EFFICIENCY SCALING RESULTS:")
            results = self.results["efficiency_scaling"]
            for strategy in results["strategies"]:
                print(f"\n  {strategy}:")
                for length in results["sequence_lengths"]:
                    if strategy in results["throughput"] and length in results["throughput"][strategy]:
                        throughput = results["throughput"][strategy][length]["mean"]
                        print(f"    Length {length:5}: {throughput:.1f} tokens/s")
                        
        print("\n" + "="*60)


def evaluate_duostream_mem(trainer: DuoStreamMemTrainer, 
                          data_path: str = "data/streaming_datasets.json",
                          output_dir: str = ".research/iteration1/images/") -> DuoStreamMemEvaluator:
    """Main evaluation function."""
    print("Starting DuoStream-Mem evaluation...")
    
    with open(data_path, 'r') as f:
        datasets = json.load(f)
        
    evaluator = DuoStreamMemEvaluator(trainer, output_dir)
    
    print("\n" + "="*50)
    print("PLAN 1: STREAMING PERPLEXITY EVALUATION")
    print("="*50)
    evaluator.evaluate_streaming_perplexity(
        datasets["perplexity"]["test"], 
        budgets_mb=[64, 128, 256]
    )
    
    print("\n" + "="*50)
    print("PLAN 2: NEEDLE-IN-A-HAYSTACK EVALUATION")
    print("="*50)
    evaluator.evaluate_needle_in_haystack(datasets["needle"]["test"])
    
    print("\n" + "="*50)
    print("PLAN 3: EFFICIENCY SCALING EVALUATION")
    print("="*50)
    evaluator.evaluate_efficiency_scaling(datasets["ultra_long"]["test"])
    
    evaluator.generate_plots()
    evaluator.save_results()
    evaluator.print_summary()
    
    return evaluator


if __name__ == "__main__":
    import torch
    from train import train_duostream_mem
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = train_duostream_mem(device=device)
    
    evaluator = evaluate_duostream_mem(trainer)
