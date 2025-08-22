import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import List, Dict, Tuple, Optional

from preprocess import Vocab, set_seed
from train import ToyCausalLM, MiMoTAKVManager


def evaluate_sequence_accuracy(model: ToyCausalLM, sequence: List[int], vocab: Vocab, 
                             manager: Optional[MiMoTAKVManager] = None, 
                             enable_sketches: bool = False) -> Dict[str, float]:
    """Evaluate model accuracy on a sequence."""
    model.reset_cache()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i, token_id in enumerate(sequence[:-1]):
            logits, _ = model.step(token_id, manager, enable_sketches)
            
            next_token = sequence[i + 1]
            predicted_token = torch.argmax(logits).item()
            
            if predicted_token == next_token:
                correct_predictions += 1
            total_predictions += 1
            
            if manager is not None and i % manager.R == 0 and i > 0:
                K, V, cache_indices = model.get_cache_tensors(add_noise=False)
                if len(cache_indices) > 0:
                    to_demote, to_evict = manager.plan_demotion_eviction(cache_indices, vocab, memory_pressure=0.3)
                    
                    for cache_idx, new_tier in to_demote:
                        model.demote_tokens([cache_idx], new_tier)
                    
                    if to_evict:
                        model.evict_tokens(to_evict)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return {
        'accuracy': accuracy,
        'correct': correct_predictions,
        'total': total_predictions
    }


def evaluate_memory_usage(model: ToyCausalLM) -> Dict[str, int]:
    """Evaluate memory usage of the model cache."""
    alive_tokens = sum(1 for item in model.cache if item['alive'])
    
    tier_counts = {'fp16': 0, 'q4': 0, 'q2': 0}
    for item in model.cache:
        if item['alive']:
            tier_counts[item['tier']] += 1
    
    return {
        'total_alive': alive_tokens,
        'fp16_count': tier_counts['fp16'],
        'q4_count': tier_counts['q4'],
        'q2_count': tier_counts['q2']
    }


def evaluate_attention_retention(model: ToyCausalLM, sequence: List[int], vocab: Vocab,
                               manager: Optional[MiMoTAKVManager] = None) -> Dict[str, List[float]]:
    """Evaluate attention retention patterns."""
    model.reset_cache()
    attention_patterns = []
    cache_sizes = []
    
    with torch.no_grad():
        for i, token_id in enumerate(sequence):
            logits, attn_weights = model.step(token_id, manager, False)
            
            if attn_weights.numel() > 0:
                attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1).mean().item()
                attention_patterns.append(attn_entropy)
            else:
                attention_patterns.append(0.0)
            
            cache_sizes.append(len([item for item in model.cache if item['alive']]))
            
            if manager is not None and i % manager.R == 0 and i > 0:
                K, V, cache_indices = model.get_cache_tensors(add_noise=False)
                if len(cache_indices) > 0:
                    to_demote, to_evict = manager.plan_demotion_eviction(cache_indices, vocab, memory_pressure=0.4)
                    
                    for cache_idx, new_tier in to_demote:
                        model.demote_tokens([cache_idx], new_tier)
                    
                    if to_evict:
                        model.evict_tokens(to_evict)
    
    return {
        'attention_entropy': attention_patterns,
        'cache_sizes': cache_sizes
    }


def run_baseline_comparison(vocab: Vocab, workloads: Dict[str, List[List[int]]], 
                          device: str = 'cpu') -> Dict[str, Dict[str, float]]:
    """Run comparison between different cache management strategies."""
    print("Running baseline comparison...")
    
    results = {}
    
    configs = {
        'full_cache': {'manager': None, 'window': None},
        'local_window_32': {'manager': None, 'window': 32},
        'local_window_64': {'manager': None, 'window': 64},
        'mimota_basic': {'manager': 'basic', 'window': None},
        'mimota_sketches': {'manager': 'sketches', 'window': None}
    }
    
    for config_name, config in configs.items():
        print(f"  Testing {config_name}...")
        config_results = {}
        
        for task_name, sequences in workloads.items():
            task_accuracies = []
            task_memory_usage = []
            
            for seq in sequences[:3]:  # Test on first 3 sequences per task
                model = ToyCausalLM(vocab, d_model=64, n_heads=4, d_kv=16, device=device)
                
                manager = None
                if config['manager'] == 'basic':
                    manager = MiMoTAKVManager(1, 4, 16, 16, device=device, enable_sketches=False)
                    manager.task_bootstrap(task_name.split('_')[0])  # Extract task type
                elif config['manager'] == 'sketches':
                    manager = MiMoTAKVManager(1, 4, 16, 16, device=device, enable_sketches=True)
                    manager.task_bootstrap(task_name.split('_')[0])
                
                acc_results = evaluate_sequence_accuracy(model, seq, vocab, manager, 
                                                       enable_sketches=(config['manager'] == 'sketches'))
                
                if config['window'] is not None:
                    model.local_window(config['window'])
                
                memory_results = evaluate_memory_usage(model)
                
                task_accuracies.append(acc_results['accuracy'])
                task_memory_usage.append(memory_results['total_alive'])
            
            config_results[f'{task_name}_accuracy'] = np.mean(task_accuracies)
            config_results[f'{task_name}_memory'] = np.mean(task_memory_usage)
        
        results[config_name] = config_results
    
    return results


def create_performance_plots(results: Dict[str, Dict[str, float]], save_dir: str):
    """Create performance comparison plots."""
    print("Creating performance plots...")
    
    tasks = ['rag_qa', 'summarization', 'coding', 'needle']
    configs = list(results.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    accuracy_data = {}
    for task in tasks:
        accuracy_data[task] = [results[config][f'{task}_accuracy'] for config in configs]
    
    x = np.arange(len(tasks))
    width = 0.15
    
    for i, config in enumerate(configs):
        accuracies = [results[config][f'{task}_accuracy'] for task in tasks]
        ax1.bar(x + i * width, accuracies, width, label=config, alpha=0.8)
    
    ax1.set_xlabel('Task Type')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Task Accuracy by Cache Management Strategy')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(tasks, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, config in enumerate(configs):
        memory_usage = [results[config][f'{task}_memory'] for task in tasks]
        ax2.bar(x + i * width, memory_usage, width, label=config, alpha=0.8)
    
    ax2.set_xlabel('Task Type')
    ax2.set_ylabel('Average Cache Size')
    ax2.set_title('Memory Usage by Cache Management Strategy')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(tasks, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_comparison.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    for task in tasks:
        x_mem = [results[config][f'{task}_memory'] for config in configs]
        y_acc = [results[config][f'{task}_accuracy'] for config in configs]
        
        ax.scatter(x_mem, y_acc, label=task, s=100, alpha=0.7)
        
        for i, config in enumerate(configs):
            ax.annotate(config, (x_mem[i], y_acc[i]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax.set_xlabel('Average Cache Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Quality vs Memory Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/quality_memory_tradeoff.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/")


def evaluate_model():
    """Main evaluation function."""
    print("Starting MiMoTA-KV evaluation...")
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    from preprocess import preprocess_data
    vocab, workloads = preprocess_data()
    
    results = run_baseline_comparison(vocab, workloads, device)
    
    print("\nResults Summary:")
    print("=" * 60)
    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        for metric, value in config_results.items():
            print(f"  {metric}: {value:.4f}")
    
    save_dir = ".research/iteration1/images"
    create_performance_plots(results, save_dir)
    
    print(f"\nEvaluation completed! Results saved to {save_dir}/")
    return results


if __name__ == "__main__":
    results = evaluate_model()
