import os
import sys
import time
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from preprocess import preprocess_data, set_seed
from train import train_model
from evaluate import evaluate_model


def main():
    """Main experimental script for MiMoTA-KV research."""
    print("=" * 80)
    print("MiMoTA-KV: Mixed-Precision Task-Aware KV Cache Management")
    print("Experimental Implementation and Evaluation")
    print("=" * 80)
    
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PHASE 1: Data Preprocessing")
    print("=" * 80)
    
    vocab, workloads = preprocess_data()
    
    print(f"\nVocabulary created with {vocab.size()} tokens")
    print("Synthetic workloads generated:")
    for task, sequences in workloads.items():
        avg_len = sum(len(seq) for seq in sequences) / len(sequences)
        print(f"  - {task}: {len(sequences)} sequences (avg length: {avg_len:.1f})")
    
    print("\n" + "=" * 80)
    print("PHASE 2: Model Training/Initialization")
    print("=" * 80)
    
    model = train_model(vocab, workloads, device)
    
    print(f"\nToy Causal LM initialized:")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Embedding dim: {model.d_model}")
    print(f"  - Attention heads: {model.n_heads}")
    print(f"  - Key/Value dim: {model.d_k}")
    
    print("\n" + "=" * 80)
    print("PHASE 3: Experimental Evaluation")
    print("=" * 80)
    
    results = evaluate_model()
    
    print("\n" + "=" * 80)
    print("PHASE 4: Results Analysis")
    print("=" * 80)
    
    print("\nKey Findings:")
    
    tasks = ['rag_qa', 'summarization', 'coding', 'needle']
    configs = list(results.keys())
    
    for task in tasks:
        best_config = max(configs, key=lambda c: results[c][f'{task}_accuracy'])
        best_acc = results[best_config][f'{task}_accuracy']
        best_mem = results[best_config][f'{task}_memory']
        
        print(f"\n{task.upper()}:")
        print(f"  Best configuration: {best_config}")
        print(f"  Accuracy: {best_acc:.4f}")
        print(f"  Memory usage: {best_mem:.1f} tokens")
        
        if 'full_cache' in results:
            full_acc = results['full_cache'][f'{task}_accuracy']
            full_mem = results['full_cache'][f'{task}_memory']
            acc_diff = best_acc - full_acc
            mem_reduction = (full_mem - best_mem) / full_mem * 100
            
            print(f"  vs Full Cache:")
            print(f"    Accuracy difference: {acc_diff:+.4f}")
            print(f"    Memory reduction: {mem_reduction:.1f}%")
    
    print(f"\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    avg_accuracies = {}
    avg_memories = {}
    
    for config in configs:
        task_accs = [results[config][f'{task}_accuracy'] for task in tasks]
        task_mems = [results[config][f'{task}_memory'] for task in tasks]
        avg_accuracies[config] = sum(task_accs) / len(task_accs)
        avg_memories[config] = sum(task_mems) / len(task_mems)
    
    print(f"\nAverage Performance Across All Tasks:")
    for config in configs:
        print(f"  {config}:")
        print(f"    Accuracy: {avg_accuracies[config]:.4f}")
        print(f"    Memory: {avg_memories[config]:.1f} tokens")
    
    print(f"\nPareto Analysis (Quality vs Memory):")
    pareto_configs = []
    for config in configs:
        is_pareto = True
        for other_config in configs:
            if (avg_accuracies[other_config] >= avg_accuracies[config] and 
                avg_memories[other_config] <= avg_memories[config] and
                (avg_accuracies[other_config] > avg_accuracies[config] or 
                 avg_memories[other_config] < avg_memories[config])):
                is_pareto = False
                break
        if is_pareto:
            pareto_configs.append(config)
    
    print(f"  Pareto optimal configurations: {', '.join(pareto_configs)}")
    
    print(f"\nExperimental artifacts saved to:")
    print(f"  - Performance plots: .research/iteration1/images/performance_comparison.pdf")
    print(f"  - Trade-off analysis: .research/iteration1/images/quality_memory_tradeoff.pdf")
    
    print(f"\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    return results


def quick_functional_test():
    """Quick functional test to verify implementation works."""
    print("Running quick functional test...")
    
    try:
        set_seed(42)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from preprocess import Vocab
        vocab = Vocab(n_base=50, n_fact=4)  # Smaller for quick test
        print(f"✓ Vocabulary created: {vocab.size()} tokens")
        
        from train import ToyCausalLM, MiMoTAKVManager
        model = ToyCausalLM(vocab, d_model=32, n_heads=2, d_kv=8, device=device)
        print(f"✓ Model initialized: {sum(p.numel() for p in model.parameters())} params")
        
        manager = MiMoTAKVManager(1, 2, 8, 8, device=device, enable_sketches=True)
        manager.task_bootstrap('rag')
        print("✓ MiMoTA-KV manager initialized")
        
        test_sequence = [1, 5, 10, 15, 20]
        model.reset_cache()
        
        for i, token_id in enumerate(test_sequence):
            logits, attn = model.step(token_id, manager, enable_sketches=True)
            assert logits.shape[0] == vocab.size(), f"Wrong logits shape: {logits.shape}"
            
            if i % 2 == 0 and i > 0:
                K, V, cache_indices = model.get_cache_tensors(add_noise=False)
                if len(cache_indices) > 0:
                    to_demote, to_evict = manager.plan_demotion_eviction(cache_indices, vocab, 0.3)
                    for cache_idx, tier in to_demote:
                        model.demote_tokens([cache_idx], tier)
                    if to_evict:
                        model.evict_tokens(to_evict)
        
        print(f"✓ Forward pass completed: {len(test_sequence)} tokens processed")
        
        from evaluate import evaluate_sequence_accuracy, evaluate_memory_usage
        acc_results = evaluate_sequence_accuracy(model, test_sequence, vocab, manager, True)
        mem_results = evaluate_memory_usage(model)
        
        print(f"✓ Evaluation completed: {acc_results['accuracy']:.3f} accuracy")
        print(f"✓ Memory tracking: {mem_results['total_alive']} alive tokens")
        
        os.makedirs(".research/iteration1/images", exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot([1, 2, 3], [0.5, 0.7, 0.6], 'o-')
        ax.set_title('Test Plot')
        plt.savefig('.research/iteration1/images/test_plot.pdf', format='pdf', dpi=150)
        plt.close()
        print("✓ PDF plot generation working")
        
        print("\n🎉 All functional tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Functional test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("QUICK FUNCTIONAL TEST")
    print("=" * 60)
    
    test_passed = quick_functional_test()
    
    if test_passed:
        print("\n" + "=" * 60)
        print("RUNNING FULL EXPERIMENT")
        print("=" * 60)
        
        results = main()
        
        print("\n✅ Full experiment completed successfully!")
    else:
        print("\n❌ Functional test failed - aborting full experiment")
        sys.exit(1)
