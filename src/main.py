"""
qSR-DPO Main Experiment Script
Orchestrates the complete experimental pipeline: preprocessing, training, and evaluation.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any

import torch
import numpy as np

from preprocess import preprocess_data
from train import TrainConfig, train_qa_dpo
from evaluate import run_comprehensive_evaluation


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu_memory():
    """Check available GPU memory and provide recommendations."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA devices available: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  Device {i}: {props.name}")
            print(f"    Total memory: {memory_gb:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
            
            if memory_gb < 16:
                print(f"    Warning: Limited memory ({memory_gb:.1f} GB < 16 GB T4 target)")
        
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"  Current allocation: {allocated:.2f} GB")
        print(f"  Current cache: {cached:.2f} GB")
        
        return True
    else:
        print("CUDA not available, using CPU")
        return False


def create_experiment_config(args) -> TrainConfig:
    """Create experiment configuration based on arguments and hardware."""
    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 8:
            batch_size = 1
            steps_per_epoch = 25
        elif gpu_memory < 16:
            batch_size = 2
            steps_per_epoch = 50
        else:
            batch_size = 4
            steps_per_epoch = 100
    else:
        batch_size = 1
        steps_per_epoch = 20
    
    config = TrainConfig(
        method=args.method,
        model_name=args.model_name,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=device,
        output_dir="models",
        log_dir=".logs"
    )
    
    print(f"Experiment Configuration:")
    print(f"  Method: {config.method}")
    print(f"  Model: {config.model_name}")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Steps per epoch: {config.steps_per_epoch}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Max length: {config.max_length}")
    
    return config


def run_experiment_pipeline(args) -> Dict[str, Any]:
    """Run the complete qSR-DPO experimental pipeline."""
    print("="*60)
    print("qSR-DPO: Quantization-aware, Self-Rewarded DPO Experiment")
    print("="*60)
    
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    print("\nHardware Check:")
    has_gpu = check_gpu_memory()
    
    print("\nConfiguration:")
    config = create_experiment_config(args)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(".logs", exist_ok=True)
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    results = {}
    
    print("\n" + "="*40)
    print("STEP 1: DATA PREPROCESSING")
    print("="*40)
    
    try:
        preprocessing_stats = preprocess_data(output_dir="data", seed=args.seed)
        results["preprocessing"] = preprocessing_stats
        print("✓ Data preprocessing completed successfully")
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return {"error": f"Preprocessing failed: {e}"}
    
    print("\n" + "="*40)
    print("STEP 2: MODEL TRAINING")
    print("="*40)
    
    train_data_path = "data/train_preferences.json"
    val_data_path = "data/val_preferences.json"
    
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        error_msg = "Training or validation data not found after preprocessing"
        print(f"✗ {error_msg}")
        return {"error": error_msg}
    
    try:
        print(f"Training {config.method} model...")
        start_time = time.time()
        
        model, training_logs = train_qa_dpo(config, train_data_path, val_data_path)
        
        training_time = time.time() - start_time
        results["training"] = {
            "method": config.method,
            "training_time": training_time,
            "total_steps": len(training_logs),
            "final_loss": training_logs[-1]["loss"] if training_logs else None
        }
        
        print(f"✓ Model training completed in {training_time:.1f} seconds")
        print(f"  Total training steps: {len(training_logs)}")
        if training_logs:
            print(f"  Final loss: {training_logs[-1]['loss']:.4f}")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return {"error": f"Training failed: {e}"}
    
    print("\n" + "="*40)
    print("STEP 3: MODEL EVALUATION")
    print("="*40)
    
    model_path = f"models/qa_dpo_model_{config.method}.pt"
    test_data_paths = {
        "balanced": "data/test_balanced.json",
        "verbose_bias": "data/test_verbose.json",
        "noise_heavy": "data/test_noisy.json"
    }
    
    if not os.path.exists(model_path):
        error_msg = f"Trained model not found at {model_path}"
        print(f"✗ {error_msg}")
        return {"error": error_msg}
    
    try:
        eval_results = run_comprehensive_evaluation(
            model_path, test_data_paths, config, 
            output_dir=".research/iteration1/images"
        )
        results["evaluation"] = eval_results
        print("✓ Model evaluation completed successfully")
        
        print("\nEvaluation Summary:")
        for pattern, metrics in eval_results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                print(f"  {pattern}: Accuracy={metrics['accuracy']:.3f}, "
                      f"Length Ratio={metrics.get('length_ratio', 0):.2f}")
        
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        return {"error": f"Evaluation failed: {e}"}
    
    print("\n" + "="*40)
    print("STEP 4: RESULTS SUMMARY")
    print("="*40)
    
    results_path = ".logs/experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Complete results saved to: {results_path}")
    
    print("\nExperiment Summary:")
    print(f"  Method: {config.method}")
    print(f"  Training samples: {results['preprocessing']['train_size']}")
    print(f"  Training time: {results['training']['training_time']:.1f}s")
    print(f"  Device used: {config.device}")
    
    if "balanced" in results["evaluation"]:
        balanced_acc = results["evaluation"]["balanced"]["accuracy"]
        print(f"  Balanced test accuracy: {balanced_acc:.3f}")
    
    if "verbose_bias" in results["evaluation"]:
        verbose_acc = results["evaluation"]["verbose_bias"]["accuracy"]
        length_ratio = results["evaluation"]["verbose_bias"]["length_ratio"]
        print(f"  Verbose bias test: Accuracy={verbose_acc:.3f}, Length Ratio={length_ratio:.2f}")
    
    if "noise_heavy" in results["evaluation"]:
        noise_acc = results["evaluation"]["noise_heavy"]["accuracy"]
        print(f"  Noise robustness: {noise_acc:.3f}")
    
    print("\nGenerated Artifacts:")
    print("  📊 Training dynamics plot: .research/iteration1/images/training_dynamics.pdf")
    print("  📈 Evaluation metrics plot: .research/iteration1/images/evaluation_metrics.pdf")
    print("  📋 Detailed results: .research/iteration1/images/evaluation_results.json")
    print("  🔧 Model checkpoint: models/qa_dpo_model_qa_dpo.pt")
    print("  📝 Training logs: .logs/training_logs_qa_dpo.json")
    
    print("\n" + "="*60)
    print("qSR-DPO EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
    print("="*60)
    
    return results


def main():
    """Main entry point for the qSR-DPO experiment."""
    parser = argparse.ArgumentParser(description="qSR-DPO: Quantization-aware, Self-Rewarded DPO")
    
    parser.add_argument("--method", type=str, default="qa_dpo", 
                       choices=["vanilla", "fixed_clip", "ema_fixed", "qa_dpo"],
                       help="DPO method variant to use")
    parser.add_argument("--model_name", type=str, default="sshleifer/tiny-gpt2",
                       help="Base model name or path")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        results = run_experiment_pipeline(args)
        
        if "error" in results:
            print(f"\n❌ Experiment failed: {results['error']}")
            sys.exit(1)
        else:
            print(f"\n✅ Experiment completed successfully!")
            
            status_file = ".logs/status.json"
            with open(status_file, 'w') as f:
                json.dump({"status_enum": "stopped"}, f)
            print(f"Status set to 'stopped' in {status_file}")
            
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
