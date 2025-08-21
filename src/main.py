"""
Main experiment runner for DuoStream-Mem.

This script orchestrates the complete experimental pipeline:
1. Data preprocessing
2. Model setup and calibration
3. Comprehensive evaluation
4. Results analysis and visualization

The script implements all three experimental plans:
- Plan 1: Streaming perplexity under fixed byte budgets
- Plan 2: Long-range factual recall (Needle-in-a-Haystack)
- Plan 3: Efficiency scaling and stability analysis
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from duostream_mem import set_seed
    from preprocess import preprocess_all_tasks
    from train import train_duostream_mem
    from evaluate import evaluate_duostream_mem
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    raise


def setup_experiment_config() -> Dict[str, Any]:
    """Setup default experiment configuration."""
    config = {
        "model": {
            "vocab_size": 1000,
            "d_model": 256,
            "n_layers": 6,
            "n_heads": 8
        },
        
        "budgets_mb": [64, 128, 256],
        
        "data": {
            "perplexity_sequences": 200,
            "needle_sequences": 150,
            "ultra_long_sequences": 20,
            "min_length": 800,
            "max_length": 3000,
            "ultra_long_length": 8000
        },
        
        "evaluation": {
            "calibrate_hyperparams": False,
            "max_eval_sequences": 10,
            "needle_eval_per_position": 5,
            "efficiency_test_sequences": 3
        },
        
        "output": {
            "data_dir": "data/",
            "models_dir": "models/",
            "results_dir": ".research/iteration1/images/",
            "save_intermediate": True
        },
        
        "system": {
            "seed": 1234,
            "device": "auto"  # auto, cpu, cuda
        }
    }
    
    return config


def save_config(config: Dict[str, Any], filepath: str):
    """Save experiment configuration."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load experiment configuration."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def setup_directories(config: Dict[str, Any]):
    """Create necessary directories."""
    dirs_to_create = [
        config["output"]["data_dir"],
        config["output"]["models_dir"],
        config["output"]["results_dir"],
        "config/"
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")


def determine_device(device_config: str) -> str:
    """Determine the best available device."""
    if device_config == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            print("CUDA not available, using CPU")
    else:
        device = device_config
        
    return device


def run_preprocessing(config: Dict[str, Any]) -> str:
    """Run data preprocessing step."""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    data_config = config["data"]
    output_dir = config["output"]["data_dir"]
    
    datasets = preprocess_all_tasks(
        output_dir=output_dir,
        vocab_size=config["model"]["vocab_size"],
        seed=config["system"]["seed"]
    )
    
    data_path = f"{output_dir}/streaming_datasets.json"
    
    print(f"\nPreprocessing completed!")
    print(f"Generated {len(datasets['perplexity']['test'])} perplexity test sequences")
    print(f"Generated {len(datasets['needle']['test'])} needle test sequences")
    print(f"Generated {len(datasets['ultra_long']['test'])} ultra-long test sequences")
    
    return data_path


def run_training(config: Dict[str, Any], data_path: str, device: str):
    """Run model setup and calibration step."""
    print("\n" + "="*60)
    print("STEP 2: MODEL SETUP AND CALIBRATION")
    print("="*60)
    
    train_config = {
        **config["model"],
        "budgets_mb": config["budgets_mb"],
        "calibrate": config["evaluation"]["calibrate_hyperparams"]
    }
    
    config_path = "config/train_config.json"
    with open(config_path, 'w') as f:
        json.dump(train_config, f, indent=2)
    
    trainer = train_duostream_mem(
        config_path=config_path,
        data_path=data_path,
        output_dir=config["output"]["models_dir"],
        device=device
    )
    
    print(f"\nModel setup completed!")
    print(f"Strategies available: {list(trainer.models.keys())}")
    
    return trainer


def run_evaluation(trainer, config: Dict[str, Any], data_path: str):
    """Run comprehensive evaluation."""
    print("\n" + "="*60)
    print("STEP 3: COMPREHENSIVE EVALUATION")
    print("="*60)
    
    evaluator = evaluate_duostream_mem(
        trainer=trainer,
        data_path=data_path,
        output_dir=config["output"]["results_dir"]
    )
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to {config['output']['results_dir']}")
    
    return evaluator


def generate_final_report(config: Dict[str, Any], evaluator, output_path: str):
    """Generate final experiment report."""
    print("\n" + "="*60)
    print("STEP 4: GENERATING FINAL REPORT")
    print("="*60)
    
    report = {
        "experiment_config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_summary": evaluator.results,
        "key_findings": {
            "best_strategy": None,
            "memory_efficiency": None,
            "recall_performance": None,
            "scalability": None
        }
    }
    
    if "streaming_perplexity" in evaluator.results:
        perp_results = evaluator.results["streaming_perplexity"]
        
        best_strategies = {}
        for budget in perp_results["budgets_mb"]:
            if budget in perp_results["perplexities"]:
                best_perp = float('inf')
                best_strategy = None
                for strategy, data in perp_results["perplexities"][budget].items():
                    if data["mean"] < best_perp:
                        best_perp = data["mean"]
                        best_strategy = strategy
                best_strategies[budget] = {"strategy": best_strategy, "perplexity": best_perp}
        
        report["key_findings"]["best_strategy"] = best_strategies
    
    if "needle_in_haystack" in evaluator.results:
        needle_results = evaluator.results["needle_in_haystack"]
        
        recall_summary = {}
        for strategy in needle_results["strategies"]:
            if strategy in needle_results["accuracies"]:
                avg_accuracy = 0.0
                count = 0
                for pos_type in needle_results["position_types"]:
                    if pos_type in needle_results["accuracies"][strategy]:
                        avg_accuracy += needle_results["accuracies"][strategy][pos_type]["accuracy"]
                        count += 1
                if count > 0:
                    recall_summary[strategy] = avg_accuracy / count
        
        report["key_findings"]["recall_performance"] = recall_summary
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Final report saved to {output_path}")
    
    print("\nKEY FINDINGS:")
    if report["key_findings"]["best_strategy"]:
        print("\nBest strategies by budget:")
        for budget, data in report["key_findings"]["best_strategy"].items():
            print(f"  {budget}MB: {data['strategy']} (perplexity: {data['perplexity']:.3f})")
    
    if report["key_findings"]["recall_performance"]:
        print("\nRecall performance:")
        for strategy, accuracy in report["key_findings"]["recall_performance"].items():
            print(f"  {strategy}: {accuracy:.3f}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="DuoStream-Mem Experiment Runner")
    parser.add_argument("--config", type=str, default=None, 
                       help="Path to experiment configuration file")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip data preprocessing step")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip model setup step")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to preprocessed data")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for computation")
    
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        config = setup_experiment_config()
        
    if args.device != "auto":
        config["system"]["device"] = args.device
        
    set_seed(config["system"]["seed"])
    setup_directories(config)
    device = determine_device(config["system"]["device"])
    
    config_path = "config/experiment_config.json"
    save_config(config, config_path)
    
    print(f"\nDuoStream-Mem Experiment Starting...")
    print(f"Device: {device}")
    print(f"Seed: {config['system']['seed']}")
    print(f"Model: {config['model']['n_layers']} layers, {config['model']['n_heads']} heads")
    print(f"Budgets: {config['budgets_mb']} MB")
    
    start_time = time.time()
    
    try:
        if not args.skip_preprocessing:
            data_path = run_preprocessing(config)
        else:
            data_path = args.data_path or f"{config['output']['data_dir']}/streaming_datasets.json"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            print(f"Using existing data: {data_path}")
        
        if not args.skip_training:
            trainer = run_training(config, data_path, device)
        else:
            raise NotImplementedError("Loading pre-trained models not implemented")
        
        evaluator = run_evaluation(trainer, config, data_path)
        
        report_path = f"{config['output']['results_dir']}/final_report.json"
        generate_final_report(config, evaluator, report_path)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Results directory: {config['output']['results_dir']}")
        print(f"Final report: {report_path}")
        
        status_data = {
            "status_enum": "stopped",
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime_seconds": total_time,
            "success": True
        }
        
        with open("experiment_status.json", 'w') as f:
            json.dump(status_data, f, indent=2)
        
        print("\nStatus set to 'stopped'")
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED: {e}")
        
        status_data = {
            "status_enum": "stopped",
            "completion_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success": False,
            "error": str(e)
        }
        
        with open("experiment_status.json", 'w') as f:
            json.dump(status_data, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()
