
import os
import sys
import time
import json
from typing import Dict, Any

def update_status(status: str):
    """Update the experiment status."""
    status_file = ".research/status.json"
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    
    status_data = {
        "status_enum": status,
        "timestamp": time.time(),
        "experiment": "HydraSpec Identity-Preserving Self-Speculative Decoding"
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    print(f"Status updated to: {status}")

def run_preprocessing():
    """Run data preprocessing step."""
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        from preprocess import preprocess_main
        preprocess_main()
        print("✓ Data preprocessing completed successfully")
        return True
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return False

def run_training():
    """Run model training step."""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    try:
        from train import train_main
        train_main()
        print("✓ Model training completed successfully")
        return True
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return False

def run_evaluation():
    """Run model evaluation step."""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)
    
    try:
        from evaluate import evaluate_main
        evaluate_main()
        print("✓ Model evaluation completed successfully")
        return True
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        return False

def check_outputs():
    """Check that all expected outputs were generated."""
    print("\n" + "=" * 60)
    print("STEP 4: OUTPUT VERIFICATION")
    print("=" * 60)
    
    expected_files = [
        "data/train_sequences.npy",
        "data/train_targets.npy",
        "models/hydraspec_model.pt",
        ".research/iteration1/images/training_curves.pdf",
        ".research/iteration1/images/performance_comparison.pdf"
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ Found: {file_path}")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All expected outputs generated successfully")
        return True

def print_experiment_summary():
    """Print a summary of the experiment."""
    print("\n" + "=" * 60)
    print("HYDRASPEC EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print("""
This experiment implemented HydraSpec: Identity-Preserving Self-Speculative Decoding,
a novel approach for accelerating language model inference on single GPUs.

Key Components Implemented:
1. Tiny backbone language model with GRU-based architecture
2. Future token prediction heads for draft generation
3. Skim head for early acceptance verification
4. Int4 quantization for efficient inference
5. Speculative decoding with acceptance/rejection sampling
6. Deterministic RNG ledger for reproducible sampling

Experiment Pipeline:
1. Data Preprocessing: Generated synthetic sequences with patterns
2. Model Training: Trained backbone + future heads with teacher forcing
3. Model Evaluation: Compared sequential vs speculative decoding performance

Key Results:
- Model perplexity on test sequences
- Speedup factor from speculative decoding
- Draft token acceptance rates
- Throughput comparison (tokens/second)

All plots saved as high-quality PDFs in .research/iteration1/images/
    """)

def main():
    """Main experiment orchestration function."""
    print("Starting HydraSpec: Identity-Preserving Self-Speculative Decoding Experiment")
    print(f"Experiment started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    update_status("running")
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs(".research/iteration1/images", exist_ok=True)
    
    success = True
    
    try:
        if not run_preprocessing():
            success = False
            
        if success and not run_training():
            success = False
            
        if success and not run_evaluation():
            success = False
            
        if success and not check_outputs():
            success = False
            
        if success:
            print_experiment_summary()
            print("\n🎉 HydraSpec experiment completed successfully!")
            update_status("stopped")
        else:
            print("\n❌ HydraSpec experiment failed!")
            update_status("failed")
            
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        update_status("interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        update_status("error")
        sys.exit(1)
    
    print(f"Experiment finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
