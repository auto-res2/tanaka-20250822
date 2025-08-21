import traceback
import sys
import os

sys.path.insert(0, 'src')

try:
    from main import run_experiment_pipeline
    import argparse
    
    args = argparse.Namespace(
        method='qa_dpo', 
        model_name='sshleifer/tiny-gpt2', 
        epochs=1, 
        learning_rate=1e-4, 
        max_length=256, 
        seed=42
    )
    
    print("Starting debug run...")
    run_experiment_pipeline(args)
    
except Exception as e:
    print('\n=== FULL TRACEBACK ===')
    traceback.print_exc()
    print('\n=== ERROR DETAILS ===')
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
