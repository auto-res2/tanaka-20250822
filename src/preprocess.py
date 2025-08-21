
import os
import numpy as np
import torch
from typing import List, Tuple

def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def generate_synthetic_sequences(vocab_size: int = 256, 
                                seq_length: int = 64,
                                num_sequences: int = 1000,
                                seed: int = 42) -> Tuple[List[List[int]], List[List[int]]]:
    """Generate synthetic training sequences for the toy language model.
    
    Creates sequences with some structure to make the prediction task non-trivial.
    Returns (input_sequences, target_sequences) where targets are shifted by 1.
    """
    seed_everything(seed)
    
    sequences = []
    targets = []
    
    for _ in range(num_sequences):
        seq = []
        for i in range(seq_length + 1):  # +1 for target
            if i == 0:
                token = np.random.randint(0, vocab_size)
            elif i < 10:
                token = (seq[-1] + np.random.randint(1, 5)) % vocab_size
            else:
                if np.random.random() < 0.7:
                    token = (seq[-1] + np.random.randint(1, 3)) % vocab_size
                else:
                    token = np.random.randint(0, vocab_size)
            seq.append(token)
        
        input_seq = seq[:-1]
        target_seq = seq[1:]
        
        sequences.append(input_seq)
        targets.append(target_seq)
    
    return sequences, targets

def save_data(sequences: List[List[int]], 
              targets: List[List[int]], 
              data_dir: str = "data") -> None:
    """Save preprocessed data to files."""
    os.makedirs(data_dir, exist_ok=True)
    
    seq_array = np.array(sequences)
    target_array = np.array(targets)
    
    np.save(os.path.join(data_dir, "train_sequences.npy"), seq_array)
    np.save(os.path.join(data_dir, "train_targets.npy"), target_array)
    
    print(f"Saved {len(sequences)} sequences to {data_dir}/")
    print(f"Sequence shape: {seq_array.shape}")
    print(f"Target shape: {target_array.shape}")

def load_data(data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed data from files."""
    sequences = np.load(os.path.join(data_dir, "train_sequences.npy"))
    targets = np.load(os.path.join(data_dir, "train_targets.npy"))
    return sequences, targets

def preprocess_main():
    """Main preprocessing function."""
    print("Starting data preprocessing for HydraSpec experiment...")
    
    sequences, targets = generate_synthetic_sequences(
        vocab_size=256,
        seq_length=64,
        num_sequences=1000,
        seed=42
    )
    
    save_data(sequences, targets)
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_main()
