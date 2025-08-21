"""
Data preprocessing for DuoStream-Mem experiments.

This module handles:
- Synthetic streaming task generation
- Data preparation for perplexity evaluation
- Needle-in-a-haystack test generation
"""

import random
import json
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np

try:
    from .duostream_mem import SyntheticStreamingTask, set_seed
except ImportError:
    from duostream_mem import SyntheticStreamingTask, set_seed


class StreamingDataPreprocessor:
    """Preprocessor for streaming evaluation tasks."""
    
    def __init__(self, vocab_size: int = 1000, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self.task = SyntheticStreamingTask(vocab_size, device=device)
        
    def generate_perplexity_sequences(self, n_sequences: int = 100, min_length: int = 1000, 
                                    max_length: int = 5000) -> List[List[int]]:
        """Generate sequences for perplexity evaluation."""
        sequences = []
        
        for _ in range(n_sequences):
            length = random.randint(min_length, max_length)
            self.task.facts.clear()  # Reset facts for each sequence
            sequence = self.task.generate_stream(length, query_prob=0.05, fact_prob=0.2)
            sequences.append(sequence)
            
        return sequences
        
    def generate_needle_sequences(self, n_sequences: int = 50, base_length: int = 2000,
                                needle_positions: List[str] = ["start", "middle", "end"]) -> List[Dict]:
        """Generate needle-in-a-haystack sequences."""
        sequences = []
        
        for pos_type in needle_positions:
            for _ in range(n_sequences // len(needle_positions)):
                self.task.facts.clear()
                base_seq = self.task.generate_stream(base_length, query_prob=0.0, fact_prob=0.1)
                
                needle_key = random.randint(1, self.vocab_size // 4)
                needle_value = random.randint(self.vocab_size // 2, self.vocab_size - 1)
                needle_fact = [needle_key, 0, needle_value]  # key=value
                
                if pos_type == "start":
                    needle_pos = random.randint(10, 50)
                elif pos_type == "middle":
                    needle_pos = len(base_seq) // 2 + random.randint(-50, 50)
                else:  # end
                    needle_pos = len(base_seq) - random.randint(50, 100)
                    
                sequence = base_seq[:needle_pos] + needle_fact + base_seq[needle_pos:]
                
                query = [needle_key, 0]  # key=?
                sequence.extend(query)
                
                sequences.append({
                    "sequence": sequence,
                    "needle_key": needle_key,
                    "needle_value": needle_value,
                    "needle_position": needle_pos,
                    "position_type": pos_type,
                    "query_position": len(sequence) - 1
                })
                
        return sequences
        
    def generate_ultra_long_sequences(self, n_sequences: int = 10, target_length: int = 10000) -> List[List[int]]:
        """Generate ultra-long sequences for efficiency testing."""
        sequences = []
        
        for _ in range(n_sequences):
            self.task.facts.clear()
            sequence = self.task.generate_stream(target_length, query_prob=0.08, fact_prob=0.25)
            sequences.append(sequence)
            
        return sequences
        
    def create_evaluation_splits(self, sequences: List, train_ratio: float = 0.0, 
                               val_ratio: float = 0.3, test_ratio: float = 0.7) -> Dict[str, List]:
        """Split sequences into train/val/test sets."""
        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        shuffled = sequences.copy()
        random.shuffle(shuffled)
        
        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:]
        }
        
        return splits
        
    def save_preprocessed_data(self, data: Dict, filepath: str):
        """Save preprocessed data to file."""
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                serializable_data[key] = value
            elif isinstance(value, dict):
                serializable_data[key] = {k: v for k, v in value.items()}
            else:
                serializable_data[key] = value
                
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
    def load_preprocessed_data(self, filepath: str) -> Dict:
        """Load preprocessed data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data


def preprocess_all_tasks(output_dir: str = "data/", vocab_size: int = 1000, seed: int = 1234):
    """Preprocess all evaluation tasks and save to files."""
    set_seed(seed)
    
    preprocessor = StreamingDataPreprocessor(vocab_size)
    
    print("Generating perplexity evaluation sequences...")
    perplexity_sequences = preprocessor.generate_perplexity_sequences(
        n_sequences=200, min_length=800, max_length=3000
    )
    perplexity_splits = preprocessor.create_evaluation_splits(perplexity_sequences)
    
    print("Generating needle-in-a-haystack sequences...")
    needle_sequences = preprocessor.generate_needle_sequences(
        n_sequences=150, base_length=1500
    )
    needle_splits = preprocessor.create_evaluation_splits(needle_sequences)
    
    print("Generating ultra-long sequences...")
    ultra_long_sequences = preprocessor.generate_ultra_long_sequences(
        n_sequences=20, target_length=8000
    )
    ultra_long_splits = preprocessor.create_evaluation_splits(ultra_long_sequences)
    
    datasets = {
        "perplexity": perplexity_splits,
        "needle": needle_splits,
        "ultra_long": ultra_long_splits,
        "vocab_size": vocab_size,
        "seed": seed
    }
    
    preprocessor.save_preprocessed_data(datasets, f"{output_dir}/streaming_datasets.json")
    
    print(f"Preprocessed data saved to {output_dir}/streaming_datasets.json")
    print(f"Perplexity sequences: {len(perplexity_sequences)}")
    print(f"Needle sequences: {len(needle_sequences)}")
    print(f"Ultra-long sequences: {len(ultra_long_sequences)}")
    
    return datasets


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    preprocess_all_tasks()
