"""
qSR-DPO Data Preprocessing: Synthetic Preference Pair Generation
Creates synthetic preference datasets with different patterns for testing DPO robustness.
"""

import os
import random
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class PairItem:
    """Single preference pair item."""
    prompt: str
    chosen: str
    rejected: str
    weight: float = 1.0
    pattern: str = "balanced"


class SyntheticPreferenceDataset:
    """Create synthetic preference pairs with multiple patterns to test robustness.
    
    Patterns:
      - balanced: labels are consistent without verbosity bias
      - verbose_bias: chosen responses tend to be longer
      - noise_heavy: some label flips (noisy preferences)
    """
    
    def __init__(self, n: int, pattern: str = "balanced", seed: int = 0):
        set_seed(seed)
        self.items: List[PairItem] = []
        self.pattern = pattern
        
        topics = [
            ("Explain photosynthesis.",
             "Photosynthesis is the process by which plants convert light into chemical energy, producing glucose and oxygen.",
             "Plants like light."),
            ("What is the capital of France?",
             "The capital of France is Paris, a major European city known for its culture and history.",
             "France is a country."),
            ("Give two benefits of exercise.",
             "Exercise improves cardiovascular health and enhances mental well-being through endorphin release.",
             "Exercise is sometimes good."),
            ("Summarize the water cycle.",
             "Water evaporates from surfaces, condenses into clouds, and precipitates back to Earth in a continuous cycle.",
             "Rain happens."),
            ("What is machine learning?",
             "Machine learning is a subset of AI that enables computers to learn patterns from data without explicit programming.",
             "Computers can learn things."),
            ("Describe gravity.",
             "Gravity is a fundamental force that attracts objects with mass toward each other, keeping planets in orbit.",
             "Things fall down."),
            ("How do vaccines work?",
             "Vaccines train the immune system to recognize and fight specific pathogens by introducing harmless antigens.",
             "Vaccines help immunity."),
            ("What causes seasons?",
             "Seasons result from Earth's axial tilt as it orbits the Sun, creating varying sunlight exposure throughout the year.",
             "Earth tilts and moves.")
        ]
        
        for i in range(n):
            prompt, good_response, bad_response = random.choice(topics)
            
            if pattern == "balanced":
                chosen, rejected = good_response, bad_response
            elif pattern == "verbose_bias":
                chosen = good_response + " This process involves multiple complex steps and intricate feedback mechanisms that maintain delicate balance in natural ecosystems worldwide."
                rejected = bad_response
            elif pattern == "noise_heavy":
                if random.random() < 0.3:
                    chosen, rejected = bad_response, good_response
                else:
                    chosen, rejected = good_response, bad_response
            else:
                chosen, rejected = good_response, bad_response
                
            self.items.append(PairItem(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                weight=1.0,
                pattern=pattern
            ))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "prompt": item.prompt,
            "chosen": item.chosen,
            "rejected": item.rejected,
            "weight": item.weight,
            "pattern": item.pattern
        }
    
    def save_to_json(self, filepath: str):
        """Save dataset to JSON file."""
        data = [asdict(item) for item in self.items]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, filepath: str):
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset = cls(0)  # Create empty dataset
        dataset.items = [PairItem(**item) for item in data]
        return dataset


def create_mixed_dataset(n_balanced: int = 200, n_verbose: int = 100, n_noisy: int = 50, seed: int = 42) -> SyntheticPreferenceDataset:
    """Create a mixed dataset with different patterns for robust testing."""
    set_seed(seed)
    
    balanced_ds = SyntheticPreferenceDataset(n_balanced, "balanced", seed)
    verbose_ds = SyntheticPreferenceDataset(n_verbose, "verbose_bias", seed + 1)
    noisy_ds = SyntheticPreferenceDataset(n_noisy, "noise_heavy", seed + 2)
    
    mixed_ds = SyntheticPreferenceDataset(0)
    mixed_ds.items = balanced_ds.items + verbose_ds.items + noisy_ds.items
    
    random.shuffle(mixed_ds.items)
    
    return mixed_ds


def preprocess_data(output_dir: str = "data", seed: int = 42):
    """Main preprocessing function to generate synthetic preference datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic preference datasets...")
    
    train_dataset = create_mixed_dataset(n_balanced=300, n_verbose=150, n_noisy=75, seed=seed)
    train_path = os.path.join(output_dir, "train_preferences.json")
    train_dataset.save_to_json(train_path)
    print(f"Training dataset saved: {train_path} ({len(train_dataset)} pairs)")
    
    val_dataset = SyntheticPreferenceDataset(100, "balanced", seed + 100)
    val_path = os.path.join(output_dir, "val_preferences.json")
    val_dataset.save_to_json(val_path)
    print(f"Validation dataset saved: {val_path} ({len(val_dataset)} pairs)")
    
    test_balanced = SyntheticPreferenceDataset(50, "balanced", seed + 200)
    test_balanced.save_to_json(os.path.join(output_dir, "test_balanced.json"))
    
    test_verbose = SyntheticPreferenceDataset(50, "verbose_bias", seed + 300)
    test_verbose.save_to_json(os.path.join(output_dir, "test_verbose.json"))
    
    test_noisy = SyntheticPreferenceDataset(50, "noise_heavy", seed + 400)
    test_noisy.save_to_json(os.path.join(output_dir, "test_noisy.json"))
    
    print("Test datasets created for different patterns")
    
    stats = {
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_sizes": {
            "balanced": len(test_balanced),
            "verbose_bias": len(test_verbose),
            "noise_heavy": len(test_noisy)
        },
        "train_pattern_distribution": {
            "balanced": sum(1 for item in train_dataset.items if item.pattern == "balanced"),
            "verbose_bias": sum(1 for item in train_dataset.items if item.pattern == "verbose_bias"),
            "noise_heavy": sum(1 for item in train_dataset.items if item.pattern == "noise_heavy")
        }
    }
    
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset statistics saved: {stats_path}")
    print("Preprocessing completed successfully!")
    
    return stats


if __name__ == "__main__":
    stats = preprocess_data()
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
