#!/usr/bin/env python3
"""
Data preprocessing module for GaP-Tent experiments.
Handles dataset loading, transformations, and corruption generation.
"""

import os
import glob
import random
from typing import List, Tuple, Dict, Optional, Callable
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF

class ImageNetCDataset(Dataset):
    """
    ImageNet-C dataset loader for corruption robustness evaluation.
    """
    def __init__(self, root: str, corruption: str, severity: int, 
                 val_labels_root: str, transform: Optional[Callable] = None):
        """
        Args:
            root: Root directory containing ImageNet-C data
            corruption: Corruption type (e.g., 'gaussian_noise', 'motion_blur')
            severity: Corruption severity level (1-5)
            val_labels_root: Directory containing validation labels mapping
            transform: Optional transform to apply to images
        """
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        
        corruption_dir = os.path.join(root, corruption, str(severity))
        self.image_paths = sorted(glob.glob(os.path.join(corruption_dir, "*.JPEG")))
        
        mapping_file = os.path.join(val_labels_root, 'val_mapping.txt')
        self.filename_to_label = {}
        
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename, label = parts[0], int(parts[1])
                        self.filename_to_label[filename] = label
        else:
            print(f"Warning: Label mapping file not found at {mapping_file}")
            for i, path in enumerate(self.image_paths):
                filename = os.path.basename(path)
                self.filename_to_label[filename] = i % 1000  # Assume 1000 classes
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.filename_to_label.get(filename, 0)
        
        return image, label

class SyntheticCorruptionDataset(Dataset):
    """
    Synthetic dataset with controllable corruptions for testing.
    """
    def __init__(self, num_samples: int = 1000, num_classes: int = 10, 
                 img_size: int = 224, corruption_type: str = 'gaussian_noise',
                 corruption_severity: float = 0.1):
        """
        Args:
            num_samples: Number of samples to generate
            num_classes: Number of classes
            img_size: Image size (square)
            corruption_type: Type of corruption to apply
            corruption_severity: Severity of corruption (0.0 to 1.0)
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.img_size = img_size
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        
        self.class_patterns = self._generate_class_patterns()
    
    def _generate_class_patterns(self):
        """Generate distinctive patterns for each class."""
        patterns = []
        for class_idx in range(self.num_classes):
            pattern = torch.zeros(3, self.img_size, self.img_size)
            
            if class_idx % 4 == 0:  # Horizontal stripes
                for i in range(0, self.img_size, 20):
                    pattern[:, i:i+10, :] = 0.8
            elif class_idx % 4 == 1:  # Vertical stripes
                for i in range(0, self.img_size, 20):
                    pattern[:, :, i:i+10] = 0.8
            elif class_idx % 4 == 2:  # Checkerboard
                for i in range(0, self.img_size, 20):
                    for j in range(0, self.img_size, 20):
                        if (i // 20 + j // 20) % 2 == 0:
                            pattern[:, i:i+20, j:j+20] = 0.8
            else:  # Solid color with noise
                pattern.fill_(0.5 + 0.3 * (class_idx / self.num_classes))
            
            patterns.append(pattern)
        
        return patterns
    
    def _apply_corruption(self, image: torch.Tensor) -> torch.Tensor:
        """Apply specified corruption to image."""
        if self.corruption_type == 'gaussian_noise':
            noise = torch.randn_like(image) * self.corruption_severity
            return torch.clamp(image + noise, 0, 1)
        
        elif self.corruption_type == 'motion_blur':
            kernel_size = max(3, int(self.corruption_severity * 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = torch.ones(1, 1, kernel_size, 1) / kernel_size
            blurred = F.conv2d(image.unsqueeze(0), kernel.repeat(3, 1, 1, 1), 
                             padding=(kernel_size//2, 0), groups=3)
            return torch.clamp(blurred.squeeze(0), 0, 1)
        
        elif self.corruption_type == 'brightness':
            factor = 1.0 + self.corruption_severity * (2.0 * random.random() - 1.0)
            return torch.clamp(image * factor, 0, 1)
        
        elif self.corruption_type == 'contrast':
            mean = image.mean()
            factor = 1.0 + self.corruption_severity * (2.0 * random.random() - 1.0)
            return torch.clamp((image - mean) * factor + mean, 0, 1)
        
        else:
            return image
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        class_idx = idx % self.num_classes
        base_pattern = self.class_patterns[class_idx].clone()
        
        variation = torch.randn_like(base_pattern) * 0.1
        image = torch.clamp(base_pattern + variation, 0, 1)
        
        image = self._apply_corruption(image)
        
        return image, class_idx

class ChannelSkewDataset(Dataset):
    """
    Dataset with controllable per-channel distribution shifts.
    """
    def __init__(self, base_dataset: Dataset, channel_skews: List[float]):
        """
        Args:
            base_dataset: Base dataset to apply skews to
            channel_skews: List of skew factors for each channel [R, G, B]
        """
        self.base_dataset = base_dataset
        self.channel_skews = torch.tensor(channel_skews)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            for c in range(min(3, image.size(0))):
                if c < len(self.channel_skews):
                    image[c] = torch.clamp(image[c] * self.channel_skews[c], 0, 1)
        
        return image, label

def get_imagenet_transforms(train: bool = False, img_size: int = 224) -> transforms.Compose:
    """
    Get standard ImageNet preprocessing transforms.
    
    Args:
        train: Whether to use training transforms (with augmentation)
        img_size: Target image size
        
    Returns:
        Composed transforms
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def create_streaming_loader(dataset: Dataset, batch_sizes: List[int], 
                          shuffle_between_batches: bool = False) -> DataLoader:
    """
    Create a data loader that varies batch sizes during streaming.
    
    Args:
        dataset: Dataset to load from
        batch_sizes: List of batch sizes to cycle through
        shuffle_between_batches: Whether to shuffle data between batch size changes
        
    Returns:
        Custom streaming data loader
    """
    batch_size = batch_sizes[0] if batch_sizes else 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_between_batches)

def create_homogeneous_stream(dataset: Dataset, target_class: int, 
                            stream_length: int = 1000) -> Dataset:
    """
    Create a homogeneous stream containing only samples from one class.
    
    Args:
        dataset: Source dataset
        target_class: Class to create homogeneous stream for
        stream_length: Length of the homogeneous stream
        
    Returns:
        Filtered dataset containing only target class
    """
    class HomogeneousDataset(Dataset):
        def __init__(self, base_dataset, target_class, max_length):
            self.base_dataset = base_dataset
            self.target_class = target_class
            self.max_length = max_length
            
            self.target_indices = []
            for i in range(len(base_dataset)):
                try:
                    _, label = base_dataset[i]
                    if label == target_class:
                        self.target_indices.append(i)
                        if len(self.target_indices) >= max_length:
                            break
                except:
                    continue
        
        def __len__(self):
            return min(len(self.target_indices), self.max_length)
        
        def __getitem__(self, idx):
            actual_idx = self.target_indices[idx % len(self.target_indices)]
            return self.base_dataset[actual_idx]
    
    return HomogeneousDataset(dataset, target_class, stream_length)

def compute_dataset_statistics(dataset: Dataset, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
    """
    Compute mean and std statistics for a dataset.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to use for statistics
        
    Returns:
        Dictionary containing mean and std tensors
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for batch_idx, (data, _) in enumerate(loader):
        if isinstance(data, torch.Tensor) and data.dim() == 4:  # [B, C, H, W]
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]
            
            mean += data.mean(dim=[0, 2]) * batch_samples
            std += data.std(dim=[0, 2]) * batch_samples
            total_samples += batch_samples
            
            if total_samples >= num_samples:
                break
    
    if total_samples > 0:
        mean /= total_samples
        std /= total_samples
    
    return {'mean': mean, 'std': std, 'num_samples': total_samples}

def create_validation_split(dataset: Dataset, val_ratio: float = 0.2, 
                          seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: Dataset to split
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    dataset_size = len(dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    return train_dataset, val_dataset

def save_dataset_samples(dataset: Dataset, save_dir: str, num_samples: int = 10):
    """
    Save sample images from dataset for visualization.
    
    Args:
        dataset: Dataset to sample from
        save_dir: Directory to save samples
        num_samples: Number of samples to save
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        try:
            image, label = dataset[i]
            
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:  # [C, H, W]
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
                    
                    image = TF.to_pil_image(image)
            
            filename = f"sample_{i:03d}_class_{label}.png"
            image.save(os.path.join(save_dir, filename))
            
        except Exception as e:
            print(f"Error saving sample {i}: {e}")
    
    print(f"Saved {num_samples} dataset samples to {save_dir}")
