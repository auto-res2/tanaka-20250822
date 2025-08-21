#!/usr/bin/env python3
"""
Evaluation module for GaP-Tent experiments.
Handles model evaluation with various metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

def evaluate_model(model: nn.Module, data_loader: DataLoader, 
                  device: str = 'cuda', compute_calibration: bool = True) -> Dict[str, float]:
    """
    Comprehensive model evaluation with accuracy, loss, and calibration metrics.
    
    Args:
        model: Neural network model to evaluate
        data_loader: Evaluation data loader
        device: Device to run evaluation on
        compute_calibration: Whether to compute calibration metrics (ECE, Brier, NLL)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.to(device)
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_probs = []
    all_targets = []
    all_predictions = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            if compute_calibration:
                all_probs.append(probs.cpu())
                all_targets.append(target.cpu())
                all_predictions.append(pred.cpu())
    
    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_samples': total_samples
    }
    
    if compute_calibration and all_probs:
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        ece = compute_ece(all_probs, all_targets)
        metrics['ece'] = ece
        
        brier = compute_brier_score(all_probs, all_targets)
        metrics['brier'] = brier
        
        nll = compute_nll(all_probs, all_targets)
        metrics['nll'] = nll
        
        confidences = all_probs.max(dim=1)[0]
        metrics['mean_confidence'] = confidences.mean().item()
        metrics['confidence_std'] = confidences.std().item()
    
    return metrics

def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities [N, C]
        targets: True labels [N]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets).float()
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item() if hasattr(ece, 'item') else float(ece)

def compute_brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Brier Score.
    
    Args:
        probs: Predicted probabilities [N, C]
        targets: True labels [N]
        
    Returns:
        Brier score
    """
    one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
    brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))
    return brier.item()

def compute_nll(probs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute Negative Log-Likelihood.
    
    Args:
        probs: Predicted probabilities [N, C]
        targets: True labels [N]
        
    Returns:
        NLL score
    """
    eps = 1e-12
    probs_clamped = torch.clamp(probs, eps, 1.0)
    nll = -torch.log(probs_clamped[torch.arange(probs.size(0)), targets]).mean()
    return nll.item()

def evaluate_corruption_robustness(model: nn.Module, corruption_loaders: Dict[str, DataLoader],
                                 device: str = 'cuda') -> Dict[str, Dict[str, float]]:
    """
    Evaluate model robustness across different corruption types.
    
    Args:
        model: Model to evaluate
        corruption_loaders: Dictionary mapping corruption names to data loaders
        device: Device to run evaluation on
        
    Returns:
        Nested dictionary with corruption results
    """
    results = {}
    
    for corruption_name, loader in corruption_loaders.items():
        print(f"Evaluating on {corruption_name}...")
        metrics = evaluate_model(model, loader, device)
        results[corruption_name] = metrics
    
    if len(results) > 1:
        mean_accuracy = np.mean([r['accuracy'] for r in results.values()])
        results['mean_accuracy'] = mean_accuracy
        
        results['mCE'] = 100.0 - mean_accuracy  # Simplified
    
    return results

def evaluate_streaming_adaptation(model: nn.Module, stream_loader: DataLoader,
                                adaptation_fn, device: str = 'cuda') -> Dict[str, List[float]]:
    """
    Evaluate model performance during streaming adaptation.
    
    Args:
        model: Model to adapt and evaluate
        stream_loader: Streaming data loader
        adaptation_fn: Function to perform adaptation step
        device: Device to run on
        
    Returns:
        Dictionary with streaming metrics
    """
    model.to(device)
    
    streaming_metrics = {
        'batch_accuracies': [],
        'batch_losses': [],
        'batch_entropies': [],
        'cumulative_accuracy': []
    }
    
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(stream_loader, desc='Streaming Evaluation')):
        data, target = data.to(device), target.to(device)
        
        if adaptation_fn:
            adaptation_fn(model, data)
        
        with torch.no_grad():
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            batch_correct = pred.eq(target).sum().item()
            batch_size = target.size(0)
            
            batch_accuracy = 100.0 * batch_correct / batch_size
            batch_entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).mean().item()
            
            total_correct += batch_correct
            total_samples += batch_size
            cumulative_accuracy = 100.0 * total_correct / total_samples
            
            streaming_metrics['batch_accuracies'].append(batch_accuracy)
            streaming_metrics['batch_losses'].append(loss.item())
            streaming_metrics['batch_entropies'].append(batch_entropy)
            streaming_metrics['cumulative_accuracy'].append(cumulative_accuracy)
    
    return streaming_metrics

def compute_memory_usage() -> Dict[str, float]:
    """
    Compute current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'max_allocated_mb': max_allocated
        }
    else:
        return {'allocated_mb': 0, 'reserved_mb': 0, 'max_allocated_mb': 0}

def print_evaluation_summary(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Print formatted evaluation summary.
    
    Args:
        metrics: Dictionary of evaluation metrics
        title: Title for the summary
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'accuracy' in key.lower():
                print(f"{key:.<30} {value:.2f}%")
            elif key.lower() in ['ece', 'brier', 'nll']:
                print(f"{key:.<30} {value:.4f}")
            else:
                print(f"{key:.<30} {value:.4f}")
        else:
            print(f"{key:.<30} {value}")
    
    print(f"{'='*50}\n")
