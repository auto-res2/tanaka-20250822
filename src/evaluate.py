"""
qSR-DPO Evaluation: Comprehensive evaluation metrics and visualization
Evaluates model performance, stability, and preference alignment with detailed plots.
"""

import os
import json
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

try:
    from sklearn.metrics import confusion_matrix, roc_auc_score
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available, using fallback implementations")

from train import (
    TrainConfig, get_model_and_tokenizer, PreferenceDataset, 
    collate_preference_batch, compute_token_logprobs, length_normalize_logprobs
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    win_rate: float
    length_ratio: float
    verbosity_score: float
    stability_score: float
    preference_alignment: float
    loss_variance: float
    ratio_p99: float
    
    def to_dict(self):
        return asdict(self)


class ModelEvaluator:
    """Comprehensive model evaluator for qSR-DPO experiments."""
    
    def __init__(self, model, tokenizer, config: TrainConfig, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.model.eval()
    
    def evaluate_preference_alignment(self, dataset_path: str, batch_size: int = 4) -> Dict[str, Any]:
        """Evaluate how well the model aligns with preference labels."""
        dataset = PreferenceDataset(dataset_path)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: collate_preference_batch(batch, self.tokenizer, self.config.max_length, self.device)
        )
        
        correct_preferences = 0
        total_pairs = 0
        chosen_scores = []
        rejected_scores = []
        length_ratios = []
        
        with torch.no_grad():
            for batch in dataloader:
                chosen_outputs = self.model(batch["chosen"]["input_ids"], batch["chosen"]["attention_mask"])
                chosen_logprobs = compute_token_logprobs(chosen_outputs.logits, batch["chosen"]["labels"])
                chosen_response_mask = batch["chosen"]["response_mask"][:, 1:]
                min_len = min(chosen_logprobs.shape[1], chosen_response_mask.shape[1])
                chosen_logprobs = chosen_logprobs[:, :min_len]
                chosen_response_mask = chosen_response_mask[:, :min_len]
                chosen_seq_scores = length_normalize_logprobs(chosen_logprobs, chosen_response_mask)
                
                rejected_outputs = self.model(batch["rejected"]["input_ids"], batch["rejected"]["attention_mask"])
                rejected_logprobs = compute_token_logprobs(rejected_outputs.logits, batch["rejected"]["labels"])
                rejected_response_mask = batch["rejected"]["response_mask"][:, 1:]
                min_len = min(rejected_logprobs.shape[1], rejected_response_mask.shape[1])
                rejected_logprobs = rejected_logprobs[:, :min_len]
                rejected_response_mask = rejected_response_mask[:, :min_len]
                rejected_seq_scores = length_normalize_logprobs(rejected_logprobs, rejected_response_mask)
                
                correct_preferences += (chosen_seq_scores > rejected_seq_scores).sum().item()
                total_pairs += chosen_seq_scores.shape[0]
                
                chosen_scores.extend(chosen_seq_scores.cpu().tolist())
                rejected_scores.extend(rejected_seq_scores.cpu().tolist())
                
                chosen_lengths = batch["chosen"]["response_mask"].sum(dim=1).float()
                rejected_lengths = batch["rejected"]["response_mask"].sum(dim=1).float()
                batch_length_ratios = chosen_lengths / (rejected_lengths + 1e-8)
                length_ratios.extend(batch_length_ratios.cpu().tolist())
        
        accuracy = correct_preferences / total_pairs if total_pairs > 0 else 0.0
        win_rate = accuracy  # Same as accuracy for preference tasks
        
        avg_length_ratio = np.mean(length_ratios)
        verbosity_score = max(0.0, float(avg_length_ratio - 1.0))
        
        score_diff = np.array(chosen_scores) - np.array(rejected_scores)
        preference_alignment = np.mean(score_diff > 0)
        
        return {
            "accuracy": accuracy,
            "win_rate": win_rate,
            "length_ratio": avg_length_ratio,
            "verbosity_score": verbosity_score,
            "preference_alignment": preference_alignment,
            "chosen_score_mean": np.mean(chosen_scores),
            "rejected_score_mean": np.mean(rejected_scores),
            "score_difference_mean": np.mean(score_diff)
        }
    
    def evaluate_generation_quality(self, prompts: List[str], max_new_tokens: int = 64) -> Dict[str, Any]:
        """Evaluate generation quality and diversity."""
        generations = []
        generation_lengths = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.config.max_length//2)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                generated = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9
                )
                
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                generations.append(response)
                generation_lengths.append(len(response.split()))
        
        unique_generations = len(set(generations))
        diversity_ratio = unique_generations / len(generations) if generations else 0.0
        
        avg_length = np.mean(generation_lengths) if generation_lengths else 0.0
        length_variance = np.var(generation_lengths) if generation_lengths else 0.0
        
        return {
            "diversity_ratio": diversity_ratio,
            "avg_generation_length": avg_length,
            "length_variance": length_variance,
            "total_generations": len(generations),
            "unique_generations": unique_generations
        }
    
    def evaluate_training_stability(self, training_logs: List[Dict]) -> Dict[str, Any]:
        """Evaluate training stability from logs."""
        if not training_logs:
            return {"loss_variance": 0.0, "stability_score": 0.0, "ratio_p99_mean": 0.0}
        
        losses = [log["loss"] for log in training_logs]
        ratio_p99s = [log.get("ratio_p99", 0.0) for log in training_logs]
        
        if len(losses) <= 1:
            return {
                "loss_variance": 0.0,
                "loss_trend": 0.0,
                "stability_score": 1.0,
                "ratio_p99_mean": float(ratio_p99s[0]) if ratio_p99s else 0.0,
                "ratio_p99_max": float(ratio_p99s[0]) if ratio_p99s else 0.0,
                "final_loss": float(losses[0]) if losses else 0.0,
                "num_steps": len(losses)
            }
        
        loss_variance = float(np.var(losses))
        
        try:
            loss_trend = float(np.polyfit(range(len(losses)), losses, 1)[0])
        except np.linalg.LinAlgError:
            loss_trend = 0.0
        
        stability_score = max(0.0, 1.0 - loss_variance - max(0.0, loss_trend))
        
        return {
            "loss_variance": loss_variance,
            "loss_trend": loss_trend,
            "stability_score": float(stability_score),
            "ratio_p99_mean": float(np.mean(ratio_p99s)),
            "ratio_p99_max": float(np.max(ratio_p99s)) if ratio_p99s else 0.0,
            "final_loss": float(losses[-1]) if losses else 0.0,
            "num_steps": len(losses)
        }


def create_evaluation_plots(
    eval_results: Dict[str, Any], 
    training_logs: List[Dict], 
    output_dir: str = ".research/iteration1/images"
):
    """Create comprehensive evaluation plots and save as PDFs."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'pdf.fonttype': 42,  # Ensure fonts are embedded
        'ps.fonttype': 42
    })
    
    if training_logs:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('qSR-DPO Training Dynamics', fontsize=16, fontweight='bold')
        
        steps = [log["step"] for log in training_logs]
        losses = [log["loss"] for log in training_logs]
        epsilons = [log.get("epsilon", 0.0) for log in training_logs]
        betas = [log.get("beta", 1.0) for log in training_logs]
        ratio_p99s = [log.get("ratio_p99", 0.0) for log in training_logs]
        
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2, alpha=0.8)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('DPO Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        ax_eps = axes[0, 1]
        ax_beta = ax_eps.twinx()
        
        line1 = ax_eps.plot(steps, epsilons, 'r-', linewidth=2, label='Epsilon (ε)', alpha=0.8)
        line2 = ax_beta.plot(steps, betas, 'g-', linewidth=2, label='Beta (β)', alpha=0.8)
        
        ax_eps.set_xlabel('Training Step')
        ax_eps.set_ylabel('Epsilon (ε)', color='r')
        ax_beta.set_ylabel('Beta (β)', color='g')
        ax_eps.set_title('Adaptive QA-DPO Parameters')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_eps.legend(lines, labels, loc='upper right')
        ax_eps.grid(True, alpha=0.3)
        
        axes[1, 0].plot(steps, ratio_p99s, 'purple', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('Token-level Ratio P99')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Ratio P99')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(losses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_dynamics.pdf'), 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('qSR-DPO Evaluation Results', fontsize=16, fontweight='bold')
    
    patterns = ['balanced', 'verbose_bias', 'noise_heavy']
    metrics_by_pattern = {}
    
    for pattern in patterns:
        if pattern in eval_results:
            metrics_by_pattern[pattern] = eval_results[pattern]
    
    if metrics_by_pattern:
        accuracies = [metrics_by_pattern[p].get('accuracy', 0.0) for p in patterns if p in metrics_by_pattern]
        pattern_labels = [p for p in patterns if p in metrics_by_pattern]
        
        bars = axes[0, 0].bar(pattern_labels, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Preference Accuracy by Pattern')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1.0)
        axes[0, 0].grid(True, alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        length_ratios = [metrics_by_pattern[p].get('length_ratio', 1.0) for p in pattern_labels]
        bars = axes[0, 1].bar(pattern_labels, length_ratios, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_title('Length Ratio (Chosen/Rejected)')
        axes[0, 1].set_ylabel('Length Ratio')
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Equal Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, ratio in zip(bars, length_ratios):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
        
        verbosity_scores = [metrics_by_pattern[p].get('verbosity_score', 0.0) for p in pattern_labels]
        bars = axes[1, 0].bar(pattern_labels, verbosity_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Verbosity Bias Score')
        axes[1, 0].set_ylabel('Verbosity Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, score in zip(bars, verbosity_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        alignments = [metrics_by_pattern[p].get('preference_alignment', 0.0) for p in pattern_labels]
        bars = axes[1, 1].bar(pattern_labels, alignments, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title('Preference Alignment')
        axes[1, 1].set_ylabel('Alignment Score')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, align in zip(bars, alignments):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{align:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.pdf'), 
               format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'method_comparison' in eval_results:
        methods = list(eval_results['method_comparison'].keys())
        metrics = ['accuracy', 'stability_score', 'verbosity_score']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [eval_results['method_comparison'][method].get(metric, 0.0) for method in methods]
            ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Method Comparison: qSR-DPO Variants')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.pdf'), 
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Evaluation plots saved to {output_dir}/")


def run_comprehensive_evaluation(
    model_path: str,
    test_data_paths: Dict[str, str],
    config: TrainConfig,
    output_dir: str = ".research/iteration1/images"
) -> Dict[str, Any]:
    """Run comprehensive evaluation of trained qSR-DPO model."""
    print("Starting comprehensive evaluation...")
    
    model, tokenizer = get_model_and_tokenizer(config)
    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    training_logs = checkpoint.get("training_logs", [])
    
    print(f"Model loaded from {model_path}")
    
    evaluator = ModelEvaluator(model, tokenizer, config, config.device)
    
    eval_results = {}
    
    for pattern_name, data_path in test_data_paths.items():
        if os.path.exists(data_path):
            print(f"Evaluating on {pattern_name} dataset...")
            pattern_results = evaluator.evaluate_preference_alignment(data_path)
            eval_results[pattern_name] = pattern_results
            
            print(f"{pattern_name} Results:")
            for metric, value in pattern_results.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"Warning: {data_path} not found, skipping {pattern_name} evaluation")
    
    test_prompts = [
        "Explain photosynthesis.",
        "What is the capital of France?",
        "Give two benefits of exercise.",
        "Summarize the water cycle.",
        "What is machine learning?"
    ]
    
    print("Evaluating generation quality...")
    generation_results = evaluator.evaluate_generation_quality(test_prompts)
    eval_results["generation"] = generation_results
    
    print("Generation Results:")
    for metric, value in generation_results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("Evaluating training stability...")
    stability_results = evaluator.evaluate_training_stability(training_logs)
    eval_results["stability"] = stability_results
    
    print("Stability Results:")
    for metric, value in stability_results.items():
        print(f"  {metric}: {value:.4f}")
    
    print("Creating evaluation plots...")
    create_evaluation_plots(eval_results, training_logs, output_dir)
    
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Detailed results saved to {results_path}")
    print("Comprehensive evaluation completed!")
    
    return eval_results


if __name__ == "__main__":
    config = TrainConfig(
        method="qa_dpo",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    model_path = "models/qa_dpo_model_qa_dpo.pt"
    test_data_paths = {
        "balanced": "data/test_balanced.json",
        "verbose_bias": "data/test_verbose.json", 
        "noise_heavy": "data/test_noisy.json"
    }
    
    if os.path.exists(model_path):
        results = run_comprehensive_evaluation(model_path, test_data_paths, config)
        print("Evaluation completed successfully!")
    else:
        print("Model not found. Please run training first.")
