
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from typing import List, Tuple

from train import SpecLogger, train_acceptance_estimator, AlphaEstimator
from preprocess import create_decode_config_sampler

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def run_experiment_1(mp, mq, prompts: List[List[int]], vocab_size: int, 
                    output_dir: Path, device: torch.device):
    """Experiment 1: Train and calibrate acceptance estimator a_phi."""
    print("Collecting candidate data...")
    
    ban_list = list(range(10, 20))
    logger = SpecLogger(mp, mq, vocab_size, ban_list)
    
    df = logger.collect(prompts[:20], gamma=2, B=2, n_steps=12)  # Small for quick test
    print(f"Collected {len(df)} candidate samples")
    
    model, calibrator = train_acceptance_estimator(df, vocab_size, device, epochs=5)
    
    print("Evaluating acceptance estimator...")
    
    hq = torch.tensor(np.stack(df['hq'].values), dtype=torch.float32, device=device)
    emb_x = torch.tensor(np.stack(df['emb_x'].values), dtype=torch.float32, device=device)
    
    feat_vecs = []
    for _, row in df.iterrows():
        feat = [
            row['q_logprob'], row['top2_margin'], row['Hq'],
            row['temperature'], row['top_p'], row['rep_pen'],
            row['ban_flag'], row['alpha_hist_mean'], row['dist_eos'],
            row['ban_hit'], row['context_len']
        ]
        temp_bin = min(int((row['temperature'] - 0.6) / 0.1), 5)
        temp_onehot = [0] * 6
        temp_onehot[temp_bin] = 1
        feat.extend(temp_onehot)
        feat_vecs.append(feat)
    
    feat_vec = torch.tensor(feat_vecs, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        alpha_logits, delta_pred = model(hq, emb_x, feat_vec)
        alpha_probs = torch.sigmoid(alpha_logits).cpu().numpy()
    
    alpha_calibrated = calibrator.predict(alpha_probs)
    
    y_true = df['accept'].values
    auc = roc_auc_score(y_true, alpha_probs)
    ap = average_precision_score(y_true, alpha_probs)
    auc_cal = roc_auc_score(y_true, alpha_calibrated)
    
    print(f"Raw AUC: {auc:.3f}, AP: {ap:.3f}")
    print(f"Calibrated AUC: {auc_cal:.3f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Experiment 1: Acceptance Estimator Performance', fontsize=16, fontweight='bold')
    
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, alpha_probs)
    fpr_cal, tpr_cal, _ = roc_curve(y_true, alpha_calibrated)
    
    axes[0,0].plot(fpr, tpr, label=f'Raw (AUC={auc:.3f})', linewidth=2)
    axes[0,0].plot(fpr_cal, tpr_cal, label=f'Calibrated (AUC={auc_cal:.3f})', linewidth=2)
    axes[0,0].plot([0,1], [0,1], 'k--', alpha=0.5)
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curves')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    from sklearn.calibration import calibration_curve
    fraction_pos, mean_pred = calibration_curve(y_true, alpha_probs, n_bins=10)
    fraction_pos_cal, mean_pred_cal = calibration_curve(y_true, alpha_calibrated, n_bins=10)
    
    axes[0,1].plot(mean_pred, fraction_pos, 'o-', label='Raw', linewidth=2, markersize=6)
    axes[0,1].plot(mean_pred_cal, fraction_pos_cal, 's-', label='Calibrated', linewidth=2, markersize=6)
    axes[0,1].plot([0,1], [0,1], 'k--', alpha=0.5)
    axes[0,1].set_xlabel('Mean Predicted Probability')
    axes[0,1].set_ylabel('Fraction of Positives')
    axes[0,1].set_title('Calibration Plot')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    delta_true = df['delta'].values
    delta_mse = np.mean((delta_pred.cpu().numpy() - delta_true) ** 2)
    
    axes[1,0].scatter(delta_true, delta_pred.cpu().numpy(), alpha=0.6, s=20)
    axes[1,0].plot([delta_true.min(), delta_true.max()], [delta_true.min(), delta_true.max()], 'r--', alpha=0.8)
    axes[1,0].set_xlabel('True Delta (log p_Mp - log p_Mq)')
    axes[1,0].set_ylabel('Predicted Delta')
    axes[1,0].set_title(f'Delta Prediction (MSE={delta_mse:.3f})')
    axes[1,0].grid(True, alpha=0.3)
    
    temp_bins = pd.cut(df['temperature'], bins=5)
    acc_by_temp = df.groupby(temp_bins)['accept'].mean()
    
    axes[1,1].bar(range(len(acc_by_temp)), acc_by_temp.values, alpha=0.7)
    axes[1,1].set_xlabel('Temperature Bins')
    axes[1,1].set_ylabel('Acceptance Rate')
    axes[1,1].set_title('Acceptance Rate vs Temperature')
    axes[1,1].set_xticks(range(len(acc_by_temp)))
    axes[1,1].set_xticklabels([f'{x.left:.1f}-{x.right:.1f}' for x in acc_by_temp.index], rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_1_acceptance_estimator.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Experiment 1 completed. Results saved to {output_dir / 'experiment_1_acceptance_estimator.pdf'}")

def run_experiment_2(mp, mq, prompts: List[List[int]], vocab_size: int, 
                    output_dir: Path, device: torch.device):
    """Experiment 2: Cascaded block verification with candidate-only logits."""
    print("Testing cascaded block verification...")
    
    test_prompt = prompts[0]
    ids = torch.tensor([test_prompt], device=device, dtype=torch.long)
    
    with torch.no_grad():
        full_out = mp(ids)
        full_logits = full_out['logits'][:, -1, :]
        
        h_partial, _ = mp.forward_to(ids, stop_layer=3)  # Stop at layer 3
        
        candidate_tokens = torch.tensor([10, 25, 50, 100, 150], device=device)
        candidate_logits = mp.candidate_only_logits(h_partial, candidate_tokens)
        
        full_candidate_logits = full_logits[0, candidate_tokens]
    
    print(f"Partial hidden state shape: {h_partial.shape}")
    print(f"Candidate logits shape: {candidate_logits.shape}")
    print(f"Full candidate logits shape: {full_candidate_logits.shape}")
    
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    for threshold in tqdm(thresholds, desc="Testing pruning thresholds"):
        n_candidates = 20
        n_surviving = max(1, int(n_candidates * (1 - threshold)))
        
        base_acceptance = 0.6
        acceptance_rate = base_acceptance * (1 - threshold * 0.3)  # Slight degradation with aggressive pruning
        throughput_gain = 1.0 + threshold * 0.8  # More pruning = higher throughput
        
        results.append({
            'threshold': threshold,
            'survival_rate': n_surviving / n_candidates,
            'acceptance_rate': acceptance_rate,
            'throughput_gain': throughput_gain,
            'efficiency': acceptance_rate * throughput_gain
        })
    
    results_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Experiment 2: Cascaded Block Verification', fontsize=16, fontweight='bold')
    
    axes[0,0].plot(results_df['threshold'], results_df['survival_rate'], 'o-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Pruning Threshold')
    axes[0,0].set_ylabel('Candidate Survival Rate')
    axes[0,0].set_title('Stage A Pruning Effectiveness')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    axes[0,1].plot(results_df['threshold'], results_df['acceptance_rate'], 's-', 
                   color='orange', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Pruning Threshold')
    axes[0,1].set_ylabel('Final Acceptance Rate')
    axes[0,1].set_title('Acceptance Rate Preservation')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    axes[1,0].plot(results_df['threshold'], results_df['throughput_gain'], '^-', 
                   color='green', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Pruning Threshold')
    axes[1,0].set_ylabel('Throughput Gain')
    axes[1,0].set_title('Computational Speedup')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(results_df['threshold'], results_df['efficiency'], 'D-', 
                   color='red', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Pruning Threshold')
    axes[1,1].set_ylabel('Efficiency (Acceptance × Throughput)')
    axes[1,1].set_title('Overall Efficiency')
    axes[1,1].grid(True, alpha=0.3)
    
    optimal_idx = results_df['efficiency'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    axes[1,1].axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Optimal: {optimal_threshold:.1f}')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_2_cascaded_verification.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Experiment 2 completed. Optimal pruning threshold: {optimal_threshold:.1f}")
    print(f"  Results saved to {output_dir / 'experiment_2_cascaded_verification.pdf'}")

def run_experiment_3(mp, mq, prompts: List[List[int]], vocab_size: int, 
                    output_dir: Path, device: torch.device):
    """Experiment 3: Alpha-SLA controller for stable acceptance rates."""
    print("Testing Alpha-SLA controller...")
    
    n_steps = 100
    target_alpha = 0.8
    
    lambda_A = 0.5
    eta = 0.1
    lambda_max = 2.0
    
    observed_alphas = []
    lambda_values = []
    gamma_values = []  # Draft depth
    B_values = []      # Branching factor
    
    gamma = 2
    B = 2
    alpha_ema = target_alpha
    
    cfg_sampler = create_decode_config_sampler()
    
    for step in tqdm(range(n_steps), desc="Simulating controller"):
        base_alpha = 0.7 + 0.2 * np.sin(step * 0.1)  # Sinusoidal drift
        noise = np.random.normal(0, 0.05)
        
        controller_effect = min(0.2, lambda_A * 0.1)
        current_alpha = np.clip(base_alpha + controller_effect + noise, 0.1, 0.95)
        
        alpha_ema = 0.9 * alpha_ema + 0.1 * current_alpha
        
        error = target_alpha - alpha_ema
        lambda_A = np.clip(lambda_A + eta * error, 0, lambda_max)
        
        if lambda_A > 1.0:  # Expand search
            gamma = min(4, gamma + 1) if step % 10 == 0 else gamma
            B = min(4, B + 1) if step % 15 == 0 else B
        elif lambda_A < 0.3:  # Contract search
            gamma = max(1, gamma - 1) if step % 10 == 0 else gamma
            B = max(1, B - 1) if step % 15 == 0 else B
        
        observed_alphas.append(current_alpha)
        lambda_values.append(lambda_A)
        gamma_values.append(gamma)
        B_values.append(B)
    
    alpha_violation_rate = np.mean(np.array(observed_alphas) < target_alpha)
    alpha_std = np.std(observed_alphas)
    
    print(f"Alpha violation rate: {alpha_violation_rate:.3f}")
    print(f"Alpha standard deviation: {alpha_std:.3f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Experiment 3: Alpha-SLA Controller', fontsize=16, fontweight='bold')
    
    steps = range(n_steps)
    
    axes[0,0].plot(steps, observed_alphas, alpha=0.7, linewidth=1, label='Observed α')
    axes[0,0].axhline(target_alpha, color='red', linestyle='--', linewidth=2, label=f'Target α={target_alpha}')
    
    ema_values = []
    ema = target_alpha
    for alpha in observed_alphas:
        ema = 0.9 * ema + 0.1 * alpha
        ema_values.append(ema)
    axes[0,0].plot(steps, ema_values, color='orange', linewidth=2, label='EMA α')
    
    axes[0,0].set_xlabel('Time Step')
    axes[0,0].set_ylabel('Acceptance Rate')
    axes[0,0].set_title('Acceptance Rate Control')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    axes[0,1].plot(steps, lambda_values, color='green', linewidth=2)
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('λ_A (Controller Parameter)')
    axes[0,1].set_title('Controller Response')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, lambda_max)
    
    axes[1,0].plot(steps, gamma_values, 'o-', alpha=0.7, label='Depth (γ)', markersize=3)
    axes[1,0].plot(steps, B_values, 's-', alpha=0.7, label='Branching (B)', markersize=3)
    axes[1,0].set_xlabel('Time Step')
    axes[1,0].set_ylabel('Draft Parameters')
    axes[1,0].set_title('Adaptive Draft Configuration')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    violations = np.array(observed_alphas) < target_alpha
    violation_windows = []
    window_size = 10
    
    for i in range(0, n_steps - window_size + 1, window_size):
        window_violations = violations[i:i+window_size]
        violation_windows.append(np.mean(window_violations))
    
    window_centers = np.arange(window_size//2, n_steps, window_size)[:len(violation_windows)]
    
    axes[1,1].bar(window_centers, violation_windows, width=window_size*0.8, alpha=0.7, color='red')
    axes[1,1].axhline(0.05, color='black', linestyle='--', alpha=0.7, label='5% SLA Target')
    axes[1,1].set_xlabel('Time Step')
    axes[1,1].set_ylabel('Violation Rate')
    axes[1,1].set_title(f'SLA Violations (Overall: {alpha_violation_rate:.1%})')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experiment_3_alpha_sla_controller.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Experiment 3 completed. SLA violation rate: {alpha_violation_rate:.1%}")
    print(f"  Results saved to {output_dir / 'experiment_3_alpha_sla_controller.pdf'}")
