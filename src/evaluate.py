
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

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

class RNGLedger:
    """Deterministic RNG stream for reproducible sampling."""
    def __init__(self, seed: int, draws_per_step: int = 16):
        self.g = np.random.default_rng(seed)
        self.draws_per_step = draws_per_step
        self.buf: List[float] = []

    def reserve(self) -> float:
        if not self.buf:
            self.buf = list(self.g.random(self.draws_per_step))
        return self.buf.pop(0)

    def recycle_unused(self):
        self.buf = []

def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int4 quantized weights."""
    lo = (packed & 0x0F).to(torch.int8)
    hi = ((packed >> 4) & 0x0F).to(torch.int8)
    out = torch.empty(packed.shape[0], packed.shape[1] * 2, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out

def gather_row_dot(h: torch.Tensor,
                   rows_idx: torch.Tensor,
                   W_packed: torch.Tensor,
                   scale: torch.Tensor,
                   zp: torch.Tensor) -> torch.Tensor:
    """Dequantize selected rows and compute dot with hidden state."""
    device = h.device
    sel = W_packed.index_select(0, rows_idx.to(W_packed.device))
    sel_int4 = unpack_int4(sel).to(torch.int32)
    s = scale.index_select(0, rows_idx.to(scale.device)).unsqueeze(1)
    z = zp.index_select(0, rows_idx.to(zp.device)).unsqueeze(1).to(torch.int32)
    deq = (sel_int4 - z).to(h.dtype) * s
    logits = torch.matmul(deq, h)
    return logits

def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    if T is None or T == 1.0:
        return logits
    return logits / T

def sample_from_logits(logits: torch.Tensor, rng: RNGLedger) -> int:
    """Sample token from logits using RNG ledger."""
    probs = torch.softmax(logits, dim=-1)
    u = rng.reserve()
    cdf = torch.cumsum(probs, dim=-1)
    u_scaled = u * float(cdf[-1])
    idx = int(torch.searchsorted(cdf, torch.tensor(u_scaled, device=cdf.device)).item())
    return idx

def sequential_decode(model, initial_tokens: List[int], max_length: int, 
                     rng: RNGLedger, device: str) -> Tuple[List[int], float]:
    """Standard sequential decoding for baseline comparison."""
    model.eval()
    tokens = initial_tokens.copy()
    
    h = model.backbone.init_state(1, torch.device(device))
    
    for token in initial_tokens:
        h = model.backbone.step(h, torch.tensor([token], device=device))
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(max_length - len(initial_tokens)):
            logits = model.backbone.full_logits(h)
            logits = apply_temperature(logits[0], 1.0)
            
            next_token = sample_from_logits(logits, rng)
            tokens.append(next_token)
            
            h = model.backbone.step(h, torch.tensor([next_token], device=device))
    
    decode_time = time.time() - start_time
    return tokens, decode_time

def speculative_decode_simple(model, initial_tokens: List[int], max_length: int,
                             rng: RNGLedger, device: str, draft_length: int = 2) -> Tuple[List[int], float, Dict]:
    """Simplified speculative decoding with future heads."""
    model.eval()
    tokens = initial_tokens.copy()
    
    h = model.backbone.init_state(1, torch.device(device))
    
    for token in initial_tokens:
        h = model.backbone.step(h, torch.tensor([token], device=device))
    
    stats = {
        'total_steps': 0,
        'accepted_drafts': 0,
        'total_draft_tokens': 0,
        'verification_calls': 0
    }
    
    start_time = time.time()
    
    with torch.no_grad():
        while len(tokens) < max_length:
            stats['total_steps'] += 1
            
            draft_tokens = []
            h_draft = h.clone()
            
            for i in range(min(draft_length, len(model.future_heads))):
                h_proj = model.future_heads[i](h_draft)
                draft_logits = torch.matmul(h_proj, model.backbone.embed.weight.T)
                draft_logits = apply_temperature(draft_logits[0], 1.0)
                
                draft_token = sample_from_logits(draft_logits, rng)
                draft_tokens.append(draft_token)
                
                h_draft = model.backbone.step(h_draft, torch.tensor([draft_token], device=device))
            
            stats['total_draft_tokens'] += len(draft_tokens)
            
            h_verify = h.clone()
            accepted = 0
            
            for i, draft_token in enumerate(draft_tokens):
                stats['verification_calls'] += 1
                
                true_logits = model.backbone.full_logits(h_verify)
                true_logits = apply_temperature(true_logits[0], 1.0)
                true_probs = torch.softmax(true_logits, dim=-1)
                
                if i < len(model.future_heads):
                    h_proj = model.future_heads[i](h)
                    draft_logits = torch.matmul(h_proj, model.backbone.embed.weight.T)
                    draft_logits = apply_temperature(draft_logits[0], 1.0)
                    draft_probs = torch.softmax(draft_logits, dim=-1)
                    
                    accept_prob = min(1.0, float(true_probs[draft_token] / draft_probs[draft_token]))
                    
                    if rng.reserve() < accept_prob:
                        tokens.append(draft_token)
                        h = model.backbone.step(h, torch.tensor([draft_token], device=device))
                        h_verify = h.clone()
                        accepted += 1
                    else:
                        corrected_token = sample_from_logits(true_logits, rng)
                        tokens.append(corrected_token)
                        h = model.backbone.step(h, torch.tensor([corrected_token], device=device))
                        break
                else:
                    true_token = sample_from_logits(true_logits, rng)
                    tokens.append(true_token)
                    h = model.backbone.step(h, torch.tensor([true_token], device=device))
                    break
                
                if len(tokens) >= max_length:
                    break
            
            stats['accepted_drafts'] += accepted
            
            rng.recycle_unused()
    
    decode_time = time.time() - start_time
    return tokens, decode_time, stats

def evaluate_model(model, test_sequences: np.ndarray, device: str) -> Dict[str, float]:
    """Evaluate model perplexity on test sequences."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for seq in tqdm(test_sequences[:100], desc="Evaluating"):  # Use subset for speed
            seq_tensor = torch.from_numpy(seq).long().unsqueeze(0).to(device)
            
            outputs = model(seq_tensor)
            
            targets = seq_tensor[:, 1:]  # Shift by 1
            logits = outputs['backbone_logits'][:, :-1]  # Remove last prediction
            
            loss = F.cross_entropy(
                logits.reshape(-1, model.backbone.vocab_size),
                targets.reshape(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    perplexity = math.exp(total_loss / total_tokens)
    return {'perplexity': perplexity, 'avg_loss': total_loss / total_tokens}

def plot_performance_comparison(results: Dict, save_path: str):
    """Plot performance comparison between sequential and speculative decoding."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    speedups = [r['speedup'] for r in results['comparisons']]
    ax1.hist(speedups, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Speedup Factor')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Speculative Decoding Speedup Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(speedups), color='red', linestyle='--', 
                label=f'Mean: {np.mean(speedups):.2f}x')
    ax1.legend()
    
    accept_rates = [r['acceptance_rate'] for r in results['comparisons']]
    ax2.hist(accept_rates, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Acceptance Rate')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Draft Token Acceptance Rate Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(accept_rates), color='red', linestyle='--',
                label=f'Mean: {np.mean(accept_rates):.3f}')
    ax2.legend()
    
    seq_tps = [r['sequential_tps'] for r in results['comparisons']]
    spec_tps = [r['speculative_tps'] for r in results['comparisons']]
    
    x = np.arange(len(seq_tps))
    width = 0.35
    
    ax3.bar(x - width/2, seq_tps, width, label='Sequential', alpha=0.7, color='orange')
    ax3.bar(x + width/2, spec_tps, width, label='Speculative', alpha=0.7, color='purple')
    ax3.set_xlabel('Test Sequence')
    ax3.set_ylabel('Tokens/Second')
    ax3.set_title('Throughput Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    metrics = ['Mean Speedup', 'Mean Acceptance Rate', 'Avg Sequential TPS', 'Avg Speculative TPS']
    values = [
        np.mean(speedups),
        np.mean(accept_rates),
        np.mean(seq_tps),
        np.mean(spec_tps)
    ]
    
    ax4.bar(metrics, values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Performance Summary')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison saved to {save_path}")

def evaluate_main():
    """Main evaluation function."""
    print("Starting HydraSpec model evaluation...")
    
    seed_everything(42)
    
    if not os.path.exists("models/hydraspec_model.pt"):
        print("Error: Model not found. Please run training first.")
        return
    
    checkpoint = torch.load("models/hydraspec_model.pt", map_location='cpu')
    
    from train import HydraSpecModel
    model = HydraSpecModel(
        vocab_size=checkpoint['vocab_size'],
        hidden_size=checkpoint['hidden_size'],
        num_heads=checkpoint['num_heads']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.backbone.export_quantized_head()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Using device: {device}")
    
    from preprocess import load_data
    train_sequences, _ = load_data()
    test_sequences = train_sequences[-100:]  # Use last 100 as test
    
    print("Evaluating model perplexity...")
    eval_results = evaluate_model(model, test_sequences, device)
    print(f"Test Perplexity: {eval_results['perplexity']:.4f}")
    
    print("Comparing sequential vs speculative decoding...")
    
    comparisons = []
    rng_seq = RNGLedger(42)
    rng_spec = RNGLedger(42)  # Same seed for fair comparison
    
    for i in tqdm(range(10), desc="Running comparisons"):  # Test on 10 sequences
        initial_tokens = test_sequences[i][:10].tolist()  # Use first 10 tokens as prompt
        max_length = 30
        
        seq_tokens, seq_time = sequential_decode(
            model, initial_tokens, max_length, rng_seq, device
        )
        
        rng_spec = RNGLedger(42)
        
        spec_tokens, spec_time, spec_stats = speculative_decode_simple(
            model, initial_tokens, max_length, rng_spec, device
        )
        
        seq_tps = (len(seq_tokens) - len(initial_tokens)) / seq_time
        spec_tps = (len(spec_tokens) - len(initial_tokens)) / spec_time
        speedup = seq_time / spec_time
        acceptance_rate = spec_stats['accepted_drafts'] / max(spec_stats['total_draft_tokens'], 1)
        
        comparisons.append({
            'sequential_time': seq_time,
            'speculative_time': spec_time,
            'sequential_tps': seq_tps,
            'speculative_tps': spec_tps,
            'speedup': speedup,
            'acceptance_rate': acceptance_rate,
            'spec_stats': spec_stats
        })
    
    results = {
        'eval_metrics': eval_results,
        'comparisons': comparisons
    }
    
    avg_speedup = np.mean([c['speedup'] for c in comparisons])
    avg_acceptance = np.mean([c['acceptance_rate'] for c in comparisons])
    
    print(f"\nEvaluation Results:")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Average Acceptance Rate: {avg_acceptance:.3f}")
    print(f"Test Perplexity: {eval_results['perplexity']:.4f}")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    plot_performance_comparison(results, ".research/iteration1/images/performance_comparison.pdf")
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    evaluate_main()
