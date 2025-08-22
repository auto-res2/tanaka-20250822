import math
import time
import random
import gc
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import Vocab, set_seed


@dataclass
class QPack:
    """Quantization pack for simulating precision tiers."""
    q: torch.Tensor      # int8 container (stores packed 2-bit/4-bit levels in int8 slots)
    scale: torch.Tensor  # scale tensor (float)
    bitwidth: int        # 2 or 4


def quantize_per_channel_sym(x: torch.Tensor, bitwidth: int, axis: int = -1) -> QPack:
    """Symmetric per-channel quantization."""
    assert bitwidth in (2, 4)
    levels = 2 ** bitwidth
    qmax = levels // 2 - 1
    qmin = -levels // 2
    scale = x.abs().amax(dim=axis, keepdim=True) / max(qmax, 1)
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp((x / scale).round(), qmin, qmax).to(torch.int8)
    return QPack(q=q, scale=scale.to(x.dtype), bitwidth=bitwidth)


def dequantize(qp: QPack) -> torch.Tensor:
    """Dequantize from QPack."""
    return qp.q.to(qp.scale.dtype) * qp.scale


def add_quant_noise(x: torch.Tensor, tier: str, strength: float = 0.02) -> torch.Tensor:
    """Add distortion to simulate attention/value sensitivity degradation at low precision."""
    if tier == 'fp16':
        return x
    if tier == 'q4':
        noise = strength * 1.0 * torch.randn_like(x)
    elif tier == 'q2':
        noise = strength * 2.5 * torch.randn_like(x)
    else:
        noise = torch.zeros_like(x)
    return x + noise


class EMA:
    """Exponential Moving Average for signal tracking."""
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None

    def update(self, x: torch.Tensor):
        if self.value is None:
            self.value = x.clone()
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value


class SegmentSketchStore:
    """Segment-level KV sketches for very-far context."""
    def __init__(self, d_k: int, d_v: int, proj_rank: int = 16, device: str = 'cpu'):
        self.d_k = d_k
        self.d_v = d_v
        self.proj_rank = proj_rank
        self.device = device
        self.sketches = []  # List of sketch dicts
        self.segment_size = 32

    def add_segment_sketch(self, K: torch.Tensor, V: torch.Tensor, start_idx: int):
        """Add a sketch for a segment of KV pairs."""
        if K.shape[0] == 0:
            return
        
        K_mean = K.mean(dim=0)  # [n_heads, d_k]
        V_mean = V.mean(dim=0)  # [n_heads, d_v]
        
        sketch = {
            'K_sketch': K_mean[:, :self.proj_rank],  # Truncate to proj_rank
            'V_sketch': V_mean[:, :self.proj_rank],
            'start_idx': start_idx,
            'end_idx': start_idx + K.shape[0],
            'coverage': K.shape[0]
        }
        self.sketches.append(sketch)

    def append_to_attention(self, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[int]]]:
        """Append sketches as pseudo-tokens to attention."""
        if len(self.sketches) == 0:
            return K, V, None
        
        sketch_K = []
        sketch_V = []
        seg_ids = []
        
        for i, sketch in enumerate(self.sketches):
            K_pad = torch.zeros(1, K.shape[1], K.shape[2], device=self.device)
            V_pad = torch.zeros(1, V.shape[1], V.shape[2], device=self.device)
            
            K_pad[0, :, :self.proj_rank] = sketch['K_sketch']
            V_pad[0, :, :self.proj_rank] = sketch['V_sketch']
            
            sketch_K.append(K_pad)
            sketch_V.append(V_pad)
            seg_ids.append(f"sketch_{i}")
        
        if sketch_K:
            sketch_K = torch.cat(sketch_K, dim=0)
            sketch_V = torch.cat(sketch_V, dim=0)
            K = torch.cat([K, sketch_K], dim=0)
            V = torch.cat([V, sketch_V], dim=0)
        
        return K, V, seg_ids


class MiMoTAKVManager:
    """MiMoTA-KV cache manager with multi-signal scoring and precision tiering."""
    
    def __init__(self, n_layers: int, n_heads: int, d_k: int, d_v: int, R: int = 16,
                 gamma: float = 0.98, device: str = 'cpu', enable_bandit: bool = False,
                 enable_sketches: bool = False, proj_rank: int = 16):
        self.R = R
        self.gamma = gamma
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.enable_bandit = enable_bandit
        self.enable_sketches = enable_sketches
        self.fp_recent_window = 16
        self.never_demote_anchors_below = 'q4'
        self.step_idx = 0

        self.attn_ema = [EMA(beta=0.9) for _ in range(n_heads)]
        self.q_ema = EMA(beta=0.9)
        torch.manual_seed(0)
        self.P_sem = torch.randn(d_k, 16, device=device) / math.sqrt(d_k)

        self.feature_names = ["attn", "surprisal", "semantic", "anchor", "recency"]
        self.w = torch.tensor([0.35, 0.2, 0.2, 0.15, 0.1], device=device)
        self.temperature = 1.0

        self.E = {('fp16', 'q4'): 0.03, ('q4', 'q2'): 0.08, ('fp16', 'q2'): 0.11}

        self.sketch_store = None
        if enable_sketches:
            self.sketch_store = SegmentSketchStore(d_k, d_v, proj_rank=proj_rank, device=device)

        self._last_attn_rows = None
        self._last_cache_indices = []
        self._last_seg_ids = None
        self._last_surprisal = 3.0
        self._last_anchor_flag = False
        self._last_q = None

    def task_bootstrap(self, task_tag: str):
        """Set priors based on task."""
        if task_tag == 'code':
            self.w = torch.tensor([0.25, 0.2, 0.15, 0.3, 0.1], device=self.device)
            self.never_demote_anchors_below = 'q4'
        elif task_tag == 'summ':
            self.w = torch.tensor([0.3, 0.25, 0.15, 0.2, 0.1], device=self.device)
        elif task_tag == 'rag':
            self.w = torch.tensor([0.4, 0.15, 0.25, 0.1, 0.1], device=self.device)
        elif task_tag == 'needle':
            self.w = torch.tensor([0.45, 0.2, 0.2, 0.1, 0.05], device=self.device)

    def compute_multi_signal_scores(self, cache_indices: List[int], vocab: Vocab) -> torch.Tensor:
        """Compute multi-signal importance scores for tokens."""
        if len(cache_indices) == 0:
            return torch.empty(0, self.n_heads, device=self.device)
        
        scores = torch.zeros(len(cache_indices), self.n_heads, device=self.device)
        
        for i, cache_idx in enumerate(cache_indices):
            if self._last_attn_rows is None or cache_idx >= self._last_attn_rows.shape[1]:
                continue
                
            attn_scores = self._last_attn_rows[:, cache_idx]
            
            surprisal = self._last_surprisal
            
            semantic = 0.5  # Placeholder
            
            token_id = cache_idx  # Simplified
            anchor = 1.0 if self._is_anchor_token(token_id, vocab) else 0.0
            
            recency = math.exp(-0.1 * (len(cache_indices) - i))
            
            for h in range(self.n_heads):
                signal_vec = torch.tensor([
                    attn_scores[h].item(),
                    surprisal,
                    semantic,
                    anchor,
                    recency
                ], device=self.device)
                
                scores[i, h] = torch.dot(self.w, signal_vec)
        
        return scores

    def _is_anchor_token(self, token_id: int, vocab: Vocab) -> bool:
        """Check if token is a structural anchor."""
        if hasattr(vocab, 'stoi'):
            anchor_tokens = [vocab.code_fence, vocab.bracket_l, vocab.bracket_r]
            return any(token_id == vocab.id(tok) for tok in anchor_tokens if tok in vocab.stoi)
        return False

    def waterfill_budgets(self, n_tokens: int) -> Dict[int, int]:
        """Allocate per-head budgets via waterfilling."""
        if n_tokens == 0:
            return {}
        
        budget_per_head = max(1, n_tokens // 4)  # Keep 1/4 of tokens per head
        return {h: budget_per_head for h in range(self.n_heads)}

    def plan_demotion_eviction(self, cache_indices: List[int], vocab: Vocab, memory_pressure: float = 0.5):
        """Plan which tokens to demote/evict based on multi-signal scores."""
        if len(cache_indices) == 0:
            return [], []
        
        scores = self.compute_multi_signal_scores(cache_indices, vocab)
        budgets = self.waterfill_budgets(len(cache_indices))
        
        global_scores = scores.max(dim=1)[0]  # Take max across heads
        sorted_indices = torch.argsort(global_scores, descending=False)
        
        n_to_process = int(memory_pressure * len(cache_indices))
        
        to_demote = []
        to_evict = []
        
        for i in range(min(n_to_process, len(sorted_indices))):
            idx = sorted_indices[i].item()
            cache_idx = cache_indices[idx]
            
            if idx >= len(cache_indices) - self.fp_recent_window:
                continue
                
            if i < n_to_process // 3:
                to_demote.append((cache_idx, 'q4'))
            elif i < 2 * n_to_process // 3:
                to_demote.append((cache_idx, 'q2'))
            else:
                to_evict.append(cache_idx)
        
        return to_demote, to_evict


class ToyCausalLM(nn.Module):
    """Toy causal language model with explicit KV cache."""
    
    def __init__(self, vocab: Vocab, d_model: int = 64, n_heads: int = 4, d_kv: int = 16, device: str = 'cpu'):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = vocab.size()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_kv
        self.d_v = d_kv
        self.device = device

        self.emb = nn.Embedding(self.vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_kv, bias=False)
        self.out_proj = nn.Linear(n_heads * d_kv, self.vocab_size, bias=False)

        self._init_associative_mapping()
        self.to(device)
        self.eval()

        self.cache: List[Dict] = []
        self.segment_size = 32

    def _init_associative_mapping(self):
        """Initialize weights to create strong attention and answer mapping behavior."""
        with torch.no_grad():
            nn.init.normal_(self.emb.weight, mean=0.0, std=0.3)
            nn.init.normal_(self.q_proj.weight, mean=0.0, std=0.2)
            nn.init.normal_(self.k_proj.weight, mean=0.0, std=0.2)
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.2)
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.2)

            nf = self.vocab.n_fact
            subdim = min(self.d_k, max(8, nf))
            
            basis = torch.zeros(nf, self.d_model)
            for i in range(nf):
                basis[i, i % self.d_model] = 3.0
            
            for i in range(nf):
                self.emb.weight[self.vocab.id(self.vocab.fact_tokens[i])] = basis[i]
                self.emb.weight[self.vocab.id(self.vocab.q_tokens[i])] = basis[i] + 0.05 * torch.randn_like(basis[i])
            
            Wq = self.q_proj.weight.view(self.n_heads, self.d_k, self.d_model)
            Wk = self.k_proj.weight.view(self.n_heads, self.d_k, self.d_model)
            for h in range(self.n_heads):
                for i in range(subdim):
                    Wq[h, i, i % self.d_model] = 1.5
                    Wk[h, i, i % self.d_model] = 1.5
            self.q_proj.weight[:] = Wq.view(self.n_heads * self.d_k, self.d_model)
            self.k_proj.weight[:] = Wk.view(self.n_heads * self.d_k, self.d_model)

            Wv = self.v_proj.weight.view(self.n_heads, self.d_v, self.d_model)
            for h in range(self.n_heads):
                for i in range(subdim):
                    Wv[h, i, i % self.d_model] = 1.0
            self.v_proj.weight[:] = Wv.view(self.n_heads * self.d_v, self.d_model)

            Wo = self.out_proj.weight
            Wo.zero_()
            for i in range(nf):
                ans_id = self.vocab.id(self.vocab.ans_tokens[i])
                for h in range(self.n_heads):
                    Wo[ans_id, h * self.d_v + (i % self.d_v)] = 3.0
            
            Wo[self.vocab.id(self.vocab.summ_ans), 0] = 2.5
            Wo[self.vocab.id(self.vocab.code_ans), 1] = 2.5

    def reset_cache(self):
        """Reset the KV cache."""
        self.cache = []

    def append_to_cache(self, token_id: int, K: torch.Tensor, V: torch.Tensor, tier: str = 'fp16'):
        """Append token to KV cache."""
        self.cache.append({
            'token_id': int(token_id),
            'K': K.detach().clone(),
            'V': V.detach().clone(),
            'tier': tier,
            'alive': True,
        })

    def get_cache_tensors(self, add_noise=True) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Get K, V tensors from cache."""
        Ks, Vs, idxs = [], [], []
        for i, item in enumerate(self.cache):
            if not item['alive']:
                continue
            K, V, tier = item['K'], item['V'], item['tier']
            if add_noise:
                K = add_quant_noise(K, tier, strength=0.03)
                V = add_quant_noise(V, tier, strength=0.03)
            Ks.append(K)
            Vs.append(V)
            idxs.append(i)
        
        if len(Ks) == 0:
            return (torch.empty(0, self.n_heads, self.d_k, device=self.device),
                    torch.empty(0, self.n_heads, self.d_v, device=self.device), [])
        
        K = torch.stack(Ks, dim=0)
        V = torch.stack(Vs, dim=0)
        return K, V, idxs

    def demote_tokens(self, token_indices: List[int], new_tier: str):
        """Demote tokens to lower precision tier."""
        for idx in token_indices:
            if 0 <= idx < len(self.cache) and self.cache[idx]['alive']:
                self.cache[idx]['tier'] = new_tier

    def evict_tokens(self, token_indices: List[int]):
        """Evict tokens from cache."""
        for idx in token_indices:
            if 0 <= idx < len(self.cache):
                self.cache[idx]['alive'] = False

    def local_window(self, W: int):
        """Keep only last W alive tokens."""
        alive_idxs = [i for i, it in enumerate(self.cache) if it['alive']]
        if len(alive_idxs) <= W:
            return
        to_evict = alive_idxs[:-W]
        self.evict_tokens(to_evict)

    @torch.no_grad()
    def step(self, token_id: int, manager=None, enable_sketches: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single forward step with KV cache management."""
        x = self.emb(torch.tensor([token_id], device=self.device))
        q = self.q_proj(x).view(1, self.n_heads, self.d_k)
        
        K, V, cache_indices = self.get_cache_tensors(add_noise=True)
        seg_ids = None
        if enable_sketches and (manager is not None) and (manager.enable_sketches):
            K, V, seg_ids = manager.sketch_store.append_to_attention(K, V)
        
        if K.shape[0] == 0:
            attn_weights = torch.zeros(self.n_heads, 0, device=self.device)
            context = torch.zeros(1, self.n_heads * self.d_v, device=self.device)
        else:
            logits = torch.einsum('bhd,Thd->hT', q, K) / math.sqrt(self.d_k)
            attn_weights = F.softmax(logits, dim=-1)
            ctx_per_head = torch.einsum('hT,Thd->hd', attn_weights, V)
            context = ctx_per_head.view(1, self.n_heads * self.d_v)
        
        logits_out = self.out_proj(context).squeeze(0)
        
        if manager is not None:
            manager._last_attn_rows = attn_weights.detach().clone()
            manager._last_cache_indices = cache_indices
            manager._last_seg_ids = seg_ids
        
        k_cur = self.k_proj(x).view(self.n_heads, self.d_k)
        v_cur = self.v_proj(x).view(self.n_heads, self.d_v)
        self.append_to_cache(token_id, k_cur, v_cur, tier='fp16')
        
        return logits_out, attn_weights


def train_model(vocab: Vocab, workloads: Dict, device: str = 'cpu') -> ToyCausalLM:
    """Train/initialize the toy causal LM."""
    print("Initializing toy causal language model...")
    
    model = ToyCausalLM(vocab, d_model=64, n_heads=4, d_kv=16, device=device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {vocab.size()}")
    
    return model


if __name__ == "__main__":
    from preprocess import preprocess_data
    
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    vocab, workloads = preprocess_data()
    model = train_model(vocab, workloads, device)
    
    print("Training completed successfully!")
