"""
Streaming LLM implementation with DuoStream-Mem integration.

This module provides a simplified streaming language model that demonstrates
the DuoStream-Mem method in action, including:
- StreamingLLM baseline with start+recent cache
- DuoStream-Mem with prototype banks and DuoAttention
- Comparison baselines (SnapKV, KIVI-style quantization)
"""

import math
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .duostream_mem import (
        PrototypeBank, DuoAttentionRouter, ByteBudgetManager, ModelCfg,
        EvictionObserver, precompute_rope_frequencies, apply_rope, inverse_rope
    )
except ImportError:
    from duostream_mem import (
        PrototypeBank, DuoAttentionRouter, ByteBudgetManager, ModelCfg,
        EvictionObserver, precompute_rope_frequencies, apply_rope, inverse_rope
    )


class SimpleAttentionLayer(nn.Module):
    """Simplified attention layer for demonstration."""
    
    def __init__(self, d_model: int, n_heads: int, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.v_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        self.o_proj = nn.Linear(d_model, d_model, bias=False, device=device)
        
        self.inv_freq = precompute_rope_frequencies(self.head_dim, device=device)
        
    def forward(self, x: torch.Tensor, kv_cache: Optional[Dict] = None, 
                position_offset: int = 0) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with optional KV caching."""
        B, T, D = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        for t in range(T):
            pos = position_offset + t
            q[:, t] = apply_rope(q[:, t], pos, self.inv_freq)
            k[:, t] = apply_rope(k[:, t], pos, self.inv_freq)
            
        if kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)
                v = torch.cat([kv_cache["v"], v], dim=1)
        
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)  # [B, H, T_kv, D]
        v = v.transpose(1, 2)  # [B, H, T_kv, D]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        
        new_cache = {"k": k.transpose(1, 2), "v": v.transpose(1, 2)}
        
        return out, new_cache


class StreamingLLM(nn.Module):
    """Streaming LLM with configurable memory management strategies."""
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, 
                 device: str = "cpu", strategy: str = "streaming_llm"):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.device = device
        self.strategy = strategy
        
        self.embed = nn.Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList([
            SimpleAttentionLayer(d_model, n_heads, device) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)
        
        self.model_cfg = ModelCfg(n_layers, n_heads, self.head_dim)
        self.kv_caches = [{"k": None, "v": None} for _ in range(n_layers)]
        self.position_offset = 0
        
        if strategy == "duostream_mem":
            self.prototype_banks = [[PrototypeBank(self.head_dim, device=device) 
                                   for _ in range(n_heads)] for _ in range(n_layers)]
            self.duo_routers = [DuoAttentionRouter() for _ in range(n_layers)]
            self.eviction_observer = EvictionObserver()
        
        self.budget_manager = None
        self.s = 256  # start tokens
        self.r = 512  # recent tokens
        
    def set_budget(self, B_bytes: int):
        """Set byte budget and allocate memory."""
        self.budget_manager = ByteBudgetManager(self.model_cfg, B_bytes, self.s)
        self.r, M_per_head = self.budget_manager.allocate()
        
        if self.strategy == "duostream_mem":
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    self.prototype_banks[layer_idx][head_idx].M_max = M_per_head[head_idx]
                    
    def _manage_kv_cache(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manage KV cache according to strategy."""
        cache = self.kv_caches[layer_idx]
        
        if cache["k"] is None:
            cache["k"] = new_k
            cache["v"] = new_v
            return new_k, new_v
            
        full_k = torch.cat([cache["k"], new_k], dim=1)
        full_v = torch.cat([cache["v"], new_v], dim=1)
        
        total_len = full_k.size(1)
        
        if total_len <= self.s + self.r:
            cache["k"] = full_k
            cache["v"] = full_v
            return full_k, full_v
            
        if self.strategy == "duostream_mem":
            return self._duostream_eviction(layer_idx, full_k, full_v)
        else:
            return self._streaming_llm_eviction(layer_idx, full_k, full_v)
            
    def _streaming_llm_eviction(self, layer_idx: int, full_k: torch.Tensor, full_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """StreamingLLM eviction: keep start + recent."""
        start_k, start_v = full_k[:, :self.s], full_v[:, :self.s]
        recent_k, recent_v = full_k[:, -self.r:], full_v[:, -self.r:]
        
        kept_k = torch.cat([start_k, recent_k], dim=1)
        kept_v = torch.cat([start_v, recent_v], dim=1)
        
        self.kv_caches[layer_idx]["k"] = kept_k
        self.kv_caches[layer_idx]["v"] = kept_v
        
        return kept_k, kept_v
        
    def _duostream_eviction(self, layer_idx: int, full_k: torch.Tensor, full_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """DuoStream-Mem eviction: keep start + recent + update prototypes."""
        evict_start = self.s
        evict_end = full_k.size(1) - self.r
        
        if evict_end > evict_start:
            evicted_k = full_k[:, evict_start:evict_end]  # [B, chunk_len, H, D]
            evicted_v = full_v[:, evict_start:evict_end]
            
            chunk_len = evicted_k.size(1)
            for head_idx in range(self.n_heads):
                bank = self.prototype_banks[layer_idx][head_idx]
                
                n_select = min(3, chunk_len)
                selected_indices = torch.randperm(chunk_len)[:n_select]
                
                for idx in selected_indices:
                    pos = evict_start + idx
                    k_vec = evicted_k[0, idx, head_idx]  # [D]
                    v_vec = evicted_v[0, idx, head_idx]  # [D]
                    
                    k_base = inverse_rope(k_vec, pos, self.layers[layer_idx].inv_freq)
                    
                    bank.insert_or_update(
                        k_base, v_vec, 
                        importance=1.0, 
                        chunk_id=pos // 128,
                        step=self.position_offset,
                        value_id=idx
                    )
                    
                bank.age_step()
                bank.quantize_all()
        
        start_k, start_v = full_k[:, :self.s], full_v[:, :self.s]
        recent_k, recent_v = full_k[:, -self.r:], full_v[:, -self.r:]
        
        kept_k = torch.cat([start_k, recent_k], dim=1)
        kept_v = torch.cat([start_v, recent_v], dim=1)
        
        self.kv_caches[layer_idx]["k"] = kept_k
        self.kv_caches[layer_idx]["v"] = kept_v
        
        return kept_k, kept_v
        
    def forward(self, input_ids: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """Forward pass with streaming memory management."""
        x = self.embed(input_ids)
        
        for layer_idx, layer in enumerate(self.layers):
            if use_cache:
                cache = self.kv_caches[layer_idx] if self.kv_caches[layer_idx]["k"] is not None else None
                
                x, new_cache = layer(x, cache, self.position_offset)
                
                if new_cache["k"] is not None:
                    self._manage_kv_cache(layer_idx, new_cache["k"], new_cache["v"])
            else:
                x, _ = layer(x, None, self.position_offset)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        self.position_offset += input_ids.size(1)
        
        return logits
        
    def generate_next_token(self, input_ids: torch.Tensor) -> int:
        """Generate next token using the model."""
        with torch.no_grad():
            logits = self.forward(input_ids)
            next_token_logits = logits[0, -1, :]  # Last token, first batch
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), 1)
            return int(next_token.item())
            
    def reset_cache(self):
        """Reset all caches and position offset."""
        self.kv_caches = [{"k": None, "v": None} for _ in range(self.n_layers)]
        self.position_offset = 0
        
        if self.strategy == "duostream_mem":
            for layer_banks in self.prototype_banks:
                for bank in layer_banks:
                    bank.k_base.clear()
                    bank.v_base.clear()
                    bank.meta.clear()
                    bank.qk.clear()
                    bank.qv.clear()
                    bank.kscales.clear()
                    bank.vscales.clear()
                    bank.set_dirty()
                    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage in bytes."""
        local_bytes = 0
        proto_bytes = 0
        
        for cache in self.kv_caches:
            if cache["k"] is not None:
                local_bytes += cache["k"].numel() * 2  # fp16
            if cache["v"] is not None:
                local_bytes += cache["v"].numel() * 2  # fp16
                
        if self.strategy == "duostream_mem":
            for layer_banks in self.prototype_banks:
                for bank in layer_banks:
                    proto_bytes += len(bank.qk) * (bank.d_k // 2 + 4)  # 4-bit + scales
                    proto_bytes += len(bank.qv) * (bank.d_v // 2 + 4)  # 4-bit + scales
                    proto_bytes += len(bank.meta) * 16  # metadata
                    
        return {"local_bytes": local_bytes, "proto_bytes": proto_bytes, "total_bytes": local_bytes + proto_bytes}
