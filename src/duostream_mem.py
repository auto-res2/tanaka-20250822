"""
DuoStream-Mem: Training-free, byte-fixed streaming memory via quantized prototypes and DuoAttention

This module implements the core components of the DuoStream-Mem method:
- PrototypeBank with 4-bit quantization for keys/values, EMA updates, aging/merging
- DuoAttentionRouter with conservative gating and context fusion
- Strict ByteBudgetManager computing start+recent and prototype bytes
- RoPE utilities and de-rotation at insertion, re-rotation at query time
- EvictionObserver for online importance voting (streaming SnapKV generalization)
"""

import math
import time
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def set_seed(seed: int = 1234):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def precompute_rope_frequencies(d_head: int, base: float = 10000.0, device: str = "cpu"):
    """Precompute RoPE inverse frequencies."""
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    return inv_freq  # [half]


def apply_rope(x: torch.Tensor, pos: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to tensor x at position pos."""
    half = x.shape[-1] // 2
    angles = inv_freq * float(pos)
    sin, cos = torch.sin(angles), torch.cos(angles)
    x1, x2 = x[..., :half], x[..., half:]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    return torch.cat([rot_x1, rot_x2], dim=-1)


def inverse_rope(x: torch.Tensor, pos: int, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply inverse RoPE rotation (de-rotation) to tensor x at position pos."""
    half = x.shape[-1] // 2
    angles = inv_freq * float(pos)
    sin, cos = torch.sin(angles), torch.cos(angles)
    x1, x2 = x[..., :half], x[..., half:]
    rot_x1 = x1 * cos + x2 * sin
    rot_x2 = -x1 * sin + x2 * cos
    return torch.cat([rot_x1, rot_x2], dim=-1)



class FourBitPerChannelQuant:
    """4-bit per-channel quantization for keys."""
    
    def __init__(self, group_size: int = 32, dtype=torch.float16):
        self.group_size = group_size
        self.dtype = dtype

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor x to 4-bit with per-group scales."""
        x = x.view(-1)
        d = x.numel()
        G = self.group_size
        n_groups = (d + G - 1) // G
        pads = n_groups * G - d
        if pads:
            x = F.pad(x, (0, pads))
        x = x.view(n_groups, G)
        max_abs = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = (max_abs / 7.0).to(self.dtype)  # signed 4-bit range [-7, 7]
        q = torch.clamp((x / scale).round(), -7, 7).to(torch.int8)
        return q.reshape(-1).contiguous(), scale.reshape(-1).contiguous()

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor, d: int) -> torch.Tensor:
        """Dequantize 4-bit tensor back to float."""
        G = self.group_size
        n_groups = (d + G - 1) // G
        q = q.view(n_groups, G).to(torch.float32)
        scale = scale.view(n_groups, 1).to(torch.float32)
        x = (q * scale).view(-1)[:d]
        return x


class FourBitPerVectorQuant:
    """4-bit per-vector quantization for values."""
    
    def __init__(self, dtype=torch.float16):
        self.dtype = dtype

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor x to 4-bit with per-vector scale."""
        max_abs = x.abs().amax().clamp(min=1e-8)
        scale = (max_abs / 7.0).to(self.dtype)
        q = torch.clamp((x / scale).round(), -7, 7).to(torch.int8)
        return q.contiguous(), scale.contiguous()

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Dequantize 4-bit tensor back to float."""
        return (q.to(torch.float32) * scale.to(torch.float32)).to(torch.float32)



class PrototypeBank:
    """Per-head prototype bank with EMA updates, aging, merging, and 4-bit quantization.

    Stores keys in base (de-rotated) space; re-apply RoPE at query time.
    Values here are small d_v vectors (same as d_k by default) but can be used
    to carry a value embedding or can be interpreted via metadata (value_id).
    """

    def __init__(self, d_k: int, d_v: Optional[int] = None, M_max: int = 16, device: str = "cpu", group_size: int = 32):
        self.d_k = d_k
        self.d_v = d_v if d_v is not None else d_k
        self.M_max = M_max
        self.device = device
        self.k_quant = FourBitPerChannelQuant(group_size=group_size)
        self.v_quant = FourBitPerVectorQuant()
        
        self.k_base: List[torch.Tensor] = []  # fp16
        self.v_base: List[torch.Tensor] = []  # fp16
        self.meta: List[Dict] = []  # {age:int, ema:float, chunk_id:int, last_hit_step:int, value_id:int}
        
        self.qk: List[torch.Tensor] = []
        self.qv: List[torch.Tensor] = []
        self.kscales: List[torch.Tensor] = []
        self.vscales: List[torch.Tensor] = []
        self._dirty = True  # track whether quantized views need refresh

    def size(self) -> int:
        """Return number of prototypes in bank."""
        return len(self.k_base)

    def set_dirty(self):
        """Mark quantized views as needing refresh."""
        self._dirty = True

    def insert_or_update(self, k_base_vec: torch.Tensor, v_vec: torch.Tensor, importance: float = 1.0, 
                        tau_merge: float = 0.90, ema_m: float = 0.8, chunk_id: int = 0, step: int = 0, 
                        value_id: Optional[int] = None):
        """Insert new prototype or update existing one via EMA if similar enough."""
        if self.size() > 0:
            K = torch.stack(self.k_base, dim=0).to(k_base_vec)
            cos = F.cosine_similarity(k_base_vec.unsqueeze(0), K, dim=-1)
            max_cos, idx = cos.max(dim=0)
        else:
            max_cos, idx = torch.tensor(-1.0, device=self.device), torch.tensor(0, device=self.device)
            
        if max_cos.item() > tau_merge:
            i = int(idx.item())
            self.k_base[i] = (ema_m * self.k_base[i] + (1 - ema_m) * k_base_vec).to(torch.float16)
            self.v_base[i] = (ema_m * self.v_base[i] + (1 - ema_m) * v_vec).to(torch.float16)
            self.meta[i]["ema"] = ema_m * self.meta[i]["ema"] + (1 - ema_m) * importance
            self.meta[i]["age"] = 0
            self.meta[i]["last_hit_step"] = step
            if value_id is not None:
                self.meta[i]["value_id"] = int(value_id)
        else:
            if self.size() >= self.M_max:
                self._merge_or_evict()
            self.k_base.append(k_base_vec.to(torch.float16).detach())
            self.v_base.append(v_vec.to(torch.float16).detach())
            self.meta.append({
                "age": 0, 
                "ema": float(importance), 
                "chunk_id": int(chunk_id), 
                "last_hit_step": int(step), 
                "value_id": int(value_id) if value_id is not None else -1
            })
        self._dirty = True

    def age_step(self):
        """Increment age of all prototypes."""
        for m in self.meta:
            m["age"] += 1

    def _merge_or_evict(self):
        """Merge nearest far-tier pair by cosine or evict oldest if only one candidate."""
        ages = torch.tensor([m["age"] for m in self.meta], device=self.device)
        far_mask = ages > torch.quantile(ages.float(), 0.5) if self.size() > 1 else torch.tensor([True], device=self.device)
        idxs = torch.nonzero(far_mask).view(-1)
        
        if idxs.numel() < 2:
            i = int(ages.argmax().item())
            self._pop_index(i)
            return
            
        K = F.normalize(torch.stack(self.k_base, dim=0).to(torch.float32), dim=-1)
        subK = K[idxs]
        sim = subK @ subK.t() - torch.eye(subK.size(0), device=subK.device)
        i_sub, j_sub = torch.nonzero(sim == sim.max()).tolist()[0]
        i, j = int(idxs[i_sub].item()), int(idxs[j_sub].item())
        
        m_i, m_j = self.meta[i]["ema"], self.meta[j]["ema"]
        w_i = m_i / (m_i + m_j + 1e-6)
        w_j = m_j / (m_i + m_j + 1e-6)
        self.k_base[i] = (w_i * self.k_base[i].float() + w_j * self.k_base[j].float()).to(torch.float16)
        self.v_base[i] = (w_i * self.v_base[i].float() + w_j * self.v_base[j].float()).to(torch.float16)
        
        self.meta[i]["ema"] = float(m_i + m_j)
        self.meta[i]["age"] = min(self.meta[i]["age"], self.meta[j]["age"]) // 2
        if self.meta[j]["value_id"] != -1:
            self.meta[i]["value_id"] = self.meta[j]["value_id"]
        self._pop_index(j)

    def _pop_index(self, j: int):
        """Remove prototype at index j."""
        for arr in [self.k_base, self.v_base, self.qk, self.qv, self.kscales, self.vscales, self.meta]:
            if len(arr) > j:
                arr.pop(j)
        self._dirty = True

    def quantize_all(self):
        """Quantize all prototypes if dirty."""
        if not self._dirty:
            return
        self.qk, self.qv, self.kscales, self.vscales = [], [], [], []
        for k, v in zip(self.k_base, self.v_base):
            qk, ks = self.k_quant.quantize(k.float())
            qv, vs = self.v_quant.quantize(v.float())
            self.qk.append(qk.to(torch.int8))
            self.kscales.append(ks.to(torch.float16))
            self.qv.append(qv.to(torch.int8))
            self.vscales.append(vs.to(torch.float16))
        self._dirty = False

    def dequant_keys_values(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Dequantize and return all prototype keys and values."""
        M = self.size()
        if M == 0:
            return None, None
        K = torch.stack([self.k_quant.dequantize(qk, ks, self.d_k) for qk, ks in zip(self.qk, self.kscales)], dim=0).to(self.device)
        V = torch.stack([self.v_quant.dequantize(qv, vs) for qv, vs in zip(self.qv, self.vscales)], dim=0).to(self.device)
        return K, V


class DuoAttentionRouter:
    """Conservative gating mechanism for fusing local and global attention routes."""
    
    def __init__(self, alpha=6.0, beta=1.0, gamma=-3.0, zeta=-1.0, delta=-2.0, cap=0.4, b_proto=-1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.delta = delta
        self.cap = cap
        self.b_proto = b_proto

    @torch.no_grad()
    def compute_gate(self, q: torch.Tensor, K_proto: Optional[torch.Tensor], local_logits: torch.Tensor, 
                    proto_age: Optional[torch.Tensor] = None) -> float:
        """Compute gating value for prototype route activation."""
        if K_proto is None or K_proto.numel() == 0:
            return 0.0
            
        qn = F.normalize(q, dim=-1)
        Kn = F.normalize(K_proto, dim=-1)
        cos = (Kn @ qn)  # [M]
        max_cos = float(cos.max().item())
        
        prob = F.softmax(local_logits, dim=-1)
        entropy = float((-(prob * (prob + 1e-9).log()).sum()).item())
        
        top2_vals, _ = torch.topk(local_logits, k=min(2, local_logits.numel()))
        if top2_vals.numel() == 2:
            margin = float((top2_vals[0] - top2_vals[1]).item())
        else:
            margin = float(top2_vals[0].item())
            
        if proto_age is None or proto_age.numel() == 0:
            age_bias = 0.0
        else:
            age_norm = proto_age.float().mean().item()
            age_mean = max(1e-6, float(age_norm))
            age_bias = min(4.0, age_norm / age_mean)
            
        g_pre = self.alpha * max_cos + self.beta * entropy + self.gamma * margin + self.zeta * age_bias + self.delta
        g = 1.0 / (1.0 + math.exp(-g_pre))
        return float(min(g, self.cap))



@dataclass
class ModelCfg:
    """Model configuration for byte budget calculations."""
    n_layers: int
    n_heads: int
    head_dim: int


class ByteBudgetManager:
    """Manages byte allocation between local cache and prototype banks."""
    
    def __init__(self, model_cfg: ModelCfg, B_total_bytes: int, s: int = 256, dtype_bytes: int = 2, 
                 proto_meta_bytes: int = 16, group_size: int = 32):
        self.cfg = model_cfg
        self.B = int(B_total_bytes)
        self.s = int(s)
        self.dtype_bytes = int(dtype_bytes)
        self.proto_meta_bytes = int(proto_meta_bytes)
        self.group_size = int(group_size)
        self.r: int = 0
        self.M_head: List[int] = [0] * self.cfg.n_heads

    def compute_proto_bytes_per_head(self, d_k: int, d_v: int) -> int:
        """Compute bytes needed per prototype per head."""
        return int(0.5 * (d_k + d_v) + 2 * (math.ceil(d_k / self.group_size) + 1) + self.proto_meta_bytes)

    def allocate(self, M_cap: int = 16, local_ratio: float = 0.7) -> Tuple[int, List[int]]:
        """Allocate bytes between local cache (r) and prototype banks (M_head)."""
        L, H, d_k = self.cfg.n_layers, self.cfg.n_heads, self.cfg.head_dim
        bytes_per_token_per_layer = H * (2 * self.dtype_bytes) * d_k * 2  # K and V
        
        B_local_target = int(local_ratio * self.B)
        r_guess = max(32, (B_local_target // (L * bytes_per_token_per_layer)) - self.s)
        
        B_used_local = L * (self.s + r_guess) * bytes_per_token_per_layer
        proto_bph = self.compute_proto_bytes_per_head(d_k, d_k)
        B_remain = max(0, self.B - B_used_local)
        total_protos_per_layer = min((B_remain // (L * proto_bph)), H * M_cap)
        
        M_per_head = [0] * H
        if H > 0 and total_protos_per_layer > 0:
            base = total_protos_per_layer // H
            rem = total_protos_per_layer % H
            for h in range(H):
                M_per_head[h] = min(M_cap, base + (1 if h < rem else 0))
                
        self.r = int(r_guess)
        self.M_head = [int(m) for m in M_per_head]
        return self.r, self.M_head

    def bytes_local(self, r: Optional[int] = None) -> int:
        """Compute bytes used by local cache."""
        r = self.r if r is None else r
        L, H, d_k = self.cfg.n_layers, self.cfg.n_heads, self.cfg.head_dim
        bytes_per_token_per_layer = H * (2 * self.dtype_bytes) * d_k * 2
        return int(L * (self.s + r) * bytes_per_token_per_layer)

    def bytes_protos(self, M_per_head: Optional[List[int]] = None) -> int:
        """Compute bytes used by prototype banks."""
        M_per_head = self.M_head if M_per_head is None else M_per_head
        L, d_k = self.cfg.n_layers, self.cfg.head_dim
        proto_bph = self.compute_proto_bytes_per_head(d_k, d_k)
        return int(L * sum(M_per_head) * proto_bph)

    def total_bytes(self, r: Optional[int] = None, M_per_head: Optional[List[int]] = None) -> int:
        """Compute total bytes used."""
        return self.bytes_local(r) + self.bytes_protos(M_per_head)



class EvictionObserver:
    """Online importance voting for evicted chunks (streaming SnapKV generalization)."""
    
    def __init__(self, w: int = 64, n_heads_sample: int = 4):
        self.w = w  # observation window
        self.n_heads_sample = n_heads_sample
        self.reset()

    def reset(self):
        """Reset observation state."""
        self.chunk_tokens = []
        self.chunk_positions = []
        self.importance_scores = []
        self.step_count = 0

    def start_observation(self, chunk_tokens: List[int], chunk_positions: List[int]):
        """Start observing importance for evicted chunk."""
        self.chunk_tokens = chunk_tokens
        self.chunk_positions = chunk_positions
        self.importance_scores = [0.0] * len(chunk_tokens)
        self.step_count = 0

    def accumulate_step(self, attention_weights: torch.Tensor, head_indices: List[int]):
        """Accumulate attention-based importance for one step."""
        if self.step_count >= self.w or not self.chunk_tokens:
            return
            
        chunk_start = len(attention_weights[0]) - len(self.chunk_tokens)
        chunk_attn = attention_weights[head_indices, chunk_start:]  # [n_sample_heads, chunk_len]
        
        step_importance = chunk_attn.mean(dim=0).cpu().numpy()
        for i, imp in enumerate(step_importance):
            self.importance_scores[i] += float(imp)
            
        self.step_count += 1

    def finalize_importance(self, value_norms: Optional[List[float]] = None, 
                          surprises: Optional[List[float]] = None) -> List[Tuple[int, float]]:
        """Finalize importance scores with priors and return sorted token-importance pairs."""
        if not self.chunk_tokens:
            return []
            
        if self.step_count > 0:
            att_scores = [s / self.step_count for s in self.importance_scores]
        else:
            att_scores = [1.0] * len(self.chunk_tokens)
            
        final_scores = []
        for i, att_score in enumerate(att_scores):
            score = att_score
            if value_norms:
                score += 0.1 * value_norms[i]  # value L2 norm prior
            if surprises:
                score += 0.2 * surprises[i]  # surprise prior
            final_scores.append(score)
            
        if len(final_scores) > 1:
            mean_score = np.mean(final_scores)
            std_score = np.std(final_scores) + 1e-6
            final_scores = [(s - mean_score) / std_score for s in final_scores]
            
        token_importance = list(zip(self.chunk_tokens, final_scores))
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance



class SyntheticStreamingTask:
    """Synthetic fact-query language for testing long-range recall."""
    
    def __init__(self, vocab_size: int = 1000, max_facts: int = 100, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.max_facts = max_facts
        self.device = device
        self.facts = {}  # key -> value mapping
        
    def generate_stream(self, length: int, query_prob: float = 0.1, fact_prob: float = 0.3) -> List[int]:
        """Generate synthetic streaming sequence with facts and queries."""
        stream = []
        positions_with_facts = []
        
        for pos in range(length):
            if random.random() < fact_prob and len(self.facts) < self.max_facts:
                key = random.randint(1, self.vocab_size // 2)
                value = random.randint(self.vocab_size // 2 + 1, self.vocab_size - 1)
                self.facts[key] = value
                stream.extend([key, 0, value])  # 0 is "=" token
                positions_with_facts.append(pos)
            elif random.random() < query_prob and self.facts:
                key = random.choice(list(self.facts.keys()))
                stream.extend([key, 0])  # query without answer
            else:
                stream.append(random.randint(1, self.vocab_size - 1))
                
        return stream
        
    def evaluate_recall(self, model_predictions: List[int], stream: List[int]) -> Dict[str, float]:
        """Evaluate recall accuracy on queries in stream."""
        correct = 0
        total_queries = 0
        
        i = 0
        while i < len(stream) - 1:
            if i < len(stream) - 2 and stream[i+1] == 0:  # Found key=
                key = stream[i]
                if i + 2 < len(stream):
                    i += 3
                else:
                    if key in self.facts and i < len(model_predictions):
                        predicted = model_predictions[i+1]  # Predict token after "="
                        if predicted == self.facts[key]:
                            correct += 1
                        total_queries += 1
                    i += 2
            else:
                i += 1
                
        accuracy = correct / max(1, total_queries)
        return {"accuracy": accuracy, "correct": correct, "total": total_queries}
