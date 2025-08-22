
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
from typing import Dict, Any, List, Tuple

from preprocess import DecodeConfig, apply_repetition_penalty, sample_next_token, create_decode_config_sampler

class TinyTransformerBlock(nn.Module):
    """Single transformer block with multi-head attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))
        
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x

class TinyLM(nn.Module):
    """Small transformer language model for CAVES experiments."""
    
    def __init__(self, vocab_size: int = 256, d_model: int = 64, n_layers: int = 6, 
                 n_heads: int = 4, d_ff: int = 128, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def causal_mask(self, T: int):
        """Create causal attention mask."""
        mask = torch.full((T, T), float('-inf'), device=next(self.parameters()).device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False) -> Dict[str, Any]:
        """Forward pass through the model."""
        B, T = input_ids.shape
        device = input_ids.device
        
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        
        attn_mask = self.causal_mask(T)
        
        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask)
            if output_hidden_states:
                hidden_states.append(x.clone())
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        out = {'logits': logits}
        if output_hidden_states:
            out['hidden_states'] = hidden_states
        return out

    @torch.no_grad()
    def forward_to(self, input_ids: torch.Tensor, stop_layer: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass up to a specific layer (for cascaded verification)."""
        B, T = input_ids.shape
        device = input_ids.device
        
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        attn_mask = self.causal_mask(T)
        
        for i, block in enumerate(self.blocks):
            x = block(x, attn_mask)
            if (i + 1) == stop_layer:
                break
        
        x = self.ln_f(x)
        h_last = x[:, -1, :]  # Last position hidden state
        return h_last, x

    @torch.no_grad()
    def candidate_only_logits(self, h_last: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for specific candidate tokens only."""
        W = self.lm_head.weight  # [V, H]
        W_sel = torch.index_select(W, dim=0, index=token_ids)
        logits = h_last @ W_sel.t()  # [B, K]
        return logits

class AlphaEstimator(nn.Module):
    """Acceptance estimator a_phi for predicting token acceptance probability."""
    
    def __init__(self, d_hq: int, d_emb: int, d_cfg: int = 13, hidden: int = 256):
        super().__init__()
        self.ln_hq = nn.LayerNorm(d_hq)
        self.ln_emb = nn.LayerNorm(d_emb)
        
        self.fc = nn.Sequential(
            nn.Linear(d_hq + d_emb + d_cfg, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        
        self.head_alpha = nn.Linear(hidden // 2, 1)  # Acceptance probability
        self.head_delta = nn.Linear(hidden // 2, 1)  # Log probability difference

    def forward(self, hq: torch.Tensor, emb_x: torch.Tensor, feat_vec: torch.Tensor):
        """Forward pass through acceptance estimator."""
        x = torch.cat([self.ln_hq(hq), self.ln_emb(emb_x), feat_vec], dim=-1)
        h = self.fc(x)
        
        alpha_logit = self.head_alpha(h).squeeze(-1)
        delta = self.head_delta(h).squeeze(-1)
        
        return alpha_logit, delta

class SpecLogger:
    """Logger for collecting speculative decoding data."""
    
    def __init__(self, mp: TinyLM, mq: TinyLM, vocab_size: int, ban_list: List[int]):
        self.mp = mp
        self.mq = mq
        self.vocab_size = vocab_size
        self.ban_list = set(ban_list)

    @torch.no_grad()
    def collect(self, prompts: List[List[int]], gamma: int = 2, B: int = 2, 
                n_steps: int = 24) -> pd.DataFrame:
        """Collect candidate data for training acceptance estimator."""
        rows = []
        cfg_sampler = create_decode_config_sampler()
        
        for pid, prompt in enumerate(tqdm(prompts, desc='Collecting candidates')):
            device = next(self.mp.parameters()).device
            ids_mp = torch.tensor([prompt], device=device, dtype=torch.long)
            ids_mq = torch.tensor([prompt], device=device, dtype=torch.long)
            
            accepted_hist = []
            
            for step_id in range(n_steps):
                cfg = cfg_sampler()
                T, P, rep_pen, ban_flag = cfg.temperature, cfg.top_p, cfg.rep_pen, cfg.ban_flag
                
                proposals = []
                frontier = [(ids_mq.clone(), [])]
                
                for depth in range(gamma):
                    new_frontier = []
                    for ids, path in frontier:
                        out_q = self.mq(ids, output_hidden_states=True)
                        logits_q = out_q['logits'][:, -1, :]
                        logits_q = apply_repetition_penalty(logits_q.clone(), ids[0], rep_pen)
                        
                        probs_q = F.softmax(logits_q / max(T, 1e-6), dim=-1)
                        topk = torch.topk(probs_q, k=B)
                        
                        sorted_lp, _ = torch.sort(torch.log_softmax(logits_q, dim=-1), descending=True)
                        top2_margin = float((sorted_lp[0, 0] - sorted_lp[0, 1]).item())
                        Hq = float((-(probs_q * torch.log(probs_q + 1e-9)).sum()).item())
                        hq = out_q['hidden_states'][-1][:, -1, :].detach()
                        
                        for k in range(B):
                            tok = int(topk.indices[0, k].item())
                            ban_hit = 1 if (ban_flag == 1 and tok in self.ban_list) else 0
                            qlp = float(torch.log(probs_q[0, tok] + 1e-9).item())
                            emb_x = self.mq.token_embed.weight[tok].detach().cpu().numpy()
                            
                            proposals.append({
                                'prompt_id': pid,
                                'step_id': step_id,
                                'tok': tok,
                                'q_logprob': qlp,
                                'top2_margin': top2_margin,
                                'Hq': Hq,
                                'hq': hq[0].detach().cpu().numpy(),
                                'emb_x': emb_x,
                                'temperature': T,
                                'top_p': P,
                                'rep_pen': rep_pen,
                                'ban_flag': ban_flag,
                                'alpha_hist_mean': float(np.mean(accepted_hist[-4:]) if len(accepted_hist) else 1.0),
                                'dist_eos': float(24 - step_id),
                                'ban_hit': ban_hit,
                                'context_len': int(ids.shape[1]),
                            })
                            
                            new_ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                            new_frontier.append((new_ids, path + [tok]))
                    
                    frontier = new_frontier
                
                out_p = self.mp(ids_mp)
                logits_p = out_p['logits'][:, -1, :]
                logits_p = apply_repetition_penalty(logits_p.clone(), ids_mp[0], rep_pen)
                
                sampled_tok = int(sample_next_token(logits_p, temperature=T, top_p=P)[0].item())
                
                p_lp = torch.log_softmax(logits_p / max(T, 1e-6), dim=-1)
                out_q_now = self.mq(ids_mq)
                q_lp_now = torch.log_softmax(out_q_now['logits'][:, -1, :] / max(T, 1e-6), dim=-1)
                
                for row in proposals:
                    t = row['tok']
                    delta = float((p_lp[0, t] - q_lp_now[0, t]).item())
                    accept = 1 if t == sampled_tok else 0
                    row.update({'delta': delta, 'accept': accept})
                
                ids_mp = torch.cat([ids_mp, torch.tensor([[sampled_tok]], device=device)], dim=-1)
                ids_mq = torch.cat([ids_mq, torch.tensor([[sampled_tok]], device=device)], dim=-1)
                accepted_hist.append(1)
                
                rows.extend(proposals)
        
        return pd.DataFrame(rows)

def train_acceptance_estimator(df: pd.DataFrame, vocab_size: int, device: torch.device, 
                             epochs: int = 10) -> Tuple[AlphaEstimator, IsotonicRegression]:
    """Train the acceptance estimator a_phi."""
    print(f"Training acceptance estimator on {len(df)} samples...")
    
    hq_dim = len(df.iloc[0]['hq'])
    emb_dim = len(df.iloc[0]['emb_x'])
    
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
    
    hq = torch.tensor(np.stack(df['hq'].values), dtype=torch.float32, device=device)
    emb_x = torch.tensor(np.stack(df['emb_x'].values), dtype=torch.float32, device=device)
    feat_vec = torch.tensor(feat_vecs, dtype=torch.float32, device=device)
    
    alpha_labels = torch.tensor(df['accept'].values, dtype=torch.float32, device=device)
    delta_labels = torch.tensor(df['delta'].values, dtype=torch.float32, device=device)
    
    model = AlphaEstimator(hq_dim, emb_dim, len(feat_vecs[0])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        alpha_logits, delta_pred = model(hq, emb_x, feat_vec)
        
        alpha_loss = F.binary_cross_entropy_with_logits(alpha_logits, alpha_labels)
        delta_loss = F.mse_loss(delta_pred, delta_labels)
        
        loss = alpha_loss + 0.1 * delta_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Alpha={alpha_loss.item():.4f}, Delta={delta_loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        alpha_logits, _ = model(hq, emb_x, feat_vec)
        alpha_probs = torch.sigmoid(alpha_logits).cpu().numpy()
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(alpha_probs, df['accept'].values)
    
    print("Training completed!")
    return model, calibrator
