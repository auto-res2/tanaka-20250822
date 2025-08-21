
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from preprocess import Claim, SyntheticFeatureBuilder, ESCALATION_STEPS

try:
    from sklearn.isotonic import IsotonicRegression
    SK_ISO_AVAILABLE = True
except Exception:
    SK_ISO_AVAILABLE = False


class ConsistencyHead(nn.Module):
    def __init__(self, in_dim: int = 12, hidden: int = 64, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class EvidencePooler(nn.Module):
    def __init__(self, in_dim: int = 64, m_tokens: int = 8, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=in_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.assign = nn.Linear(in_dim, m_tokens)
        self.ln = nn.LayerNorm(in_dim)

    def forward(self, token_matrix: torch.Tensor, attr_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(attr_scores, dim=-1).unsqueeze(-1)
        x = token_matrix * w
        for layer in self.enc_layers:
            x = layer(x)
        x = self.ln(x)
        logits = self.assign(x)           # [B, T, M]
        A = torch.softmax(logits, dim=1)  # [B, T, M]
        A = A.transpose(1, 2)             # [B, M, T]
        evid = A @ x                      # [B, M, H]
        return evid, A


class IsoOrHistogram:
    def __init__(self):
        self.is_iso = SK_ISO_AVAILABLE
        self.ir = None
        self.bin_edges = None
        self.bin_vals = None

    def fit(self, scores: np.ndarray, fails: np.ndarray):
        if self.is_iso:
            self.ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
            self.ir.fit(scores, fails)
        else:
            bins = np.linspace(scores.min(), scores.max() + 1e-8, 11)
            idx = np.digitize(scores, bins) - 1
            means = []
            for b in range(10):
                m = fails[idx == b].mean() if np.any(idx == b) else 0.5
                means.append(m)
            self.bin_edges = bins
            self.bin_vals = np.array(means)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.is_iso and self.ir is not None:
            return self.ir.predict(scores)
        if self.bin_edges is None or self.bin_vals is None:
            return np.full_like(scores, 0.5)
        idx = np.digitize(scores, self.bin_edges) - 1
        idx = np.clip(idx, 0, len(self.bin_vals) - 1)
        return self.bin_vals[idx]


class ConformalG:
    def __init__(self):
        self.bucket_map: Dict[Tuple, IsoOrHistogram] = {}
        self.global_model = IsoOrHistogram()
        self.has_global = False

    def fit(self, calib_rows: List[Dict]):
        buckets: Dict[Tuple, List[Tuple[float, int]]] = {}
        all_scores, all_fails = [], []
        for r in calib_rows:
            pb = r['plan_bucket']
            buckets.setdefault(pb, []).append((r['score'], r['fail']))
            all_scores.append(r['score'])
            all_fails.append(r['fail'])
        for pb, rows in buckets.items():
            xs = np.array([x for x, _ in rows], dtype=np.float64)
            ys = np.array([y for _, y in rows], dtype=np.float64)
            model = IsoOrHistogram()
            model.fit(xs, ys)
            self.bucket_map[pb] = model
        xs_all = np.array(all_scores, dtype=np.float64)
        ys_all = np.array(all_fails, dtype=np.float64)
        self.global_model.fit(xs_all, ys_all)
        self.has_global = True

    def bound(self, score: float, plan_bucket: Tuple) -> float:
        if plan_bucket in self.bucket_map:
            return float(self.bucket_map[plan_bucket].predict(np.array([score]))[0])
        if self.has_global:
            return float(self.global_model.predict(np.array([score]))[0] + 0.05)  # conservative pad
        return 1.0


def train_consistency_head(claims: List[Claim], device: str = 'cpu', epochs: int = 20, batch_size: int = 32) -> Tuple[ConsistencyHead, ConformalG]:
    """Train the consistency head and fit conformal calibrator."""
    print(f"Training consistency head on {len(claims)} claims...")
    
    feat_builder = SyntheticFeatureBuilder()
    head = ConsistencyHead(in_dim=12, hidden=64, num_classes=3).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train, calib_data = [], [], []
    
    for claim in claims:
        for plan in ESCALATION_STEPS:
            feat_vec, aux = feat_builder.build_features(claim, plan, seed=42)
            X_train.append(feat_vec)
            y_train.append(claim.label)
            
            plan_bucket = (plan['dim'], plan['k'], plan['hops'], tuple(sorted(plan['sources'])), plan['ens'])
            fail = 1 if claim.label != 0 else 0  # fail if not supported
            calib_data.append({
                'score': aux['score'],
                'fail': fail,
                'plan_bucket': plan_bucket
            })
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.long)
    
    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    head.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    print("Fitting conformal calibrator...")
    conformal_g = ConformalG()
    conformal_g.fit(calib_data)
    
    print("Training completed!")
    return head, conformal_g
