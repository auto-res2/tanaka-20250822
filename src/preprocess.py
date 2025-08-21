
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Claim:
    text: str
    label: int  # 0=Supported, 1=Insufficient, 2=Conflicted
    hardness: float  # [0,1], higher = harder
    id: int


class SyntheticFeatureBuilder:
    def __init__(self, n_views: int = 3, noise_scale: float = 0.08):
        self.n_views = n_views
        self.noise_scale = noise_scale

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def per_view_probs(self, claim: Claim, plan: Dict, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.RandomState(seed + claim.id + plan.get('level', 0) * 17)
        else:
            rng = np.random
        L = plan.get('level', 0)
        h = claim.hardness
        base_noise = self.noise_scale * (1.0 + 0.5 * h)
        probs = []
        for v in range(self.n_views):
            z = rng.randn(3) * base_noise
            if claim.label == 0:  # Supported
                pS = min(0.55 + 0.12 * L - 0.15 * h + z[0], 0.97)
                pC = max(0.03 + 0.02 * L + 0.05 * h + z[1], 0.01)
                pI = 1.0 - pS - pC
            elif claim.label == 1:  # Insufficient
                pS = max(0.15 + 0.05 * L - 0.1 * h + z[0], 0.01)
                pC = max(0.05 + 0.03 * L + 0.05 * h + z[1], 0.01)
                pI = 1.0 - pS - pC
            else:  # Conflicted
                pC = min(0.50 + 0.10 * L - 0.05 * h + z[2], 0.97)
                pS = max(0.10 + 0.02 * L - 0.05 * h + z[0], 0.01)
                pI = 1.0 - pS - pC
            vec = np.array([pS, pI, pC])
            vec = np.clip(vec, 1e-3, 1.0)
            vec = vec / vec.sum()
            probs.append(vec)
        return np.vstack(probs)

    def build_features(self, claim: Claim, plan: Dict, seed: Optional[int] = None,
                       prev_pS: Optional[float] = None, ensemble_var: float = 0.0) -> Tuple[np.ndarray, Dict[str, float]]:
        probs_views = self.per_view_probs(claim, plan, seed=seed)
        p_means = probs_views.mean(axis=0)
        p_vars = probs_views.var(axis=0)
        coverage = min(1.0, 0.4 + 0.15 * plan.get('level', 0) - 0.2 * claim.hardness + np.random.randn() * 0.02)
        diversity = min(1.0, 0.5 + 0.1 * plan.get('hops', 0) + 0.05 * len(plan.get('sources', [])) + np.random.randn() * 0.02)
        conflict_score = float(probs_views[:, 2].max())
        entropy = float(-(p_means * np.log(p_means + 1e-8)).sum())
        pS_curr = float(p_means[0])
        delta_speed = 0.0 if prev_pS is None else float(pS_curr - prev_pS)
        feat_vec = np.concatenate([
            p_means, p_vars,
            np.array([coverage, diversity, conflict_score, entropy, delta_speed, ensemble_var], dtype=np.float32)
        ]).astype('float32')
        aux = {
            'p_support': pS_curr,
            'coverage': coverage,
            'conflict': conflict_score,
            'entropy': entropy,
            'score': 1.0 - pS_curr
        }
        return feat_vec, aux


class MatryoshkaRetrieverSim:
    def __init__(self, D_full: int = 512, dims: Tuple[int, ...] = (64, 128, 256, 512)):
        self.D_full = D_full
        self.dims = dims
        self.docs = None  # [N, D_full]

    def build(self, n_docs: int = 500, seed: int = 123):
        rng = np.random.RandomState(seed)
        X = rng.randn(n_docs, self.D_full).astype('float32')
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        self.docs = X

    def search(self, query_vec: np.ndarray, d: int, k: int, seed: int = 0) -> List[int]:
        assert self.docs is not None
        rng = np.random.RandomState(seed)
        q = query_vec[:d]
        Xd = self.docs[:, :d]
        noise = rng.randn(*Xd.shape) * 0.001
        sims = Xd.dot(q) + (noise @ q)
        topk = np.argsort(-sims)[:k]
        return topk.tolist()


ESCALATION_STEPS = [
    {"dim": 64,  "k": 10, "hops": 0, "sources": {"local"},                  "ens": 1, "level": 0},
    {"dim": 128, "k": 20, "hops": 0, "sources": {"local"},                  "ens": 1, "level": 1},
    {"dim": 256, "k": 20, "hops": 1, "sources": {"local", "web"},          "ens": 1, "level": 2},
    {"dim": 256, "k": 40, "hops": 1, "sources": {"local", "web"},          "ens": 2, "level": 3},
    {"dim": 512, "k": 50, "hops": 2, "sources": {"local", "web", "kg"},    "ens": 3, "level": 4},
]


def generate_synthetic_data(n_claims: int = 200, n_docs: int = 500, seed: int = 42) -> Tuple[List[Claim], MatryoshkaRetrieverSim]:
    """Generate synthetic claims and retriever for experiments."""
    np.random.seed(seed)
    
    claims = []
    for i in range(n_claims):
        label = i % 3
        hardness = 0.1 + 0.8 * (i / n_claims)
        text = f"Synthetic claim {i} with label {label}"
        claims.append(Claim(text=text, label=label, hardness=hardness, id=i))
    
    retriever = MatryoshkaRetrieverSim()
    retriever.build(n_docs=n_docs, seed=seed)
    
    print(f"Generated {len(claims)} claims and retriever with {retriever.docs.shape[0] if retriever.docs is not None else 0} documents")
    
    return claims, retriever
