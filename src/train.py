import os
import math
import time
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# =============================================================
# StarCascade core components (kept here so other scripts can import)
# =============================================================


def set_deterministic(seed: int = 0, threads: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import random
        import os as _os
        import numpy as _np
        random.seed(seed)
        _np.random.seed(seed)
        _os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if threads:
        torch.set_num_threads(max(1, threads))
        try:
            torch.set_num_interop_threads(max(1, threads // 2))
        except Exception:
            pass


def device_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class SSWConfig:
    window: int
    strides: List[int]
    star_hubs_per_seg: int
    segment_len: int
    pad_to: int = 64


def build_star_mask(S: int, cfg: SSWConfig) -> torch.Tensor:
    """
    Build deterministic, batch-invariant gather indices for SSW.
    Returns idx: [S, M] int32 with causal indices for each position t, padded to cfg.pad_to.
    Receptive set = {local window} ∪ {stride links} ∪ {star hubs} ∪ {self}.
    """
    strides = sorted(set([int(s) for s in cfg.strides if int(s) > 0]))
    Mlist = []
    for t in range(S):
        neigh = set()
        wstart = max(0, t - cfg.window)
        for u in range(wstart, t + 1):
            neigh.add(u)
        for s in strides:
            ts = t - s
            if ts >= 0:
                neigh.add(ts)
        seg_id = t // cfg.segment_len
        seg_start = seg_id * cfg.segment_len
        if cfg.star_hubs_per_seg > 0:
            step = max(1, cfg.segment_len // cfg.star_hubs_per_seg)
            for i in range(cfg.star_hubs_per_seg):
                h = min(seg_start + i * step, seg_start + cfg.segment_len - 1)
                if h <= t:
                    neigh.add(h)
        neigh = sorted(neigh)
        if len(neigh) < cfg.pad_to:
            neigh = neigh + [neigh[-1]] * (cfg.pad_to - len(neigh))
        else:
            neigh = neigh[-cfg.pad_to:]
        Mlist.append(neigh)
    idx = torch.tensor(Mlist, dtype=torch.int32)
    return idx


@dataclass
class KVPConfig:
    window: int  # recent ring buffer size (full-fidelity)
    levels: int  # L
    anchors_per_seg: int  # A
    segment_len: int  # base segment length for level 0
    recent_bits: int = 8
    anchor_bits: int = 6
    ema_alpha: float = 0.1


class QuantTensor:
    def __init__(self, x: torch.Tensor, bits: int):
        self.bits = bits
        self.scale = x.abs().amax(dim=(-1,), keepdim=True).clamp(min=1e-8) / (2 ** (bits - 1) - 1)
        q = (x / self.scale).round().clamp(min=-(2 ** (bits - 1)), max=2 ** (bits - 1) - 1)
        self.q = q.to(torch.int8 if bits <= 8 else torch.int32)

    def dequant(self) -> torch.Tensor:
        return self.q.to(torch.float32) * self.scale


class KVPyramid:
    """
    Maintains recent ring buffer and multi-level anchors. Shared across groups.
    Single-batch-focused for simplicity.
    """
    def __init__(self, kv_dim: int, cfg: KVPConfig, device: torch.device):
        self.kv_dim = kv_dim
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        self.recent_K: List[QuantTensor] = []
        self.recent_V: List[QuantTensor] = []
        self.anchors_K: List[List[List[Optional[QuantTensor]]]] = []
        self.anchors_V: List[List[List[Optional[QuantTensor]]]] = []
        for _ in range(self.cfg.levels):
            self.anchors_K.append([])
            self.anchors_V.append([])
        self.t = 0

    def _quant(self, x: torch.Tensor, bits: int) -> QuantTensor:
        return QuantTensor(x, bits)

    def update(self, K_t: torch.Tensor, V_t: torch.Tensor):
        # K_t, V_t: [D]
        self.t += 1
        self.recent_K.append(self._quant(K_t, self.cfg.recent_bits))
        self.recent_V.append(self._quant(V_t, self.cfg.recent_bits))
        if len(self.recent_K) > self.cfg.window:
            self.recent_K.pop(0)
            self.recent_V.pop(0)
        # Update anchors via EMA per level
        with torch.no_grad():
            for lvl in range(self.cfg.levels):
                seg_len = self.cfg.segment_len * (2 ** lvl)
                seg_id = (self.t - 1) // seg_len
                while len(self.anchors_K[lvl]) <= seg_id:
                    self.anchors_K[lvl].append([None] * self.cfg.anchors_per_seg)
                    self.anchors_V[lvl].append([None] * self.cfg.anchors_per_seg)
                step = max(1, seg_len // max(1, self.cfg.anchors_per_seg))
                a_idx = min(((self.t - 1) % seg_len) // step, self.cfg.anchors_per_seg - 1)
                alpha = self.cfg.ema_alpha
                for is_k, store in [(True, self.anchors_K[lvl]), (False, self.anchors_V[lvl])]:
                    cur = K_t if is_k else V_t
                    qt = self._quant(cur, self.cfg.anchor_bits)
                    old = store[seg_id][a_idx]
                    if old is None:
                        store[seg_id][a_idx] = qt
                    else:
                        dq = old.dequant()
                        new_dq = (1 - alpha) * dq + alpha * cur
                        store[seg_id][a_idx] = self._quant(new_dq, self.cfg.anchor_bits)

    def gather(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        D = self.kv_dim
        M = indices.shape[0]
        K_out = torch.zeros(M, D, device=self.device)
        V_out = torch.zeros(M, D, device=self.device)
        base = self.t
        for i in range(M):
            idx = int(indices[i].item())
            age = base - (idx + 1)
            if age < 0:
                continue
            if age < len(self.recent_K):
                qtK = self.recent_K[-age - 1]
                qtV = self.recent_V[-age - 1]
                K_out[i] = qtK.dequant()
                V_out[i] = qtV.dequant()
            else:
                # pick nearest available anchor from coarsest applicable level
                for lvl in reversed(range(self.cfg.levels)):
                    seg_len = self.cfg.segment_len * (2 ** lvl)
                    seg_id = idx // seg_len
                    if seg_id < len(self.anchors_K[lvl]):
                        step = max(1, seg_len // max(1, self.cfg.anchors_per_seg))
                        a_idx = min((idx % seg_len) // step, self.cfg.anchors_per_seg - 1)
                        qtK = self.anchors_K[lvl][seg_id][a_idx]
                        qtV = self.anchors_V[lvl][seg_id][a_idx]
                        if qtK is not None:
                            K_out[i] = qtK.dequant()
                            V_out[i] = qtV.dequant()
                            break
        return K_out, V_out


@dataclass
class CGAConfig:
    d_model: int
    groups: int  # G
    c_per_group: int  # Cg
    d_kv: int  # shared K/V dim per token
    ffn_ratio_small: float = 1.0
    ffn_ratio_large: float = 2.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class ClampAct(nn.Module):
    def __init__(self, init=3.0):
        super().__init__()
        self.clamp = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x.clamp(min=-self.clamp, max=self.clamp)


class CGABlock(nn.Module):
    """
    Minimal StarCascade block: per-group Q, shared K/V per token, RMSNorms and clamps
    to reduce activation outliers.
    """
    def __init__(self, cfg: CGAConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        G = cfg.groups
        Cg = cfg.c_per_group
        self.pre_attn = RMSNorm(D)
        self.q_proj = nn.Linear(D, G * Cg, bias=False)
        self.kv_proj = nn.Linear(D, cfg.d_kv * 2, bias=False)
        self.post_attn = nn.Linear(G * Cg, D, bias=False)
        self.per_group_norm = RMSNorm(G * Cg)
        # FFN sandwich
        self.ffn_in = nn.Sequential(
            RMSNorm(D), nn.Linear(D, int(D * cfg.ffn_ratio_small)), nn.SiLU(), ClampAct(), nn.Linear(int(D * cfg.ffn_ratio_small), D)
        )
        self.ffn_out = nn.Sequential(
            RMSNorm(D), nn.Linear(D, int(D * cfg.ffn_ratio_large)), nn.SiLU(), ClampAct(), nn.Linear(int(D * cfg.ffn_ratio_large), D)
        )

    def forward_step(self, h_t: torch.Tensor, mask_idx_t: torch.Tensor, kvp: KVPyramid) -> torch.Tensor:
        """
        h_t: [B, D], mask_idx_t: [M], returns h_{t+1}
        Note: KVP is maintained for batch index 0 for simplicity in this demo.
        """
        B, D = h_t.shape
        G, Cg = self.cfg.groups, self.cfg.c_per_group
        x = self.ffn_in(h_t)
        y = self.pre_attn(x)
        Q = self.q_proj(y).view(B, G, Cg)
        kv = self.kv_proj(y).view(B, 2, self.cfg.d_kv)
        K_t, V_t = kv[:, 0, :], kv[:, 1, :]
        kvp.update(K_t[0].detach(), V_t[0].detach())
        K_hist, V_hist = kvp.gather(mask_idx_t)
        out_groups = []
        for g in range(G):
            qg = Q[:, g, :]  # [B, Cg]
            if Cg != self.cfg.d_kv:
                bridge = F.pad(qg, (0, self.cfg.d_kv - Cg)) if Cg < self.cfg.d_kv else qg[:, : self.cfg.d_kv]
            else:
                bridge = qg
            scores = (K_hist @ bridge[0].unsqueeze(-1)).squeeze(-1) / math.sqrt(self.cfg.d_kv)
            attn = scores.softmax(dim=-1)
            og = (attn.unsqueeze(0) @ V_hist.unsqueeze(0)).view(1, -1)  # [1, Dkv]
            og = og[:, :Cg]
            out_groups.append(og)
        O = torch.cat(out_groups, dim=-1)  # [B, G*Cg]
        O = self.per_group_norm(O)
        z = self.post_attn(O)
        z = z + x  # residual
        z = z + self.ffn_out(z)  # residual
        return z


class EarlyExitGate(nn.Module):
    def __init__(self, d_model: int, groups: int, target_ms: float = 20.0):
        super().__init__()
        self.fc = nn.Linear(d_model, 2)
        self.groups = groups
        self.target_ms = target_ms

    def forward(self, h_t: torch.Tensor) -> int:
        logits = self.fc(h_t)
        probs = logits.softmax(dim=-1)
        conf = probs.max(dim=-1)[0].mean().item()
        g_use = int(max(1, round(self.groups * (1.0 - conf))))
        return g_use


# =============================================================
# Training utilities
# =============================================================


class NeedleDataset(torch.utils.data.IterableDataset):
    def __init__(self, seqlen: int, vocab: int, min_dist: int, max_dist: int, batch_size: int = 8):
        super().__init__()
        self.S = seqlen
        self.vocab = vocab
        self.min_d = min_dist
        self.max_d = max_dist
        self.bs = batch_size

    def __iter__(self):
        while True:
            B = self.bs
            x = torch.randint(0, self.vocab, (B, self.S), dtype=torch.long)
            key = torch.randint(1000, 2000, (B, 1))
            val = torch.randint(2000, 3000, (B, 1))
            dist = torch.randint(self.min_d, self.max_d + 1, (B,))
            for b in range(B):
                qpos = torch.randint(0, max(2, self.S - dist[b] - 2), (1,)).item()
                apos = min(self.S - 2, qpos + dist[b].item())
                x[b, qpos] = 42  # Q marker
                x[b, qpos + 1] = key[b]
                x[b, apos] = 43  # A marker
                x[b, apos + 1] = val[b]
            yield x, key.squeeze(-1), val.squeeze(-1), dist


def sizeof_dtype(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
    }[dtype]


def make_dirs(dirs: Dict[str, str]):
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)


def tiny_lm_block(d_model: int, vocab: int) -> Tuple[nn.Embedding, CGABlock, nn.Linear]:
    emb = nn.Embedding(vocab + 4000, d_model)
    block = CGABlock(CGAConfig(d_model=d_model, groups=4, c_per_group=64, d_kv=128))
    lm_head = nn.Linear(d_model, vocab + 4000, bias=False)
    return emb, block, lm_head


def train_model(config: Dict) -> str:
    # Directories
    dirs = {
        "images": config.get("dirs", {}).get("images_dir", ".research/iteration1/images"),
        "data": config.get("dirs", {}).get("data_dir", "data"),
        "models": config.get("dirs", {}).get("models_dir", "models"),
    }
    make_dirs(dirs)

    # Matplotlib global settings for high-quality PDF (when used via eval script)
    try:
        import matplotlib as mpl
        mpl.rcParams.update({
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
        })
    except Exception:
        pass

    seed = int(config.get("seed", 0))
    set_deterministic(seed)

    device_conf = str(config.get("device", "auto"))
    if device_conf == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_conf)
    print(f"[Train] Using device: {device}")

    # Model dimensions
    cga_cfg = config.get("cga", {})
    d_model = int(cga_cfg.get("d_model", 256))
    vocab = int(config.get("e2", {}).get("vocab", 2000))

    emb, block, head = tiny_lm_block(d_model, vocab)
    emb, block, head = emb.to(device), block.to(device), head.to(device)

    opt = torch.optim.AdamW(list(emb.parameters()) + list(block.parameters()) + list(head.parameters()), lr=1e-3)

    # Dataset
    e2 = config.get("e2", {})
    train_seqlen = int(e2.get("train_seqlen", 512))
    batch_size = int(e2.get("batch_size", 1))
    train_steps = int(e2.get("train_steps", 10))
    min_dist = int(e2.get("min_dist", 64))
    max_dist = int(e2.get("max_dist", 256))

    ds = NeedleDataset(seqlen=train_seqlen, vocab=vocab, min_dist=min_dist, max_dist=max_dist, batch_size=batch_size)
    it = iter(torch.utils.data.DataLoader(ds, batch_size=None))

    # SSW/KVP config
    ssw_conf = config.get("ssw", {})
    ssw = build_star_mask(
        S=train_seqlen,
        cfg=SSWConfig(
            window=int(ssw_conf.get("window", 512)),
            strides=list(ssw_conf.get("strides", [97, 223])),
            star_hubs_per_seg=int(ssw_conf.get("star_hubs_per_seg", 4)),
            segment_len=int(ssw_conf.get("segment_len", 256)),
            pad_to=int(ssw_conf.get("pad_to", 64)),
        ),
    ).to(device)

    kvp_conf = config.get("kvp", {})
    kvp_cfg = KVPConfig(
        window=int(kvp_conf.get("window", 512)),
        levels=int(kvp_conf.get("levels", 2)),
        anchors_per_seg=int(kvp_conf.get("anchors_per_seg", 4)),
        segment_len=int(kvp_conf.get("segment_len", 256)),
        recent_bits=int(kvp_conf.get("recent_bits", 8)),
        anchor_bits=int(kvp_conf.get("anchor_bits", 6)),
        ema_alpha=float(kvp_conf.get("ema_alpha", 0.1)),
    )

    losses_hist = []

    print(f"[Train] Steps={train_steps}  S={train_seqlen}  B={batch_size}")

    for step in range(train_steps):
        x, key, val, dist = next(it)
        x = x.to(device)
        B = x.size(0)
        kvps = [KVPyramid(kv_dim=128, cfg=kvp_cfg, device=device) for _ in range(B)]
        for kvp in kvps:
            kvp.reset()
        h = emb(x[:, 0])
        losses = []
        for t in range(train_seqlen - 1):
            h_next = torch.zeros_like(h)
            logits_list = []
            for b in range(B):
                hb = h[b:b+1]
                hb = block.forward_step(hb, ssw[t, :], kvps[b])
                h_next[b:b+1] = hb
                logits_list.append(head(hb))
            h = h_next
            logits = torch.cat(logits_list, dim=0)
            target = x[:, t + 1]
            loss = F.cross_entropy(logits, target)
            losses.append(loss)
        loss = torch.stack(losses).mean()
        losses_hist.append(float(loss.item()))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, train_steps // 5) == 0:
            print(f"[Train] step {step+1}/{train_steps} | loss {loss.item():.4f}")

    # Save artifacts
    ckpt = {
        "emb": emb.state_dict(),
        "block": block.state_dict(),
        "head": head.state_dict(),
        "meta": {
            "d_model": d_model,
            "vocab": vocab + 4000,
            "ssw": ssw_conf,
            "kvp": kvp_conf,
        },
        "config": config,
        "loss_curve": losses_hist,
    }
    ckpt_path = os.path.join(dirs["models"], "starcascade_e2.pt")
    torch.save(ckpt, ckpt_path)
    print(f"[Train] Saved checkpoint to {ckpt_path}")

    # Plot training loss curve to images dir
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(1, len(losses_hist) + 1), losses_hist, marker='o')
        plt.xlabel('Step')
        plt.ylabel('Training loss (CE)')
        plt.title('E2 training loss')
        plt.grid(True)
        out_pdf = os.path.join(dirs["images"], 'training_loss_e2.pdf')
        plt.savefig(out_pdf, bbox_inches='tight')
        plt.close()
        print(f"[Train] Saved plot: {out_pdf}")
    except Exception as e:
        print(f"[Train] Plotting failed: {e}")

    return ckpt_path


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_model(cfg)
