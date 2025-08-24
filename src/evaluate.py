import os
import math
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reuse core classes from train.py
from train import (
    SSWConfig,
    KVPConfig,
    CGAConfig,
    CGABlock,
    KVPyramid,
    EarlyExitGate,
    build_star_mask,
    device_sync,
    set_deterministic,
)


def mean_ci(xs: List[float]) -> Tuple[float, float]:
    import statistics
    if len(xs) == 0:
        return 0.0, 0.0
    m = statistics.mean(xs)
    if len(xs) == 1:
        return m, 0.0
    s = statistics.stdev(xs)
    ci = 1.96 * s / math.sqrt(len(xs))
    return m, ci


def sizeof_dtype(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.int32: 4,
    }[dtype]


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    d_model = int(meta.get("d_model", 256))
    vocab = int(meta.get("vocab", 6000))
    emb = nn.Embedding(vocab, d_model).to(device)
    block = CGABlock(CGAConfig(d_model=d_model, groups=4, c_per_group=64, d_kv=128)).to(device)
    head = nn.Linear(d_model, vocab, bias=False).to(device)
    emb.load_state_dict(ckpt["emb"])  
    block.load_state_dict(ckpt["block"]) 
    head.load_state_dict(ckpt["head"])  
    return emb, block, head, ckpt


def exp1_throughput(config: Dict):
    dirs = config.get("dirs", {})
    images_dir = dirs.get("images_dir", ".research/iteration1/images")
    os.makedirs(images_dir, exist_ok=True)

    device_conf = str(config.get("device", "auto"))
    device = torch.device("cuda" if (device_conf == "auto" and torch.cuda.is_available()) else device_conf)

    set_deterministic(int(config.get("seed", 0)))

    print("[E1] PyTorch:", torch.__version__, "Device:", device)

    D = int(config.get("cga", {}).get("d_model", 256))
    G = 4
    Cg = 64
    Dkv = 128
    block = CGABlock(CGAConfig(d_model=D, groups=G, c_per_group=Cg, d_kv=Dkv)).to(device)

    e1 = config.get("e1", {})
    seqlens = list(e1.get("seqlens", [256, 512, 1024]))

    ssw_conf = config.get("ssw", {})
    max_S = max(seqlens)
    ssw = build_star_mask(
        S=max_S,
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
    kvp = KVPyramid(kv_dim=Dkv, cfg=kvp_cfg, device=device)

    sc_S, sc_ms, sc_kvmb = [], [], []

    def run_seq(S):
        kvp.reset()
        h = torch.randn(1, D, device=device)
        # Warm-up few tokens
        for t in range(min(5, S)):
            h = block.forward_step(h, ssw[t, :], kvp)
        device_sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for t in range(S):
            h = block.forward_step(h, ssw[t, :], kvp)
        device_sync(device)
        t1 = time.perf_counter()
        ms_tok = (t1 - t0) * 1000.0 / S
        # KV footprint estimate
        recent_bytes = int(kvp_cfg.window) * (sizeof_dtype(torch.int8) * 2) * Dkv
        anchors = 0
        for lvl in range(kvp_cfg.levels):
            num_segs = math.ceil(S / (kvp_cfg.segment_len * (2 ** lvl)))
            anchors += num_segs * kvp_cfg.anchors_per_seg
        anchor_bytes = anchors * (sizeof_dtype(torch.int8) * 2) * Dkv
        kv_mb = (recent_bytes + anchor_bytes) / 1e6
        return ms_tok, kv_mb

    runs = int(e1.get("runs", 2))

    print("[E1] StarCascade (SSW + Shared-KV + KVP)")
    for S in seqlens:
        run_times = []
        for _ in range(runs):
            ms_tok, _ = run_seq(S)
            run_times.append(ms_tok)
        ms, ci = mean_ci(run_times)
        _, kv_mb = run_seq(S)
        sc_S.append(S)
        sc_ms.append(ms)
        sc_kvmb.append(kv_mb)
        print(f"[E1] S={S:7d} | {ms:.2f}±{ci:.2f} ms/tok | KV≈{kv_mb:8.1f} MB")

    # Baseline: sliding window only (no anchors)
    print("[E1] Baseline: Sliding window only (no KVP anchors)")
    ssw_sw = build_star_mask(
        S=max_S,
        cfg=SSWConfig(
            window=int(ssw_conf.get("window", 512)),
            strides=[], star_hubs_per_seg=0,
            segment_len=int(ssw_conf.get("segment_len", 256)),
            pad_to=int(ssw_conf.get("pad_to", 64)),
        ),
    ).to(device)
    sw_ms, sw_kvmb = [], []
    for S in seqlens:
        kvp_sw = KVPyramid(kv_dim=Dkv, cfg=KVPConfig(window=int(kvp_conf.get("window", 512)), levels=0, anchors_per_seg=0, segment_len=int(kvp_conf.get("segment_len", 256))), device=device)
        kvp_sw.reset()
        h = torch.randn(1, D, device=device)
        for _ in range(min(5, S)):
            h = block.forward_step(h, ssw_sw[0, :], kvp_sw)
        device_sync(device)
        t0 = time.perf_counter()
        for t in range(S):
            h = block.forward_step(h, ssw_sw[t, :], kvp_sw)
        device_sync(device)
        t1 = time.perf_counter()
        ms_tok = (t1 - t0) * 1000.0 / S
        sw_ms.append(ms_tok)
        recent_bytes = int(kvp_conf.get("window", 512)) * (sizeof_dtype(torch.int8) * 2) * Dkv
        sw_kvmb.append(recent_bytes / 1e6)
        print(f"[E1] SW S={S:7d} | {ms_tok:.2f} ms/tok | KV≈{(recent_bytes/1e6):8.1f} MB")

    # Plots to images dir
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(sc_S, sc_ms, marker='o', label='StarCascade')
    plt.plot(sc_S, sw_ms, marker='s', label='SlidingWindow')
    plt.xlabel('Sequence length (tokens)')
    plt.ylabel('ms/token (decode)')
    plt.title('Decode latency vs sequence length')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    out_pdf = os.path.join(images_dir, 'inference_latency_starcascade.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[E1] Saved {out_pdf}")

    plt.figure()
    plt.plot(sc_S, sc_kvmb, marker='o', label='StarCascade (KVP)')
    plt.plot(sc_S, sw_kvmb, marker='s', label='SlidingWindow')
    plt.xlabel('Sequence length (tokens)')
    plt.ylabel('KV footprint (MB, estimated)')
    plt.title('KV memory vs sequence length')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    out_pdf = os.path.join(images_dir, 'kv_footprint_starcascade.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[E1] Saved {out_pdf}")


def eval_retrieval(config: Dict, ckpt_path: str):
    dirs = config.get("dirs", {})
    images_dir = dirs.get("images_dir", ".research/iteration1/images")
    os.makedirs(images_dir, exist_ok=True)

    device_conf = str(config.get("device", "auto"))
    device = torch.device("cuda" if (device_conf == "auto" and torch.cuda.is_available()) else device_conf)

    set_deterministic(int(config.get("seed", 0)))

    emb, block, head, ckpt = load_model_from_ckpt(ckpt_path, device)
    meta = ckpt.get("meta", {})
    vocab = int(meta.get("vocab", 6000)) - 4000

    e2 = config.get("e2", {})
    eval_seqlen = int(e2.get("eval_seqlen", 1024))
    eval_trials = int(e2.get("eval_trials", 20))
    eval_distances = list(e2.get("eval_distances", [64, 128, 256, 512]))

    ssw_conf = config.get("ssw", {})
    ssw_sc = build_star_mask(
        S=eval_seqlen,
        cfg=SSWConfig(
            window=int(ssw_conf.get("window", 512)),
            strides=list(ssw_conf.get("strides", [97, 223])),
            star_hubs_per_seg=int(ssw_conf.get("star_hubs_per_seg", 4)),
            segment_len=int(ssw_conf.get("segment_len", 256)),
            pad_to=int(ssw_conf.get("pad_to", 64)),
        ),
    ).to(device)
    ssw_sw = build_star_mask(
        S=eval_seqlen,
        cfg=SSWConfig(
            window=int(ssw_conf.get("window", 512)),
            strides=[], star_hubs_per_seg=0,
            segment_len=int(ssw_conf.get("segment_len", 256)),
            pad_to=int(ssw_conf.get("pad_to", 64)),
        ),
    ).to(device)

    kvp_conf = config.get("kvp", {})
    kvp_cfg_sc = KVPConfig(
        window=int(kvp_conf.get("window", 512)),
        levels=int(kvp_conf.get("levels", 2)),
        anchors_per_seg=int(kvp_conf.get("anchors_per_seg", 4)),
        segment_len=int(kvp_conf.get("segment_len", 256)),
        recent_bits=int(kvp_conf.get("recent_bits", 8)),
        anchor_bits=int(kvp_conf.get("anchor_bits", 6)),
        ema_alpha=float(kvp_conf.get("ema_alpha", 0.1)),
    )
    kvp_cfg_sw = KVPConfig(window=int(kvp_conf.get("window", 512)), levels=0, anchors_per_seg=0, segment_len=int(kvp_conf.get("segment_len", 256)))

    def run_eval_variant(dist: int, use_starcascade: bool):
        class _DS(torch.utils.data.IterableDataset):
            def __init__(self, seqlen: int, vocab: int, dist: int):
                super().__init__()
                self.S = seqlen
                self.vocab = vocab
                self.dist = dist
            def __iter__(self):
                while True:
                    x = torch.randint(0, self.vocab, (1, self.S), dtype=torch.long)
                    key = torch.randint(1000, 2000, (1, 1))
                    val = torch.randint(2000, 3000, (1, 1))
                    qpos = torch.randint(0, max(2, self.S - self.dist - 2), (1,)).item()
                    apos = min(self.S - 2, qpos + self.dist)
                    x[0, qpos] = 42; x[0, qpos + 1] = key
                    x[0, apos] = 43; x[0, apos + 1] = val
                    yield x, key.squeeze(-1), val.squeeze(-1)
        it = iter(torch.utils.data.DataLoader(_DS(eval_seqlen, vocab, dist), batch_size=None))
        correct = 0
        if use_starcascade:
            ssw = ssw_sc
            kvp = KVPyramid(kv_dim=128, cfg=kvp_cfg_sc, device=device)
        else:
            ssw = ssw_sw
            kvp = KVPyramid(kv_dim=128, cfg=kvp_cfg_sw, device=device)
        with torch.no_grad():
            for _ in range(eval_trials):
                x, key, val = next(it)
                x = x.to(device); val = val.to(device)
                kvp.reset()
                h = emb(x[:, 0])
                predicted_val = None
                for t in range(eval_seqlen):
                    h = block.forward_step(h, ssw[t, :], kvp)
                    logits = head(h)
                    if t > 0 and x[0, t - 1].item() == 43:
                        pred = logits.argmax(dim=-1)
                        predicted_val = pred[0].item()
                if predicted_val is not None and predicted_val == val.item():
                    correct += 1
        return correct / eval_trials

    sc_accs, sw_accs = [], []
    for dist in eval_distances:
        acc_sc = run_eval_variant(int(dist), True)
        acc_sw = run_eval_variant(int(dist), False)
        sc_accs.append(acc_sc)
        sw_accs.append(acc_sw)
        print(f"[E2] distance={int(dist):5d} | StarCascade acc={acc_sc:.3f} | SlidingWindow acc={acc_sw:.3f}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(eval_distances, sc_accs, marker='o', label='StarCascade')
    plt.plot(eval_distances, sw_accs, marker='s', label='SlidingWindow')
    plt.xlabel('Distance (tokens)')
    plt.ylabel('Retrieval accuracy')
    plt.title('E2 Retrieval accuracy vs distance')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.ylim(0.0, 1.0)
    out_pdf = os.path.join(images_dir, 'accuracy_retrieval_distance.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[E2] Saved {out_pdf}")

    # Confusion-like matrix (mod-16 bins) at largest distance for StarCascade
    last_dist = int(eval_distances[-1])
    bins = 16
    cm = np.zeros((bins, bins), dtype=np.int64)
    class _DS2(torch.utils.data.IterableDataset):
        def __init__(self, seqlen: int, vocab: int, dist: int):
            super().__init__()
            self.S = seqlen
            self.vocab = vocab
            self.dist = dist
        def __iter__(self):
            while True:
                x = torch.randint(0, self.vocab, (1, self.S), dtype=torch.long)
                key = torch.randint(1000, 2000, (1, 1))
                val = torch.randint(2000, 3000, (1, 1))
                qpos = torch.randint(0, max(2, self.S - self.dist - 2), (1,)).item()
                apos = min(self.S - 2, qpos + self.dist)
                x[0, qpos] = 42; x[0, qpos + 1] = key
                x[0, apos] = 43; x[0, apos + 1] = val
                yield x, key.squeeze(-1), val.squeeze(-1)
    it2 = iter(torch.utils.data.DataLoader(_DS2(eval_seqlen, vocab, last_dist), batch_size=None))
    ssw = ssw_sc
    kvp = KVPyramid(kv_dim=128, cfg=kvp_cfg_sc, device=device)
    with torch.no_grad():
        for _ in range(min(200, eval_trials)):
            x, key, val = next(it2)
            x = x.to(device); val = val.to(device)
            kvp.reset(); h = emb(x[:, 0])
            predicted_val = None
            for t in range(eval_seqlen):
                h = block.forward_step(h, ssw[t, :], kvp)
                logits = head(h)
                if t > 0 and x[0, t - 1].item() == 43:
                    pred = logits.argmax(dim=-1)
                    predicted_val = pred[0].item()
            if predicted_val is not None:
                cm[int(val.item()) % bins, predicted_val % bins] += 1
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title('Confusion matrix (mod-16 bins)')
    plt.xlabel('Predicted bin')
    plt.ylabel('True bin')
    plt.colorbar()
    out_pdf = os.path.join(images_dir, 'confusion_matrix_starcascade.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[E2] Saved {out_pdf}")


def activation_stats(x: torch.Tensor) -> Dict[str, float]:
    flat = x.detach().float().view(-1)
    mean = flat.mean().item()
    std = float(flat.std().item() + 1e-8)
    kurt = torch.mean(((flat - mean) / std) ** 4).item()
    maxabs = float(flat.abs().max().item())
    over_6sigma = float((flat.abs() > (6 * std)).float().mean().item())
    return {"kurtosis": kurt, "maxabs": maxabs, "over_6sigma": over_6sigma}


class QuantTensor:
    def __init__(self, x: torch.Tensor, bits: int):
        self.bits = bits
        self.scale = x.abs().amax(dim=(-1,), keepdim=True).clamp(min=1e-8) / (2 ** (bits - 1) - 1)
        q = (x / self.scale).round().clamp(min=-(2 ** (bits - 1)), max= 2 ** (bits - 1) - 1)
        self.q = q.to(torch.int8 if bits <= 8 else torch.int32)
    def dequant(self) -> torch.Tensor:
        return self.q.to(torch.float32) * self.scale


def fake_quant(x: torch.Tensor, bits: int) -> torch.Tensor:
    q = QuantTensor(x, bits)
    return q.dequant()


def exp3_quant_early_exit(config: Dict):
    dirs = config.get("dirs", {})
    images_dir = dirs.get("images_dir", ".research/iteration1/images")
    os.makedirs(images_dir, exist_ok=True)

    device_conf = str(config.get("device", "auto"))
    device = torch.device("cuda" if (device_conf == "auto" and torch.cuda.is_available()) else device_conf)
    set_deterministic(int(config.get("seed", 0)))

    D = int(config.get("cga", {}).get("d_model", 256))
    G = 4
    Cg = 64
    Dkv = 128

    block = CGABlock(CGAConfig(d_model=D, groups=G, c_per_group=Cg, d_kv=Dkv)).to(device)

    ssw_conf = config.get("ssw", {})
    e3 = config.get("e3", {})
    q_seqlen = int(e3.get("q_seqlen", 512))

    ssw = build_star_mask(
        S=q_seqlen,
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
    kvp = KVPyramid(kv_dim=Dkv, cfg=kvp_cfg, device=device)

    gate = EarlyExitGate(d_model=D, groups=G, target_ms=float(e3.get("target_ms", 20.0))).to(device)

    vocab = 8000
    x = torch.randint(0, vocab, (1, q_seqlen), device=device)
    emb_table = torch.randn(vocab, D, device=device)

    # FP pass
    kvp.reset(); h = F.embedding(x[:, 0], emb_table)
    base_stats = []
    device_sync(device); t0 = time.perf_counter()
    for t in range(q_seqlen):
        h = block.forward_step(h, ssw[t, :], kvp)
        base_stats.append(activation_stats(h))
    device_sync(device); t1 = time.perf_counter()
    base_ms_tok = (t1 - t0) * 1000 / q_seqlen

    # W4A8 PTQ pass: fake quantize activations
    def forward_with_quant(h, idx):
        with torch.no_grad():
            hq = fake_quant(h, 8)
            out = block.forward_step(hq, idx, kvp)
            out = fake_quant(out, 8)
        return out

    kvp.reset(); h = F.embedding(x[:, 0], emb_table)
    q_stats = []
    device_sync(device); t0 = time.perf_counter()
    for t in range(q_seqlen):
        h = forward_with_quant(h, ssw[t, :])
        q_stats.append(activation_stats(h))
    device_sync(device); t1 = time.perf_counter()
    q_ms_tok = (t1 - t0) * 1000 / q_seqlen

    def mean_metric(stats, key):
        return sum(s[key] for s in stats) / max(1, len(stats))

    print(f"[E3] FP ms/tok={base_ms_tok:.2f} | W4A8 ms/tok={q_ms_tok:.2f} | kurt FP={mean_metric(base_stats,'kurtosis'):.2f} W4A8={mean_metric(q_stats,'kurtosis'):.2f} | >6σ FP={mean_metric(base_stats,'over_6sigma'):.4f} W4A8={mean_metric(q_stats,'over_6sigma'):.4f}")

    # Activation outlier plot (FP vs W4A8)
    import matplotlib.pyplot as plt
    toks = np.arange(q_seqlen)
    plt.figure()
    plt.plot(toks, [s['kurtosis'] for s in base_stats], label='FP kurtosis')
    plt.plot(toks, [s['kurtosis'] for s in q_stats], label='W4A8 kurtosis')
    plt.xlabel('Token step')
    plt.ylabel('Kurtosis')
    plt.title('Activation kurtosis per token (FP vs W4A8)')
    plt.grid(True)
    plt.legend()
    out_pdf = os.path.join(images_dir, 'activation_outliers_w4a8_vs_fp.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[E3] Saved {out_pdf}")

    # Early-exit sweep: emulate by zeroing queries of later groups
    fracs = [0.25, 0.5, 0.75, 1.0]
    ee_ms = []
    for frac in fracs:
        g_use = max(1, int(round(G * frac)))
        kvp.reset(); h = F.embedding(x[:, 0], emb_table)
        orig_qw = block.q_proj.weight.data.clone()
        device_sync(device); t0 = time.perf_counter()
        for t in range(q_seqlen):
            block.q_proj.weight.data[g_use * Cg:, :] = 0.0
            h = block.forward_step(h, ssw[t, :], kvp)
        device_sync(device); t1 = time.perf_counter()
        block.q_proj.weight.data.copy_(orig_qw)
        ms_tok = (t1 - t0) * 1000 / q_seqlen
        ee_ms.append(ms_tok)
        print(f"[E3] Early-exit groups={g_use}/{G} | {ms_tok:.2f} ms/tok")

    plt.figure()
    plt.plot([int(round(G*f)) for f in fracs], ee_ms, marker='o')
    plt.xlabel('Groups executed')
    plt.ylabel('ms/token')
    plt.title('Early-exit latency vs groups')
    plt.grid(True)
    out_pdf = os.path.join(images_dir, 'inference_latency_early_exit.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"[E3] Saved {out_pdf}")


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument("--ckpt", type=str, default="models/starcascade_e2.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp1_throughput(cfg)
    eval_retrieval(cfg, args.ckpt)
    exp3_quant_early_exit(cfg)
