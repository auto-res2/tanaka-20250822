import os
import json
from typing import Dict

import torch

# Reuse build_star_mask and SSWConfig from train.py to avoid duplication
from train import build_star_mask, SSWConfig


def ensure_dirs(config: Dict) -> Dict[str, str]:
    dirs = {
        "images": config.get("dirs", {}).get("images_dir", ".research/iteration1/images"),
        "data": config.get("dirs", {}).get("data_dir", "data"),
        "models": config.get("dirs", {}).get("models_dir", "models"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    # also create subfolders
    os.makedirs(os.path.join(dirs["data"], "masks"), exist_ok=True)
    return dirs


def preprocess(config: Dict) -> Dict:
    dirs = ensure_dirs(config)

    # Determine sequence lengths we will precompute masks for
    e1 = config.get("e1", {})
    e2 = config.get("e2", {})
    e3 = config.get("e3", {})
    lengths = set()
    for k in [e1.get("seqlens", []), [e2.get("train_seqlen", 512)], [e2.get("eval_seqlen", 1024)], [e3.get("q_seqlen", 512)]]:
        for s in k:
            try:
                lengths.add(int(s))
            except Exception:
                pass

    ssw_conf = config.get("ssw", {})
    ssw_cfg = SSWConfig(
        window=int(ssw_conf.get("window", 512)),
        strides=list(ssw_conf.get("strides", [97, 223])),
        star_hubs_per_seg=int(ssw_conf.get("star_hubs_per_seg", 4)),
        segment_len=int(ssw_conf.get("segment_len", 256)),
        pad_to=int(ssw_conf.get("pad_to", 64)),
    )

    mask_paths = {}
    for S in sorted(lengths):
        mask = build_star_mask(S=S, cfg=ssw_cfg)
        out_path = os.path.join(dirs["data"], "masks", f"star_mask_S{S}_pad{ssw_cfg.pad_to}.pt")
        torch.save({"mask": mask, "S": S, "ssw_cfg": ssw_conf}, out_path)
        mask_paths[str(S)] = out_path
        print(f"[Preprocess] Saved mask for S={S} to {out_path}")

    meta = {
        "masks": mask_paths,
        "config": config,
    }
    meta_path = os.path.join(dirs["data"], "preprocess_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Preprocess] Wrote meta to {meta_path}")

    return meta


if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    preprocess(cfg)
