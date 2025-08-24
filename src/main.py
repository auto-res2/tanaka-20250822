import os
import yaml

from preprocess import preprocess
from train import train_model
from evaluate import exp1_throughput, eval_retrieval, exp3_quant_early_exit


def main():
    # Load config
    cfg_path = os.environ.get("STARCASCADE_CONFIG", "config/experiment.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Ensure dirs exist
    dirs = cfg.get("dirs", {})
    images_dir = dirs.get("images_dir", ".research/iteration1/images")
    data_dir = dirs.get("data_dir", "data")
    models_dir = dirs.get("models_dir", "models")
    for d in [images_dir, data_dir, models_dir]:
        os.makedirs(d, exist_ok=True)

    print("=== StarCascade: Iteration 1 ===")

    # 1) Preprocess
    print("\n[Stage 1/4] Preprocessing...")
    meta = preprocess(cfg)

    # 2) Train
    print("\n[Stage 2/4] Training retrieval model (E2)...")
    ckpt_path = train_model(cfg)

    # 3) Evaluate
    print("\n[Stage 3/4] Evaluating retrieval & plotting (E2, E1, E3)...")
    exp1_throughput(cfg)
    eval_retrieval(cfg, ckpt_path)
    exp3_quant_early_exit(cfg)

    print("\n[Stage 4/4] Completed. Artifacts:")
    print(f" - Images (PDF): {images_dir}")
    print(f" - Models: {ckpt_path}")
    print(f" - Data/meta: {os.path.join(data_dir, 'preprocess_meta.json')}")


if __name__ == "__main__":
    main()
