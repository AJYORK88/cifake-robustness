"""Evaluate a saved checkpoint on validation (default) or, once, on the official test set.

    python -m src.evaluate --checkpoint results/smallcnn_best.pt
    python -m src.evaluate --checkpoint results/smallcnn_best.pt --corrupt jpeg40
    python -m src.evaluate --checkpoint results/smallcnn_best.pt --split test --corrupt none

Writes results/eval_<model>_<split>_<corrupt>.json. Every printed line names split and corruption.

THE OFFICIAL TEST SET IS FOR FINAL REPORTING ONLY. Select models, thresholds, and
hyper-parameters on validation. Each test run is appended to results/test_runs.log so the
team can show how many times it was touched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.config import get_device, load_config, repo_path, set_seed
from src.corrupt import corruption_names, get_corruption
from src.dataset import build_loader
from src.inference import load_checkpoint, predict_scores
from src.metrics import compute_metrics, format_metrics, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml",
                    help="used for data paths, threshold, and corruption params (model config comes from the checkpoint)")
    ap.add_argument("--checkpoint", default="results/smallcnn_best.pt")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--corrupt", default="none")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.corrupt not in corruption_names(cfg):
        ap.error(f"--corrupt must be one of {corruption_names(cfg)}")
    set_seed(cfg["seed"])
    tag = f"[split={args.split} corrupt={args.corrupt}]"

    results_dir = repo_path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        print(f"{tag} WARNING: evaluating on the OFFICIAL TEST SET. Final reporting only.")
        with open(results_dir / "test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}\t{args.checkpoint}\t{args.corrupt}\n")

    device = get_device()
    model, ckpt = load_checkpoint(repo_path(args.checkpoint), device)
    model_name = ckpt["model_name"]
    tag = f"[model={model_name} split={args.split} corrupt={args.corrupt}]"
    print(f"{tag} checkpoint={args.checkpoint} (epoch {ckpt['epoch']}, val_auc {ckpt['val_auc']:.4f}) device={device}")

    loader = build_loader(cfg, args.split, corrupt=get_corruption(args.corrupt, cfg), shuffle=False)
    y_true, y_score = predict_scores(model, loader, device, desc=tag)
    metrics = compute_metrics(y_true, y_score, cfg["eval"]["threshold"])
    print(format_metrics(metrics, args.split, args.corrupt, model_name))

    out = results_dir / f"eval_{model_name}_{args.split}_{args.corrupt}.json"
    save_json({"model": model_name, "split": args.split, "corrupt": args.corrupt,
               "corrupt_params": cfg["corruptions"].get(args.corrupt), "checkpoint": args.checkpoint,
               "checkpoint_epoch": ckpt["epoch"], "metrics": metrics}, out)
    print(f"{tag} wrote {out}")


if __name__ == "__main__":
    main()
