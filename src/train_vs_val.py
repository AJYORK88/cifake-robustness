"""Generalization check: score a trained checkpoint on its own TRAIN split and on VAL.

    python -m src.train_vs_val --checkpoint results/smallcnn_best.pt

A memorizing model scores near-perfectly on train and much worse on val; a small train-val
gap means the model learned patterns that transfer to unseen images. Never touches test.

Output: results/<model>_train_vs_val.json, plus a printed table.
"""

from __future__ import annotations

import argparse

from src.config import get_device, load_config, repo_path, set_seed
from src.dataset import build_loader
from src.inference import load_checkpoint, predict_scores
from src.metrics import compute_metrics, save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="results/smallcnn_best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    model, ckpt = load_checkpoint(repo_path(args.checkpoint), device)
    model_name = ckpt["model_name"]

    rows = {}
    for split in ("train", "val"):
        loader = build_loader(cfg, split, shuffle=False)       # no shuffle, no corruption
        y_true, y_score = predict_scores(model, loader, device, desc=f"[model={model_name} split={split}]")
        rows[split] = compute_metrics(y_true, y_score, cfg["eval"]["threshold"])

    gap = {k: rows["train"][k] - rows["val"][k] for k in ("roc_auc", "accuracy", "macro_f1")}
    out = repo_path(cfg["output"]["results_dir"]) / f"{model_name}_train_vs_val.json"
    save_json({"model": model_name, "checkpoint": args.checkpoint, "checkpoint_epoch": ckpt["epoch"],
               "corrupt": "none", "train": rows["train"], "val": rows["val"], "train_minus_val": gap}, out)

    print(f"\n[model={model_name} corrupt=none] checkpoint epoch {ckpt['epoch']}")
    print(f"{'split':<6} {'n':>6} {'ROC-AUC':>8} {'acc':>7} {'FPR_real':>9} {'miss_fake':>10}")
    for split, m in rows.items():
        print(f"{split:<6} {m['n']:>6} {m['roc_auc']:>8.4f} {m['accuracy']:>7.4f} "
              f"{m['fpr_real']:>9.4f} {m['miss_rate_fake']:>10.4f}")
    print(f"{'gap':<6} {'':>6} {gap['roc_auc']:>8.4f} {gap['accuracy']:>7.4f}   (train minus val)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
