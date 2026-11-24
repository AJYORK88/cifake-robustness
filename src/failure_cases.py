"""Select representative successes and failures for qualitative analysis. This file does no plotting.

    python -m src.failure_cases --checkpoint results/smallcnn_best.pt --corrupt blur
    python -m src.failure_cases --checkpoint results/smallcnn_aug_best.pt --corrupt blur

Scores the VALIDATION images (never test) and writes results/failure_cases_<run>_<corrupt>.json. For each
group it lists the k most confident examples: image path, true label, P(FAKE) on the clean image, and
P(FAKE) on the corrupted image. "counts" gives the full size of every group.

Groups:
  correct_real          real images predicted REAL, most confident first (lowest P(FAKE))
  correct_fake          fake images predicted FAKE, most confident first (highest P(FAKE))
  false_positive_real   real images predicted FAKE, most confident first (highest P(FAKE))
  missed_fake           fake images predicted REAL, most confident first (lowest P(FAKE))
  broken_by_corruption  fakes caught on the clean image but missed after the corruption,
                        largest drop in P(FAKE) first (only when --corrupt is not "none")

Image paths are relative to the dataset folder: data_root(repo_path(cfg["data"]["raw_dir"])) / path.
The figures are made by the team's own plotting code from this file.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config import get_device, load_config, repo_path, set_seed
from src.corrupt import corruption_names, get_corruption
from src.dataset import build_loader
from src.inference import load_checkpoint, predict_scores
from src.metrics import save_json


def select_groups(y: np.ndarray, clean: np.ndarray, corrupted: np.ndarray | None,
                  threshold: float, k: int) -> tuple[dict, dict]:
    """Return ({group: indices of its k most confident examples}, {group: full group size})."""
    pred_fake = clean >= threshold
    real, fake = y == 0, y == 1

    def top(mask: np.ndarray, priority: np.ndarray) -> np.ndarray:
        idx = np.flatnonzero(mask)                           # positions where the mask is True
        return idx[np.argsort(-priority[idx], kind="stable")][:k]   # highest priority first

    masks = {
        "correct_real": (real & ~pred_fake, -clean),
        "correct_fake": (fake & pred_fake, clean),
        "false_positive_real": (real & pred_fake, clean),
        "missed_fake": (fake & ~pred_fake, -clean),
    }
    if corrupted is not None:
        masks["broken_by_corruption"] = (fake & pred_fake & (corrupted < threshold), clean - corrupted)
    groups = {name: top(mask, priority) for name, (mask, priority) in masks.items()}
    counts = {name: int(mask.sum()) for name, (mask, _) in masks.items()}
    return groups, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default="results/smallcnn_best.pt")
    ap.add_argument("--corrupt", default="blur")
    ap.add_argument("--k", type=int, default=8, help="examples kept per group")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.corrupt not in corruption_names(cfg):
        ap.error(f"--corrupt must be one of {corruption_names(cfg)}")
    set_seed(cfg["seed"])
    device = get_device()
    threshold = cfg["eval"]["threshold"]
    model, ckpt = load_checkpoint(repo_path(args.checkpoint), device)
    run = ckpt.get("run_name", ckpt["model_name"])

    clean_loader = build_loader(cfg, "val", shuffle=False)      # shuffle=False: score i belongs to path i
    y, clean = predict_scores(model, clean_loader, device, desc=f"[model={run} split=val corrupt=none]")
    paths = clean_loader.dataset.rel_paths
    corrupted = None
    if args.corrupt != "none":
        loader = build_loader(cfg, "val", corrupt=get_corruption(args.corrupt, cfg), shuffle=False)
        _, corrupted = predict_scores(model, loader, device, desc=f"[model={run} split=val corrupt={args.corrupt}]")

    groups, counts = select_groups(y, clean, corrupted, threshold, args.k)
    record = {
        "model": run, "checkpoint": args.checkpoint, "split": "val", "corrupt": args.corrupt,
        "threshold": threshold, "k": args.k, "counts": counts,
        "groups": {
            name: [{"path": paths[i], "label": "FAKE" if y[i] else "REAL",
                    "p_fake_clean": float(clean[i]),
                    "p_fake_corrupted": None if corrupted is None else float(corrupted[i])}
                   for i in idx]
            for name, idx in groups.items()
        },
    }
    out = repo_path(cfg["output"]["results_dir"]) / f"failure_cases_{run}_{args.corrupt}.json"
    save_json(record, out)
    print(f"[model={run} split=val corrupt={args.corrupt}] group sizes: {counts}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()