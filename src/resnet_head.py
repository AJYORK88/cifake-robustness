"""Frozen ResNet18 trained on cached backbone features.

    python -m src.resnet_head                # extract features (once), fit the head, score val under every corruption
    python -m src.resnet_head --split test   # FINAL REPORTING ONLY: score the saved head on the official test set

Why this exists: FrozenResNet18's backbone never changes, so its 512-number output for an image is the same
every epoch. Training through train.py recomputes it for every image every epoch (~100 min per epoch on our
CPU). Here each image goes through the backbone ONCE; the features are saved, and only the 513-parameter
linear head is trained on them. The resulting model is identical; this file checks that on real images.

Head training mirrors train.py exactly: Adam, learning rate, batch size, max epochs, BCE-with-logits loss,
and early stopping on validation ROC-AUC, all read from the config.

Outputs (in results/):
    features/resnet18_<split>_<corrupt>_<size>px.npz   cached features (gitignored: ~180 MB for train)
    resnet18_best.pt                                   standard checkpoint: works with evaluate.py too (gitignored)
    resnet18_config.yaml, resnet18_history.json        run config and per-epoch log of the head
    eval_resnet18_<split>_<corrupt>.json               same format as evaluate.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from src.config import get_device, load_config, repo_path, set_seed
from src.corrupt import corruption_names, get_corruption
from src.dataset import build_loader, split_paths
from src.inference import load_checkpoint, save_checkpoint
from src.metrics import compute_metrics, format_metrics, save_json
from src.model import build_model

MODEL_NAME = "resnet18"


def feature_path(cfg: dict, split: str, corrupt: str):
    size = cfg["model"]["resnet18"]["input_size"]
    return repo_path(cfg["output"]["results_dir"]) / "features" / f"resnet18_{split}_{corrupt}_{size}px.npz"


@torch.no_grad()
def extract_features(model: nn.Module, cfg: dict, split: str, corrupt: str, device) -> tuple[np.ndarray, np.ndarray]:
    """512-d backbone features and labels for every image in a split, cached on disk.

    The cache stores the image paths it was built from; if the split files change, it is rebuilt.
    """
    path = feature_path(cfg, split, corrupt)
    paths = split_paths(cfg, split)
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if cached["paths"].tolist() == paths:
            return cached["features"], cached["labels"]
        print(f"[resnet18] {path.name} was built from a different split; re-extracting")

    loader = build_loader(cfg, split, corrupt=get_corruption(corrupt, cfg), shuffle=False)
    head = model.resnet.fc
    model.resnet.fc = nn.Identity()  # with the head removed, FrozenResNet18.forward returns the 512-d features
    model.eval()
    feats, labels = [], []
    try:
        for x, y in tqdm(loader, desc=f"[resnet18 features split={split} corrupt={corrupt}]"):
            feats.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    finally:
        model.resnet.fc = head
    features = np.concatenate(feats).astype(np.float32)
    label_arr = np.concatenate(labels).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=label_arr, paths=np.array(paths))
    return features, label_arr


def fit_head(cfg: dict, X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray):
    """Train a linear head (512 -> 1 logit) on cached features, with train.py's settings.

    Returns (head with the best-val-AUC weights, best epoch, best val AUC, per-epoch history).
    """
    torch.manual_seed(cfg["seed"])
    head = nn.Linear(X_tr.shape[1], 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=float(cfg["train"]["lr"]))
    criterion = nn.BCEWithLogitsLoss()
    X_train, y_train = torch.from_numpy(X_tr), torch.from_numpy(y_tr).float()
    X_val = torch.from_numpy(X_va)
    shuffler = torch.Generator().manual_seed(cfg["seed"])
    batch_size = int(cfg["train"]["batch_size"])
    patience = int(cfg["train"]["early_stopping"]["patience"])

    best_auc, best_epoch, best_state = float("-inf"), 0, None
    history, epochs_without_improvement = [], 0
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        head.train()
        total_loss = 0.0
        for idx in torch.randperm(len(X_train), generator=shuffler).split(batch_size):
            optimizer.zero_grad()
            loss = criterion(head(X_train[idx]).squeeze(1), y_train[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(head(X_val).squeeze(1)).numpy()
        val = compute_metrics(y_va, val_scores, cfg["eval"]["threshold"])
        history.append({"epoch": epoch, "train_loss": total_loss / len(X_train),
                        "val_auc": val["roc_auc"], "val_accuracy": val["accuracy"]})
        print(f"[resnet18 head epoch {epoch:02d}] train_loss={total_loss / len(X_train):.6f} "
              f"val_auc={val['roc_auc']:.6f}")
        if val["roc_auc"] > best_auc:
            best_auc, best_epoch, epochs_without_improvement = val["roc_auc"], epoch, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch} with val_auc={best_auc:.6f}")
            break
    head.load_state_dict(best_state)
    return head, best_epoch, best_auc, history


@torch.no_grad()
def check_matches_full_model(model: nn.Module, cfg: dict, device, n: int = 256) -> float:
    """Score the first n val images through the FULL model (as evaluate.py does) and through the cached
    features; they must agree. Returns the largest difference in P(FAKE)."""
    model.eval()
    x, _ = next(iter(build_loader(cfg, "val", shuffle=False, batch_size=n)))
    full = torch.sigmoid(model(x.to(device))).cpu().numpy()
    X_va, _ = extract_features(model, cfg, "val", "none", device)
    cached = torch.sigmoid(model.resnet.fc(torch.from_numpy(X_va[: len(full)]).to(device)).squeeze(1)).cpu().numpy()
    diff = float(np.abs(full - cached).max())
    if diff > 1e-4:
        raise RuntimeError(f"Cached-feature scores differ from the full model by {diff:.2e}")
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    results_dir = repo_path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / f"{MODEL_NAME}_best.pt"

    if args.split == "val":
        model = build_model(MODEL_NAME, cfg).to(device)  # first use downloads the ImageNet weights (~45 MB)
        X_tr, y_tr = extract_features(model, cfg, "train", "none", device)
        X_va, y_va = extract_features(model, cfg, "val", "none", device)
        head, epoch, val_auc, history = fit_head(cfg, X_tr, y_tr, X_va, y_va)
        model.resnet.fc.load_state_dict(head.state_dict())
        save_checkpoint(checkpoint_path, model, MODEL_NAME, cfg, epoch, val_auc)
        with open(results_dir / f"{MODEL_NAME}_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        save_json({"model": MODEL_NAME, "trained_on": "cached backbone features", "epochs": history},
                  results_dir / f"{MODEL_NAME}_history.json")
        diff = check_matches_full_model(model, cfg, device)
        print(f"[resnet18] cached features match the full model (max difference {diff:.1e})")
    else:
        print("[resnet18 split=test] WARNING: evaluating on the OFFICIAL TEST SET. Final reporting only.")
        with open(results_dir / "test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}\t{checkpoint_path.name}\tall corruptions\n")
        model, ckpt = load_checkpoint(checkpoint_path, device)
        epoch = ckpt["epoch"]

    for corrupt in corruption_names(cfg):
        X, y = extract_features(model, cfg, args.split, corrupt, device)
        with torch.no_grad():
            scores = torch.sigmoid(model.resnet.fc(torch.from_numpy(X).to(device)).squeeze(1)).cpu().numpy()
        metrics = compute_metrics(y, scores, cfg["eval"]["threshold"])
        print(format_metrics(metrics, args.split, corrupt, MODEL_NAME))
        save_json({"model": MODEL_NAME, "split": args.split, "corrupt": corrupt,
                   "corrupt_params": cfg["corruptions"].get(corrupt), "checkpoint": f"results/{checkpoint_path.name}",
                   "checkpoint_epoch": epoch, "scored_from": "cached backbone features", "metrics": metrics},
                  results_dir / f"eval_{MODEL_NAME}_{args.split}_{corrupt}.json")


if __name__ == "__main__":
    main()
