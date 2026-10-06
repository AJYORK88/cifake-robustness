"""Checkpoint format and batched scoring, shared by train.py (val AUC) and evaluate.py.

Checkpoint contract -- train.py must save exactly this dict with torch.save:
    {
        "model_name": "smallcnn" | "resnet18",
        "state_dict": model.state_dict(),
        "config":     cfg,          # the full config dict used for the run
        "epoch":      int,          # epoch of the best val AUC
        "val_auc":    float,
    }
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import build_model


def save_checkpoint(path: str | Path, model: nn.Module, model_name: str, cfg: dict,
                    epoch: int, val_auc: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "config": cfg,
                "epoch": epoch, "val_auc": val_auc}, path)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the model from the config stored IN the checkpoint, load weights, set eval()."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device,
                   desc: str = "predict") -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, P(FAKE)) over a loader. Puts the model in eval() mode."""
    model.eval()
    ys, scores = [], []
    for x, y in tqdm(loader, desc=desc, leave=False):
        logits = model(x.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(scores)
