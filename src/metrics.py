"""Shared metric definitions so the baseline and every CNN are scored identically.

Convention: label REAL = 0, FAKE = 1 (positive class). Scores are P(FAKE).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def fpr_on_real(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False-positive rate on real images: P(pred = FAKE | true = REAL)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    real = y_true == 0
    if not real.any():
        return float("nan")
    return float((y_pred[real] == 1).mean())


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """ROC-AUC, accuracy, macro-F1, and real-image FPR.

    y_score is P(FAKE) in [0, 1]; predictions use score >= threshold -> FAKE.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "roc_auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "fpr_real": fpr_on_real(y_true, y_pred),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }


def format_metrics(metrics: dict, split: str, corrupt: str, model: str = "") -> str:
    """One-line summary that always names the split and corruption."""
    prefix = f"[model={model} split={split} corrupt={corrupt}]" if model else f"[split={split} corrupt={corrupt}]"
    return (
        f"{prefix} AUC={metrics['roc_auc']:.4f} acc={metrics['accuracy']:.4f} "
        f"macroF1={metrics['macro_f1']:.4f} FPR_real={metrics['fpr_real']:.4f} n={metrics['n']}"
    )


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
