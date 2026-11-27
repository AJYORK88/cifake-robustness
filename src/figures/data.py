"""Read saved results for the figure scripts. No plotting here.

Every number in a figure should come through these helpers, so it traces back to a file in results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import load_config, repo_path
from src.dataset import data_root

MODELS = ("baseline", "smallcnn", "resnet18", "smallcnn_aug")
CONDITIONS = ("none", "jpeg70", "jpeg40", "blur", "resample")   # test-time corruptions, mildest first

RESULTS = repo_path("results")
FIGURES = RESULTS / "figures"


def load_json(name: str) -> dict:
    """Load results/<name>; raises FileNotFoundError if that experiment hasn't been run yet."""
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiment that writes it first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def eval_metrics(model: str, corrupt: str = "none", split: str = "val") -> dict:
    """Metrics dict (roc_auc, accuracy, macro_f1, fpr_real, miss_rate_fake, n, ...) for one evaluation."""
    return load_json(f"eval_{model}_{split}_{corrupt}.json")["metrics"]


def history(run: str) -> list[dict]:
    """Per-epoch training log: list of {"epoch", "train_loss", "val_auc", ...}."""
    return load_json(f"{run}_history.json")["epochs"]


def failure_cases(run: str, corrupt: str = "blur") -> dict:
    """Selected examples from src.failure_cases: {"counts": ..., "groups": {name: [{"path", ...}]}}."""
    return load_json(f"failure_cases_{run}_{corrupt}.json")


def gradcam(corrupt: str = "blur", target: str = "fake") -> dict:
    """Grad-CAM arrays saved by src.gradcam (see that module's docstring for the shapes)."""
    path = RESULTS / f"gradcam_{corrupt}_{target}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the experiment that writes it first")
    with np.load(path) as arrays:
        return {k: arrays[k] for k in arrays.files}


def image_path(rel_path: str) -> Path:
    """Absolute path of a dataset image from a split file or a failure_cases record."""
    return data_root(repo_path(load_config()["data"]["raw_dir"])) / rel_path


def figure_path(name: str) -> Path:
    """Where to save a figure: results/figures/<name> (folder created if needed)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name
