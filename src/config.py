"""Config loading, seeding, and device selection shared by every script."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load a YAML config. Relative paths resolve against the repo root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def repo_path(rel: str | Path) -> Path:
    """Resolve a config path (e.g. cfg['data']['raw_dir']) against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) and request deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
