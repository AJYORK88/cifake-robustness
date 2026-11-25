"""Config loading, seeding, and device selection shared by every script."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Override's values win; nested dicts are merged key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load a YAML config. Relative paths resolve against the repo root.

    A config may start with `base: <other config>`; it then contains only what differs from that base
    (used by experiment configs such as configs/smallcnn_aug.yaml).
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        cfg = _deep_merge(load_config(cfg.pop("base")), cfg)
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
