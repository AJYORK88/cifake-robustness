"""Fixtures: a tiny synthetic CIFAKE-shaped folder so tests run without the real dataset."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from PIL import Image

from src.config import load_config


def _write_images(folder, n, seed):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(folder / f"{i} ({i % 3}).jpg", quality=90)  # CIFAKE-style names with spaces


@pytest.fixture
def fake_raw(tmp_path):
    """data/raw with train/{REAL,FAKE} x 20 and test/{REAL,FAKE} x 6."""
    raw = tmp_path / "raw"
    for s, (pool, cls, n) in enumerate([("train", "REAL", 20), ("train", "FAKE", 20),
                                         ("test", "REAL", 6), ("test", "FAKE", 6)]):
        _write_images(raw / pool / cls, n, seed=s)
    return raw


@pytest.fixture
def cfg(fake_raw, tmp_path):
    c = copy.deepcopy(load_config("configs/default.yaml"))
    c["data"]["raw_dir"] = str(fake_raw)
    c["data"]["splits_dir"] = str(tmp_path / "splits")
    c["data"]["val_per_class"] = 5
    c["output"]["results_dir"] = str(tmp_path / "results")
    return c
