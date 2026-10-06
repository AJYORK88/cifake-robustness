import shutil
from pathlib import Path

import pytest
import torch

from src.dataset import (build_dataset, find_class_dirs, label_from_path, make_split,
                         read_split, write_split)


def test_split_is_stratified_disjoint_and_train_pool_only(fake_raw):
    train, val = make_split(fake_raw, val_per_class=5, seed=42)
    assert len(val) == 10 and len(train) == 30
    assert sum(label_from_path(p) for p in val) == 5               # 5 FAKE, 5 REAL
    assert not set(train) & set(val)
    assert all(p.startswith("train/") for p in train + val)        # never the test pool


def test_split_is_deterministic_and_seed_sensitive(fake_raw):
    assert make_split(fake_raw, 5, 42) == make_split(fake_raw, 5, 42)
    assert make_split(fake_raw, 5, 42)[1] != make_split(fake_raw, 5, 7)[1]


def test_split_paths_are_relative_posix(fake_raw, tmp_path):
    train, val = make_split(fake_raw, 5, 42)
    assert all("\\" not in p and not p.startswith("/") and ":" not in p for p in train + val)
    write_split(val, tmp_path / "val.txt")
    assert read_split(tmp_path / "val.txt") == val                 # survives spaces in filenames


def test_layout_detection_handles_nested_and_case(tmp_path, fake_raw):
    nested = tmp_path / "raw2"
    shutil.copytree(fake_raw, nested / "cifake")
    (nested / "cifake" / "train" / "REAL").rename(nested / "cifake" / "train" / "real")
    dirs = find_class_dirs(nested)
    assert dirs[("train", "REAL")].name == "real"
    assert make_split(nested, 5, 42)[1][0].startswith("cifake/train/")


def test_dataset_returns_unit_range_tensors(cfg):
    train, val = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    x, y = build_dataset(cfg, "val")[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0 and y in (0, 1)
    assert len(build_dataset(cfg, "test")) == 12


def test_training_split_refuses_corruption(cfg):
    with pytest.raises(ValueError, match="test-time only"):
        build_dataset(cfg, "train", corrupt=lambda im: im)
