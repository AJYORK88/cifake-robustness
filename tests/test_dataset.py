import shutil
from pathlib import Path

import pytest
import torch

from src.dataset import (build_dataset, find_class_dirs, label_from_path, make_split,
                         read_split, write_split)


def test_split_is_stratified_disjoint_and_train_pool_only(fake_raw):
    train, val, _ = make_split(fake_raw, val_per_class=5, seed=42)
    assert len(val) == 10 and len(train) == 30
    assert sum(label_from_path(p) for p in val) == 5               # 5 FAKE, 5 REAL
    assert not set(train) & set(val)
    assert all(p.startswith("train/") for p in train + val)        # never the test pool


def test_split_is_deterministic_and_seed_sensitive(fake_raw):
    assert make_split(fake_raw, 5, 42) == make_split(fake_raw, 5, 42)
    assert make_split(fake_raw, 5, 42)[1] != make_split(fake_raw, 5, 7)[1]


def test_split_paths_are_relative_posix(fake_raw, tmp_path):
    train, val, _ = make_split(fake_raw, 5, 42)
    assert all("\\" not in p and not p.startswith("/") and ":" not in p for p in train + val)
    write_split(val, tmp_path / "val.txt")
    assert read_split(tmp_path / "val.txt") == val                 # survives spaces in filenames


def test_layout_detection_handles_nested_and_case(tmp_path, fake_raw):
    nested = tmp_path / "raw2"
    shutil.copytree(fake_raw, nested / "cifake")
    (nested / "cifake" / "train" / "REAL").rename(nested / "cifake" / "train" / "real")
    dirs = find_class_dirs(nested)
    assert dirs[("train", "REAL")].name == "real"
    # Split paths are relative to the folder holding train/ and test/, so a nested unzip
    # produces the same split files as a flat one.
    assert make_split(nested, 5, 42)[1] == [p.replace("/REAL/", "/real/") for p in make_split(fake_raw, 5, 42)[1]]


def test_split_files_from_flat_unzip_load_from_nested_unzip(cfg, tmp_path):
    train, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    nested = tmp_path / "raw_nested"
    shutil.copytree(cfg["data"]["raw_dir"], nested / "archive")      # e.g. Windows "Extract All"
    cfg["data"]["raw_dir"] = str(nested)
    x, _ = build_dataset(cfg, "val")[0]
    assert x.shape == (3, 32, 32)


def test_missing_images_fail_fast_with_a_clear_message(cfg):
    write_split(["train/REAL/not-there.jpg"], Path(cfg["data"]["splits_dir"]) / "val.txt")
    with pytest.raises(FileNotFoundError, match="src.dataset --check"):
        build_dataset(cfg, "val")


def test_dataset_returns_unit_range_tensors(cfg):
    train, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    x, y = build_dataset(cfg, "val")[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0 and y in (0, 1)
    assert len(build_dataset(cfg, "test")) == 12


def test_training_split_refuses_corruption(cfg):
    with pytest.raises(ValueError, match="test-time only"):
        build_dataset(cfg, "train", corrupt=lambda im: im)


def test_train_copies_of_val_and_test_images_are_removed(fake_raw):
    _, val_before, _ = make_split(fake_raw, 5, 42, drop_eval_duplicates=False)
    train_before, _, _ = make_split(fake_raw, 5, 42, drop_eval_duplicates=False)
    # Plant byte-identical copies: one train image duplicates a test image, another duplicates a val image.
    test_img = sorted((fake_raw / "test" / "FAKE").iterdir())[0]
    shutil.copy(test_img, fake_raw / train_before[0])
    shutil.copy(fake_raw / val_before[0], fake_raw / train_before[1])

    train, val, removed = make_split(fake_raw, 5, 42)
    assert val == val_before                                        # val is unaffected by dedup
    assert removed == sorted([train_before[0], train_before[1]])
    assert not set(removed) & set(train) and len(train) == len(train_before) - 2
