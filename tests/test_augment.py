import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import load_config
from src.corrupt import TrainAugment, build_train_augment
from src.dataset import build_dataset, make_split, write_split

OPS = {"jpeg": {"quality": [30, 90]}, "gaussian_blur": {"sigma": [0.5, 1.5]}}


@pytest.fixture
def noisy():
    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8))


def test_p0_leaves_images_alone_and_p1_always_corrupts(noisy):
    assert TrainAugment(0.0, OPS, seed=1)(noisy) is noisy
    aug = TrainAugment(1.0, OPS, seed=1)
    for _ in range(5):
        out = aug(noisy)
        assert out.size == (32, 32) and not np.array_equal(np.asarray(out), np.asarray(noisy))


def test_same_seed_same_sequence(noisy):
    a, b = TrainAugment(0.5, OPS, seed=7), TrainAugment(0.5, OPS, seed=7)
    for _ in range(10):
        assert np.array_equal(np.asarray(a(noisy)), np.asarray(b(noisy)))


def test_integer_ranges_draw_integers_and_unknown_ops_fail():
    aug = TrainAugment(1.0, OPS, seed=3)
    assert all(isinstance(aug._draw([30, 90]), int) for _ in range(20))
    assert all(isinstance(aug._draw([0.5, 1.5]), float) for _ in range(20))
    with pytest.raises(ValueError, match="Unknown augmentation"):
        TrainAugment(0.5, {"snow": {}}, seed=0)


def test_experiment_config_changes_only_training_augmentation():
    base, aug = load_config("configs/default.yaml"), load_config("configs/smallcnn_aug.yaml")
    assert build_train_augment(base) is None and build_train_augment(aug) is not None
    assert "downsample_upsample" not in aug["train"]["augment"]["ops"]          # held-out corruption
    base_train = {k: v for k, v in base["train"].items()}
    aug_train = {k: v for k, v in aug["train"].items() if k not in ("augment", "run_name")}
    assert aug_train == base_train                                                 # same optimizer, lr, epochs...
    assert {k: v for k, v in aug.items() if k != "train"} == {k: v for k, v in base.items() if k != "train"}


def test_augmentation_refused_outside_training(cfg):
    with pytest.raises(ValueError, match="training only"):
        build_dataset(cfg, "val", augment=TrainAugment(1.0, OPS, seed=0))


def test_augmented_run_trains_and_evaluates_under_its_own_name(cfg, tmp_path, monkeypatch):
    from src import evaluate, train
    train_paths, val_paths, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train_paths, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val_paths, Path(cfg["data"]["splits_dir"]) / "val.txt")
    cfg["train"].update(epochs=1, run_name="smallcnn_aug", augment={"p": 1.0, "ops": OPS})
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    results = Path(cfg["output"]["results_dir"])

    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path)])
    train.main()
    assert (results / "smallcnn_aug_best.pt").exists() and not (results / "smallcnn_best.pt").exists()

    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path), "--checkpoint",
                                      str(results / "smallcnn_aug_best.pt"), "--corrupt", "blur"])
    evaluate.main()
    out = json.loads((results / "eval_smallcnn_aug_val_blur.json").read_text())
    assert out["model"] == "smallcnn_aug" and out["architecture"] == "smallcnn"
