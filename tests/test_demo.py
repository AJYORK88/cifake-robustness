from pathlib import Path

import numpy as np
from PIL import Image

from src.demo import example_rows, predict
from src.dataset import make_split, write_split
from src.model import build_model


def test_predict_returns_a_row_per_model_for_any_image_size(cfg):
    cfg["model"]["resnet18"].update(weights=None, input_size=64)
    models = {"SmallCNN": build_model("smallcnn", cfg).eval(), "Frozen ResNet18": build_model("resnet18", cfg).eval()}
    big = Image.fromarray(np.random.default_rng(0).integers(0, 256, (100, 80, 3), dtype=np.uint8))
    shown, rows = predict(models, cfg, "cpu", big, "blur")
    assert shown.size == (256, 256)                       # 32x32 enlarged 8x
    assert [r[0] for r in rows] == ["SmallCNN", "Frozen ResNet18"]
    assert all(0.0 <= r[1] <= 1.0 and r[2] in ("REAL", "FAKE") for r in rows)


def test_examples_fall_back_to_validation_images(cfg):
    train, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    rows = example_rows(cfg)
    assert len(rows) == 2 and all(Path(p).exists() and "train" in Path(p).parts for p, _ in rows)
