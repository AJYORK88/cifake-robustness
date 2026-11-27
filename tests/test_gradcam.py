import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src import gradcam
from src.dataset import make_split
from src.inference import save_checkpoint
from src.model import build_model


def test_heatmap_shape_range_and_model_unchanged(cfg):
    torch.manual_seed(0)
    model = build_model("smallcnn", cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cam, p = gradcam.GradCAM(model, gradcam.last_conv_activation(model))(torch.rand(1, 3, 32, 32))
    assert cam.shape == (32, 32) and 0.0 <= cam.min() and cam.max() <= 1.0 and 0.0 <= p <= 1.0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())   # explaining never trains


def test_end_to_end_writes_heatmaps_for_both_models(cfg, tmp_path, monkeypatch):
    results = Path(cfg["output"]["results_dir"])
    results.mkdir(parents=True)
    for name in gradcam.MODELS:
        save_checkpoint(results / f"{name}_best.pt", build_model("smallcnn", cfg), "smallcnn", cfg, 1, 0.5, run_name=name)
    _, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    fake = [p for p in val if "/FAKE/" in p]
    real = [p for p in val if "/REAL/" in p]
    record = lambda p, lab: {"path": p, "label": lab, "p_fake_clean": 0.9, "p_fake_corrupted": 0.1}
    groups = {"broken_by_corruption": [record(p, "FAKE") for p in fake[:2]],
              "correct_real": [record(real[0], "REAL")], "false_positive_real": []}
    (results / "failure_cases_smallcnn_blur.json").write_text(json.dumps({"groups": groups}))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path)])
    gradcam.main()

    data = np.load(results / "gradcam_blur_fake.npz")
    assert data["images"].shape == (3, 2, 32, 32, 3) and data["cams"].shape == (2, 3, 2, 32, 32)
    assert list(data["groups"]) == ["broken_by_corruption", "broken_by_corruption", "correct_real"]
    summary = json.loads((results / "gradcam_blur_fake.json").read_text())
    assert set(summary["models"]) == {"smallcnn", "smallcnn_aug"}
