import json
import sys
from pathlib import Path

import yaml

from src import train_vs_val
from src.dataset import make_split, write_split
from src.inference import save_checkpoint
from src.model import build_model


def test_writes_train_and_val_rows(cfg, tmp_path, monkeypatch):
    train, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    ckpt = tmp_path / "smallcnn_best.pt"
    save_checkpoint(ckpt, build_model("smallcnn", cfg), "smallcnn", cfg, 1, 0.5)

    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path), "--checkpoint", str(ckpt)])
    train_vs_val.main()

    out = json.loads((Path(cfg["output"]["results_dir"]) / "smallcnn_train_vs_val.json").read_text())
    assert out["train"]["n"] == 30 and out["val"]["n"] == 10
    assert set(out["train_minus_val"]) == {"roc_auc", "accuracy", "macro_f1"}
