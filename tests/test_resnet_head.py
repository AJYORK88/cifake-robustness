import json
import sys
from pathlib import Path

import yaml

from src import resnet_head
from src.dataset import make_split, write_split
from src.inference import load_checkpoint


def test_cached_head_pipeline_end_to_end(cfg, tmp_path, monkeypatch):
    train, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    cfg["model"]["resnet18"].update(weights=None, input_size=64)   # random init + small input: fast, no download
    cfg["train"]["epochs"] = 2
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    results = Path(cfg["output"]["results_dir"])

    # Validation run: extracts features, fits the head, checks cached == full model, scores every corruption.
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path)])
    resnet_head.main()
    model, ckpt = load_checkpoint(results / "resnet18_best.pt", "cpu")
    assert ckpt["model_name"] == "resnet18"
    for c in ("none", "jpeg40", "jpeg70", "blur", "resample"):
        assert json.loads((results / f"eval_resnet18_val_{c}.json").read_text())["metrics"]["n"] == 10

    # Test run reuses the saved head and logs that the test set was used.
    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path), "--split", "test"])
    resnet_head.main()
    assert json.loads((results / "eval_resnet18_test_none.json").read_text())["metrics"]["n"] == 12
    assert "resnet18_best.pt" in (results / "test_runs.log").read_text()
