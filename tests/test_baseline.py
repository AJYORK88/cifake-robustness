import json
import sys
from pathlib import Path

import yaml

from src import baseline
from src.dataset import make_split, write_split


def test_baseline_scores_test_split_and_logs_it(cfg, tmp_path, monkeypatch):
    train, val, _ = make_split(cfg["data"]["raw_dir"], 5, 42)
    write_split(train, Path(cfg["data"]["splits_dir"]) / "train.txt")
    write_split(val, Path(cfg["data"]["splits_dir"]) / "val.txt")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    results = Path(cfg["output"]["results_dir"])

    monkeypatch.setattr(sys, "argv", ["x", "--config", str(cfg_path), "--split", "test", "--corrupt", "none", "blur"])
    baseline.main()

    for c in ("none", "blur"):
        out = json.loads((results / f"eval_baseline_test_{c}.json").read_text())
        assert out["split"] == "test" and out["metrics"]["n"] == 12
    assert not (results / "baseline_val.json").exists()            # test run never touches the val file
    assert "baseline_logreg" in (results / "test_runs.log").read_text()
