import json

import pytest

from src.figures import __main__ as runner
from src.figures import data


def test_runner_handles_unwritten_and_missing_figures(capsys):
    runner.sys.argv = ["x"]
    runner.main()                                   # stubs report TODO instead of crashing
    assert "TODO" in capsys.readouterr().out


def test_helpers_read_results_and_explain_missing_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "RESULTS", tmp_path)
    (tmp_path / "eval_smallcnn_val_blur.json").write_text(json.dumps({"metrics": {"roc_auc": 0.74}}))
    (tmp_path / "smallcnn_history.json").write_text(json.dumps({"epochs": [{"epoch": 1, "val_auc": 0.9}]}))
    assert data.eval_metrics("smallcnn", "blur")["roc_auc"] == 0.74
    assert data.history("smallcnn")[0]["epoch"] == 1
    with pytest.raises(FileNotFoundError, match="run the experiment"):
        data.eval_metrics("resnet18", "blur")
