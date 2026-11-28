import json

import pytest

from src.figures import __main__ as runner
from src.figures import data


def test_runner_reports_unwritten_and_broken_figures_without_stopping(capsys, monkeypatch):
    import types
    fake = {"todo": lambda: (_ for _ in ()).throw(NotImplementedError()),
            "broken": lambda: 1 / 0,
            "ok": lambda: "ok.png"}
    monkeypatch.setattr(runner, "FIGURE_MODULES", tuple(fake))
    monkeypatch.setattr(runner.importlib, "import_module",
                        lambda name: types.SimpleNamespace(make=fake[name.rsplit(".", 1)[1]]))
    runner.sys.argv = ["x"]
    with pytest.raises(SystemExit):                  # a broken figure makes the command fail...
        runner.main()
    out = capsys.readouterr().out
    assert "todo: TODO" in out and "broken: FAILED" in out and "ok: saved ok.png" in out   # ...but the rest still run


def test_helpers_read_results_and_explain_missing_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "RESULTS", tmp_path)
    (tmp_path / "eval_smallcnn_val_blur.json").write_text(json.dumps({"metrics": {"roc_auc": 0.74}}))
    (tmp_path / "smallcnn_history.json").write_text(json.dumps({"epochs": [{"epoch": 1, "val_auc": 0.9}]}))
    assert data.eval_metrics("smallcnn", "blur")["roc_auc"] == 0.74
    assert data.history("smallcnn")[0]["epoch"] == 1
    with pytest.raises(FileNotFoundError, match="run the experiment"):
        data.eval_metrics("resnet18", "blur")
