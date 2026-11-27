#To run
#python -m src.track

#After running that line then terminal will show "Creating new experiment"

#Run "mlflow ui"

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import yaml

from src.config import load_config, repo_path
from src.figures import data

def _flatten_params(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """flatten nested dictionaries into string key/value pairs for mlflow.log_params"""
    flat: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat.update(_flatten_params(v, prefix=f"{prefix}{k}." if prefix else f"{k}."))
        elif isinstance(v, (list, tuple)):
            flat[f"{prefix}{k}"] = json.dumps(v)
        elif v is not None:
            flat[f"{prefix}{k}"] = str(v)
    return flat

#Go back and check this if any errors
def _load_yaml_if_exists(path: Path) -> dict[str, Any]:
    """Load a YAML file if it exists, otherwise return an empty dict."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

def _log_eval_metrics(results_dir: Path, model: str) -> None:
    """Log evaluation metrics from results/eval_<model>_<split>_<corrupt>.json to MLflow."""
    for path in sorted(results_dir.glob(f"eval_{model}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        split = payload.get("split", "unknown")
        corrupt = payload.get("corrupt", "unknown")
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        suffix = f"_{corrupt}" if corrupt != "none" else ""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{split}{suffix}/{metric_name}", value)
                if metric_name == "roc_auc":
                    mlflow.log_metric(f"{split}{suffix}/auc", value)  # log alias for convenience


def _log_history(results_dir: Path, run_name: str) -> None:
    """Log training history from results/<run_name>_history.json to MLflow."""
    """Log per-epoch training metrics from results/<run_name>_history.json to MLflow."""
    history_path = results_dir / f"{run_name}_history.json"
    if not history_path.exists():
        return

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = payload.get("epochs", [])
    if not isinstance(epochs, list):
        return

    for epoch_data in epochs:
        if not isinstance(epoch_data, dict):
            continue
        epoch = epoch_data.get("epoch")
        if epoch is None:
            continue
        for metric_name, value in epoch_data.items():
            if metric_name == "epoch" or not isinstance(value, (int, float)):
                continue
            mlflow.log_metric(f"epoch_{metric_name}", value, step=epoch)


def main() -> None:
    config = load_config("configs/default.yaml")
    results_dir = repo_path(config["output"]["results_dir"])
    mlflow.set_experiment("cifake-robustness")

    for run in data.MODELS:
        run_config_path = results_dir / f"{run}_config.yaml"
        run_config = _load_yaml_if_exists(run_config_path)

        with mlflow.start_run(run_name=run):
            mlflow.set_tag("model", run)
            params = _flatten_params(run_config) if run_config else {"model": run}
            if params:
                mlflow.log_params(params)

            if run_config_path.exists():
                mlflow.log_artifact(str(run_config_path))

            _log_history(results_dir, run)
            _log_eval_metrics(results_dir, run)


if __name__ == "__main__":
    main()