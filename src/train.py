"""Train a model on data/splits/train.txt.

    python -m src.train                      # SmallCNN, configs/default.yaml
    python -m src.train --model resnet18

Requirements (from the project spec):
  - Adam, binary cross-entropy, and batch size / epochs / lr from the config. Fixed seed.
  - After each epoch, compute validation ROC-AUC. Early stopping on val AUC, using the patience in the config.
  - Save the best checkpoint to results/<model>_best.pt, in the format documented in src/inference.py
    (evaluate.py depends on that format).
  - Log the train loss and val AUC for every epoch to results/<model>_history.json.
  - Never use the test split. Never apply corruptions to training data.

Shared helpers you can use: src.dataset.build_loader, src.model.build_model,
src.inference.predict_scores / save_checkpoint, src.metrics.compute_metrics / save_json.

STATUS: CLI, config, seeding, and device setup are implemented. Training is TODO for the team.
"""

from __future__ import annotations

import argparse

import yaml

from src.config import get_device, load_config, repo_path, set_seed
from src.model import MODEL_NAMES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=MODEL_NAMES, default=None, help="overrides train.model in the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["train"]["model"] = args.model
    model_name = cfg["train"]["model"]
    set_seed(cfg["seed"])
    device = get_device()
    results_dir = repo_path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{model_name}_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[train model={model_name}] device={device} seed={cfg['seed']}")

    raise NotImplementedError("train.main: training loop")


if __name__ == "__main__":
    main()
