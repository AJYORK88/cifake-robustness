"""Train a model on data/splits/train.txt, early-stopping on validation ROC-AUC.

    python -m src.train                      # SmallCNN, configs/default.yaml
    python -m src.train --model resnet18

Outputs (in results/):
    <model>_best.pt          best-val-AUC checkpoint, format defined in src/inference.py (gitignored)
    <model>_config.yaml      exact config used for the run (commit this)
    <model>_history.json     per-epoch log (commit this)

STATUS: CLI, config, seeding, and device setup are implemented. The training loop is TODO.
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

    # TODO(team): implement the training loop. Suggested shape:
    #
    #   from src.dataset import build_loader
    #   from src.model import build_model
    #   from src.inference import predict_scores, save_checkpoint
    #   from src.metrics import compute_metrics, save_json
    #
    #   train_loader = build_loader(cfg, "train")                  # shuffled, seeded, NO corruption
    #   val_loader   = build_loader(cfg, "val", shuffle=False)
    #   model = build_model(model_name, cfg).to(device)
    #   params = [p for p in model.parameters() if p.requires_grad]   # head-only for resnet18
    #   optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    #   loss_fn = torch.nn.BCEWithLogitsLoss()                    # labels must be float: y.float()
    #
    #   for epoch in 1..epochs:
    #       model.train(); accumulate mean train loss
    #       y, s = predict_scores(model, val_loader, device)      # scores = P(FAKE)
    #       val = compute_metrics(y, s, cfg["eval"]["threshold"])
    #       history.append({"epoch", "train_loss", "val_auc", "val_accuracy", "seconds"})
    #       if val AUC improved: save_checkpoint(results_dir / f"{model_name}_best.pt", ...); reset patience
    #       else: patience += 1; stop when patience == cfg["train"]["early_stopping"]["patience"]
    #       save_json(history, results_dir / f"{model_name}_history.json")   # every epoch, survives crashes
    #
    # Never touch the test split here.
    raise NotImplementedError("Training loop is TODO -- see comments in src/train.py")


if __name__ == "__main__":
    main()
