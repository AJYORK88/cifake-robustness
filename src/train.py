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

STATUS: implemented.
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml
from torch import nn
from tqdm import tqdm

from src.config import get_device, load_config, repo_path, set_seed
from src.dataset import build_loader
from src.inference import predict_scores, save_checkpoint
from src.metrics import compute_metrics, save_json
from src.model import MODEL_NAMES, build_model


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=None,
        help="overrides train.model in the config",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.model:
        cfg["train"]["model"] = args.model

    model_name = cfg["train"]["model"]
    set_seed(cfg["seed"])
    device = get_device()

    results_dir = repo_path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(
        results_dir / f"{model_name}_config.yaml",
        "w",
        encoding="utf-8",
    ) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[train model={model_name}] device={device} seed={cfg['seed']}")

    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val", shuffle=False)

    model = build_model(model_name, cfg).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
    )

    max_epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stopping"]["patience"])

    history = {
        "model": model_name,
        "epochs": [],
    }

    best_val_auc = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = results_dir / f"{model_name}_best.pt"
    history_path = results_dir / f"{model_name}_history.json"

    for epoch in range(1, max_epochs + 1):
        model.train()

        total_loss = 0.0
        total_examples = 0
        epoch_start = time.time()

        for x, y in tqdm(train_loader, desc=f"train epoch {epoch}/{max_epochs}", leave=False):
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            batch_size = x.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        train_loss = total_loss / total_examples

        y_true, y_score = predict_scores(
            model,
            val_loader,
            device,
            desc=f"val epoch {epoch}",
        )
        val_metrics = compute_metrics(y_true, y_score, cfg["eval"]["threshold"])
        val_auc = float(val_metrics["roc_auc"])

        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_auc": val_auc,
                "val_accuracy": float(val_metrics["accuracy"]),
                "seconds": round(time.time() - epoch_start, 1),
            }
        )
        save_json(history, history_path)

        print(
            f"[epoch {epoch:02d}/{max_epochs}] "
            f"train_loss={train_loss:.6f} "
            f"val_auc={val_auc:.6f} "
            f"({time.time() - epoch_start:.0f}s)"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path,
                model,
                model_name,
                cfg,
                epoch,
                val_auc,
            )
            print(f"  saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch} "
                f"with val_auc={best_val_auc:.6f}"
            )
            break

    print(
        f"Training complete. Best validation AUC={best_val_auc:.6f} "
        f"at epoch {best_epoch}."
    )


if __name__ == "__main__":
    main()