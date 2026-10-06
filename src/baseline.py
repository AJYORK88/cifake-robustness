"""Classical baseline: logistic regression on hand-crafted features.

    python -m src.baseline

Requirements (from the project spec):
  - Features: 32-bin RGB color histograms, local binary patterns, and high-frequency energy
    (FFT or DCT magnitude in the upper frequencies). Parameters are in the config under `baseline:`.
  - Fit the scaler and the classifier on the TRAIN split only. Report on VAL.
  - Report ROC-AUC, accuracy, macro-F1, and real-image FPR, using src.metrics so the numbers
    are defined the same way as for the CNNs.
  - Save results/baseline_val.json.

STATUS: TODO for the team.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config, set_seed


def color_histogram(img: np.ndarray, bins: int) -> np.ndarray:
    """img: uint8 array (32, 32, 3). Return a feature vector of RGB color histograms."""
    raise NotImplementedError("baseline.color_histogram")


def lbp_histogram(img: np.ndarray, points: int, radius: int, method: str) -> np.ndarray:
    """img: uint8 array (32, 32, 3). Return a feature vector describing local binary patterns."""
    raise NotImplementedError("baseline.lbp_histogram")


def highfreq_energy(img: np.ndarray, cutoff: float) -> np.ndarray:
    """img: uint8 array (32, 32, 3). Return a feature vector describing high-frequency energy."""
    raise NotImplementedError("baseline.highfreq_energy")


def build_feature_matrix(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the given split: one feature row and one label per image."""
    raise NotImplementedError("baseline.build_feature_matrix")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    raise NotImplementedError("baseline.main")


if __name__ == "__main__":
    main()
