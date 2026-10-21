"""Classical baseline: logistic regression on hand-crafted features.

    python -m src.baseline

Requirements (from the project spec):
  - Features: 32-bin RGB color histograms, local binary patterns, and high-frequency energy
    (FFT or DCT magnitude in the upper frequencies). Parameters are in the config under `baseline:`.
  - Fit the scaler and the classifier on the TRAIN split only. Report on VAL.
  - Report ROC-AUC, accuracy, macro-F1, and real-image FPR, using src.metrics so the numbers
    are defined the same way as for the CNNs.
  - Save results/baseline_val.json.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import load_config, repo_path, set_seed
from src.dataset import data_root, label_from_path, split_paths
from src.metrics import compute_metrics, format_metrics, save_json


def color_histogram(img: np.ndarray, bins: int) -> np.ndarray:
    """img: uint8 array (32, 32, 3). Return a feature vector of RGB color histograms."""
    feats = []
    for c in range(3):
        counts, _ = np.histogram(img[:, :, c], bins=bins, range=(0, 256))
        feats.append(counts / img[:, :, c].size)
    return np.concatenate(feats).astype(np.float32)


def lbp_histogram(img: np.ndarray, points: int, radius: int, method: str) -> np.ndarray:
    """img: uint8 array (32, 32, 3). Return a feature vector describing local binary patterns."""
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    codes = local_binary_pattern(gray, points, radius, method)
    n_bins = points + 2 if method == "uniform" else 2 ** points
    counts, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
    return (counts / codes.size).astype(np.float32)


def highfreq_energy(img: np.ndarray, cutoff: float) -> np.ndarray:
    """img: uint8 array (32, 32, 3). Return a feature vector describing high-frequency energy."""
    gray = rgb2gray(img)
    power = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(yy - h // 2, xx - w // 2)
    dist = dist / dist.max()
    total = power[dist > 0].sum() + 1e-12
    high = power[dist > cutoff].sum()
    return np.array([high / total, np.log(high + 1e-12)], dtype=np.float32)


def build_feature_matrix(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the given split: one feature row and one label per image."""
    raw_dir = data_root(repo_path(cfg["data"]["raw_dir"]))
    b = cfg["baseline"]
    rows, labels = [], []
    for rel in tqdm(split_paths(cfg, split), desc=f"features [{split}]"):
        with Image.open(raw_dir / rel) as im:
            img = np.asarray(im.convert("RGB"))
        rows.append(np.concatenate([
            color_histogram(img, b["color_bins"]),
            lbp_histogram(img, b["lbp"]["points"], b["lbp"]["radius"], b["lbp"]["method"]),
            highfreq_energy(img, b["highfreq"]["cutoff"]),
        ]))
        labels.append(label_from_path(rel))
    return np.stack(rows), np.array(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    X_tr, y_tr = build_feature_matrix(cfg, "train")
    X_va, y_va = build_feature_matrix(cfg, "val")

    scaler = StandardScaler().fit(X_tr)
    lr = cfg["baseline"]["logreg"]
    clf = LogisticRegression(max_iter=lr["max_iter"], C=lr["C"], random_state=cfg["seed"])
    clf.fit(scaler.transform(X_tr), y_tr)
    scores = clf.predict_proba(scaler.transform(X_va))[:, 1]

    metrics = compute_metrics(y_va, scores, cfg["eval"]["threshold"])
    print(format_metrics(metrics, "val", "none", "baseline"))
    out = repo_path(cfg["output"]["results_dir"]) / "baseline_val.json"
    save_json({"model": "baseline_logreg", "split": "val", "corrupt": "none",
               "n_features": int(X_tr.shape[1]), "metrics": metrics}, out)
    print(f"[model=baseline split=val corrupt=none] wrote {out}")


if __name__ == "__main__":
    main()