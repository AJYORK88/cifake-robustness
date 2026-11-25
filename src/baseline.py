"""Classical baseline: logistic regression on hand-crafted features.

    python -m src.baseline                            # clean validation only
    python -m src.baseline --corrupt all              # clean + every corruption in the config
    python -m src.baseline --corrupt none jpeg40 blur # any subset
    python -m src.baseline --split test --corrupt all # FINAL REPORTING ONLY: official test set

Requirements (from the project spec):
  - Features: 32-bin RGB color histograms, local binary patterns, and high-frequency energy
    (FFT or DCT magnitude in the upper frequencies). Parameters are in the config under `baseline:`.
  - Fit the scaler and the classifier on the TRAIN split only. Report on VAL.
  - Report ROC-AUC, accuracy, macro-F1, and real-image FPR, using src.metrics so the numbers
    are defined the same way as for the CNNs.
  - Save results/baseline_val.json.

The model is fit ONCE on clean training images; --corrupt only changes the images it is scored on
(test-time corruption). Each corruption writes results/eval_baseline_<split>_<corrupt>.json, named like
evaluate.py's output; the clean validation result is also kept as results/baseline_val.json.
--split test scores the official test set; each such run is logged to results/test_runs.log.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import load_config, repo_path, set_seed
from src.corrupt import corruption_names, get_corruption
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


def build_feature_matrix(cfg: dict, split: str, corrupt: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the given split: one feature row and one label per image.

    `corrupt` names a test-time corruption from the config, applied to each image before features.
    """
    if split == "train" and corrupt != "none":
        raise ValueError("Corruptions are test-time only; refusing to corrupt the training split.")
    corrupt_fn = get_corruption(corrupt, cfg)
    raw_dir = data_root(repo_path(cfg["data"]["raw_dir"]))
    b = cfg["baseline"]
    rows, labels = [], []
    for rel in tqdm(split_paths(cfg, split), desc=f"features [{split} corrupt={corrupt}]"):
        with Image.open(raw_dir / rel) as im:
            pil = im.convert("RGB")
        if corrupt_fn is not None:
            pil = corrupt_fn(pil)
        img = np.asarray(pil)
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
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="images to score; 'test' is for final reporting only")
    ap.add_argument("--corrupt", nargs="+", default=["none"],
                    help="corruptions to score: 'none', names from the config, or 'all'")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    valid = corruption_names(cfg)
    corruptions = valid if args.corrupt == ["all"] else args.corrupt
    unknown = [c for c in corruptions if c not in valid]
    if unknown:
        ap.error(f"unknown corruption(s) {unknown}; choose from {valid} or 'all'")

    results_dir = repo_path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "test":
        print("[model=baseline split=test] WARNING: evaluating on the OFFICIAL TEST SET. Final reporting only.")
        with open(results_dir / "test_runs.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}\tbaseline_logreg\t{' '.join(corruptions)}\n")

    X_tr, y_tr = build_feature_matrix(cfg, "train")
    scaler = StandardScaler().fit(X_tr)
    lr = cfg["baseline"]["logreg"]
    clf = LogisticRegression(max_iter=lr["max_iter"], C=lr["C"], random_state=cfg["seed"])
    clf.fit(scaler.transform(X_tr), y_tr)

    for corrupt in corruptions:
        X_ev, y_ev = build_feature_matrix(cfg, args.split, corrupt)
        scores = clf.predict_proba(scaler.transform(X_ev))[:, 1]
        metrics = compute_metrics(y_ev, scores, cfg["eval"]["threshold"])
        print(format_metrics(metrics, args.split, corrupt, "baseline"))
        record = {"model": "baseline_logreg", "split": args.split, "corrupt": corrupt,
                  "corrupt_params": cfg["corruptions"].get(corrupt), "n_features": int(X_tr.shape[1]),
                  "metrics": metrics}
        outs = [results_dir / f"eval_baseline_{args.split}_{corrupt}.json"]
        if args.split == "val" and corrupt == "none":
            outs.append(results_dir / "baseline_val.json")
        for out in outs:
            save_json(record, out)
        print(f"[model=baseline split={args.split} corrupt={corrupt}] wrote {', '.join(o.name for o in outs)}")


if __name__ == "__main__":
    main()