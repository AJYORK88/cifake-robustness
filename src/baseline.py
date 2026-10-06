"""Classical baseline: logistic regression on hand-crafted features.

    python -m src.baseline

Features per image (concatenated):
  1. RGB color histograms: `color_bins` bins per channel -> 3 * 32 = 96 dims
  2. Local binary pattern histogram on grayscale (skimage.feature.local_binary_pattern)
  3. High-frequency energy from the 2-D FFT (or DCT) magnitude above a radial cutoff

Fit on data/splits/train.txt ONLY (StandardScaler fit on train only), report on val.
Output: results/baseline_val.json via src.metrics (same definitions as the CNNs).

STATUS: CLI and output plumbing are implemented. Feature extraction and fitting are TODO.
"""

from __future__ import annotations

import argparse

import numpy as np

from src.config import load_config, repo_path, set_seed


def color_histogram(img: np.ndarray, bins: int) -> np.ndarray:
    """img: uint8 (32, 32, 3). Return a normalized histogram per channel, concatenated (3*bins,).

    TODO(team): np.histogram per channel with range=(0, 256); divide by pixel count.
    """
    raise NotImplementedError("baseline.color_histogram")


def lbp_histogram(img: np.ndarray, points: int, radius: int, method: str) -> np.ndarray:
    """LBP histogram on the grayscale image.

    TODO(team): skimage.color.rgb2gray -> local_binary_pattern(gray, points, radius, method).
      With method="uniform" codes are 0..points+1, so use points+2 bins. Normalize.
      skimage warns on float input; converting gray to uint8 (gray*255) avoids it.
    """
    raise NotImplementedError("baseline.lbp_histogram")


def highfreq_energy(img: np.ndarray, cutoff: float) -> np.ndarray:
    """Share of spectral energy above `cutoff` * max radial frequency (grayscale).

    TODO(team): np.fft.fftshift(np.fft.fft2(gray)); radial distance from center; energy = |F|^2.
      Return a small vector, e.g. [high/total, log(high + eps)]. Exclude the DC term.
      Diffusion upsampling often leaves spectral artifacts, which is why this feature is included.
    """
    raise NotImplementedError("baseline.highfreq_energy")


def extract_features(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Concatenate the three feature groups for one image."""
    b = cfg["baseline"]
    return np.concatenate([
        color_histogram(img, b["color_bins"]),
        lbp_histogram(img, b["lbp"]["points"], b["lbp"]["radius"], b["lbp"]["method"]),
        highfreq_energy(img, b["highfreq"]["cutoff"]),
    ])


def build_feature_matrix(cfg: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for 'train' or 'val'.

    TODO(team): use src.dataset.split_paths(cfg, split) and label_from_path; open each image
      with PIL (convert("RGB"), np.asarray -> uint8). 90k images takes a few minutes on CPU;
      consider caching X/y to results/*.npy (gitignore them if large).
    """
    raise NotImplementedError("baseline.build_feature_matrix")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # TODO(team):
    #   X_tr, y_tr = build_feature_matrix(cfg, "train")
    #   X_va, y_va = build_feature_matrix(cfg, "val")
    #   scaler = StandardScaler().fit(X_tr)                       # train only
    #   clf = LogisticRegression(max_iter=..., C=..., random_state=cfg["seed"]).fit(scaler.transform(X_tr), y_tr)
    #   scores = clf.predict_proba(scaler.transform(X_va))[:, 1]  # column 1 = P(FAKE) since FAKE=1
    #   metrics = compute_metrics(y_va, scores, cfg["eval"]["threshold"])
    #   print(format_metrics(metrics, "val", "none", "baseline"))
    #   save_json({"model": "baseline_logreg", "split": "val", "corrupt": "none",
    #              "feature_dims": X_tr.shape[1], "metrics": metrics},
    #             repo_path(cfg["output"]["results_dir"]) / "baseline_val.json")
    #
    # Later (robustness comparison): score the baseline on corrupted val images too, by applying
    # src.corrupt.get_corruption(name, cfg) to each PIL image before feature extraction.
    raise NotImplementedError("Baseline is TODO -- see comments in src/baseline.py")


if __name__ == "__main__":
    main()
