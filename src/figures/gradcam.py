"""Model interpretation (bonus): Grad-CAM heatmaps over images, clean vs corrupted, SmallCNN vs SmallCNN + augmentation.

Data: data.gradcam("blur", "fake") returns the arrays saved by src.gradcam:
  images (n, 2, 32, 32, 3) uint8 [clean, corrupted]; cams (models, n, 2, 32, 32) in [0, 1];
  p_fake (models, n, 2); models, conditions, groups, paths, labels.
Suggested: for a few "broken_by_corruption" images, show each image clean and corrupted with each model's
heatmap overlaid and its P(FAKE) in the title. The written explanation is what earns the bonus.
Save: data.figure_path("gradcam.png") and return that path.
Caption in the report must state the split (validation), the corruption, and which decision is explained.
"""

from __future__ import annotations

from src.figures import data

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Used AI for like half of this code :(
#missing smallcnn_best.pt and smallcnn_aug_best.pt in results folder, so i cannot run this code to check if it works

def _load_rgb(path: str) -> np.ndarray:
    """Load an image from the dataset and return it as a (32, 32, 3) RGB array."""
    with Image.open(data.image_path(path)) as img:
        return np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # normalize to [0, 1]


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    arr=np.array(cam, dtype=np.float32)
    if arr.ndim == 3:
      arr = np.squeeze(arr)
      vmin, vmax = float(np.min(arr)), float(np.max(arr))
      if vmax > vmin:
          return (arr - vmin) / (vmax - vmin)
      return np.zeros_like(arr)


def _overlay_cam(ax, image: np.ndarray, cam: np.ndarray, title: str) -> None:
    """Overlay a Grad-CAM heatmap on an image and display it on the given axis."""
    ax.imshow(image, interpolation="nearest")
    cam_normalized = _normalize_cam(cam)
    cam_resized = np.array(Image.fromarray((cam_normalized * 255).astype(np.uint8)).resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR), dtype=np.float32) / 255.0
    ax.imshow(cam_resized, cmap="jet", alpha=0.4)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def make(split: str = "val", max_rows: int = 4):
    """Make the Grad-CAM figure for the given split (val or test)."""
    gc_data = data.gradcam("blur", "fake")

    clean_paths = gc_data.get("clean_paths", gc_data.get("paths", []))
    blur_paths = gc_data.get("blur_paths", clean_paths)

    gc_blur = data.gradcam("blur", "fake")

    p_clean_base = gc_blur.get("p_fake_clean_smallcnn", gc_blur.get("prob_clean_smallcnn", np.zeros(len(clean_paths))))
    p_blur_base = gc_blur.get("p_fake_blur_smallcnn", gc_blur.get("prob_blur_smallcnn", np.zeros(len(blur_paths))))
    p_blur_aug = gc_blur.get("p_fake_blur_smallcnn_aug", gc_blur.get("prob_blur_smallcnn_aug", np.zeros(len(blur_paths))))

    cam_clean_base = gc_blur.get("cam_clean_smallcnn", np.zeros((len(clean_paths), 8, 8)))
    cam_blur_base = gc_blur.get("cam_blur_smallcnn", np.zeros((len(blur_paths), 8, 8)))
    cam_blur_aug = gc_blur.get("cam_blur_smallcnn_aug", np.zeros((len(blur_paths), 8, 8)))

    n_rows = min(max_rows, len(clean_paths))
    if n_rows == 0:
        raise ValueError("No Grad-CAM rows found in results/gradcam_blur_fake.npz")

    fig, axes = plt.subplots(n_rows, 5, figsize=(11.5, 2.3 * n_rows), dpi=300)
    if n_rows == 1:
        axes = axes.reshape(1, 5)

    col_headers = [
        "Clean Image",
        "SmallCNN CAM",
        "Blurred Image",
        "SmallCNN CAM",
        "SmallCNN+Aug CAM",
    ]
    for c_idx, header in enumerate(col_headers):
        axes[0, c_idx].text(
            0.5,
            1.28,
            header,
            transform=axes[0, c_idx].transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    for r_idx in range(n_rows):
        img_clean = _load_rgb(str(clean_paths[r_idx]))
        img_blur = _load_rgb(str(blur_paths[r_idx]))

        _overlay_cam(
            axes[r_idx, 0],
            img_clean,
            np.zeros(img_clean.shape[:2]),
            f"P(FAKE)={float(p_clean_base[r_idx]):.2f}",
        )
        axes[r_idx, 0].imshow(img_clean, interpolation="nearest")
        axes[r_idx, 0].set_title(f"P(FAKE)={float(p_clean_base[r_idx]):.2f}", fontsize=8)

        _overlay_cam(
            axes[r_idx, 1],
            img_clean,
            cam_clean_base[r_idx],
            f"P(FAKE)={float(p_clean_base[r_idx]):.2f}",
        )

        axes[r_idx, 2].imshow(img_blur, interpolation="nearest")
        axes[r_idx, 2].set_title(f"P(FAKE)={float(p_blur_base[r_idx]):.2f}", fontsize=8)
        axes[r_idx, 2].set_xticks([])
        axes[r_idx, 2].set_yticks([])

        _overlay_cam(
            axes[r_idx, 3],
            img_blur,
            cam_blur_base[r_idx],
            f"P(FAKE)={float(p_blur_base[r_idx]):.2f}",
        )
        _overlay_cam(
            axes[r_idx, 4],
            img_blur,
            cam_blur_aug[r_idx],
            f"P(FAKE)={float(p_blur_aug[r_idx]):.2f}",
        )

    split_name = "Validation" if split == "val" else "Test"
    fig.text(
        0.01,
        0.01,
        f"Split: {split_name} | Metric: ROC-AUC & P(FAKE) | Coarse 8x8 Grad-CAM overlay (alpha=0.4).",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = data.figure_path("gradcam.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
