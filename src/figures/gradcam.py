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


def make():
    raise NotImplementedError("gradcam")
