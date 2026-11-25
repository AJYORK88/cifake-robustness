"""Before/after: SmallCNN vs SmallCNN + augmentation under each corruption; mark resample as the unseen corruption.

Data: data.eval_metrics("smallcnn", c) and data.eval_metrics("smallcnn_aug", c) for c in data.CONDITIONS (roc_auc, and optionally miss_rate_fake).
Save: data.figure_path("augmentation.png") and return that path.
Caption in the report must state the split and the metric.
"""

from __future__ import annotations

from src.figures import data


def make():
    raise NotImplementedError("augmentation")
