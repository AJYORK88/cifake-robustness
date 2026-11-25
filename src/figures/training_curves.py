"""Training curves: training loss and validation ROC-AUC per epoch, for each trained model.

Data: data.history("smallcnn"), data.history("smallcnn_aug") and, if present, data.history("resnet18").
Save: data.figure_path("training_curves.png") and return that path.
Caption in the report must state the split and the metric.
"""

from __future__ import annotations

from src.figures import data


def make():
    raise NotImplementedError("training_curves")
