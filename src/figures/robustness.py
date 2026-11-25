"""Main result: validation ROC-AUC under each test-time corruption, for each model.

Data: data.eval_metrics(model, corrupt)["roc_auc"] for model in data.MODELS (skip any not yet run) and corrupt in data.CONDITIONS.
Save: data.figure_path("robustness.png") and return that path.
Caption in the report must state the split and the metric.
"""

from __future__ import annotations

from src.figures import data


def make():
    raise NotImplementedError("robustness")
