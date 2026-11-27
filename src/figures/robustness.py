"""Main result: validation ROC-AUC under each test-time corruption, for each model.

Data: data.eval_metrics(model, corrupt)["roc_auc"] for model in data.MODELS (skip any not yet run) and corrupt in data.CONDITIONS.
Save: data.figure_path("robustness.png") and return that path.
Caption in the report must state the split and the metric.
"""

#To make graph
#python -m src.figures.robustness

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.figures import data


def make(split: str = "val") -> None:
    """Make the robustness figure for the given split (val or test)."""
    data.apply_style()
    #raise NotImplementedError("robustness")

    models = data.MODELS
    conditions = data.CONDITIONS

    x = np.arange(len(conditions))
    width = 0.19

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    for idx, model in enumerate(models):
        scores = [data.eval_metrics(model, condition, split=split)["roc_auc"] 
                  for condition in conditions]
        offset = (idx -(len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=data.MODEL_LABELS[model], color=data.MODEL_COLORS[model])

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha='center', va='bottom', fontsize=8)


        split_name = data.split_label(split)
        ax.set_title(f"Robustness of {split_name} Set")
        ax.set_ylabel("ROC-AUC")
        ax.set_xticks(x)
        ax.set_xticklabels([data.CONDITION_LABELS[condition] for condition in conditions])
        ax.set_ylim(0.5, 1.08)
        ax.legend(loc='lower left', fontsize=10)    

        fig.text(0.1, 0.1, f"Data Split: {split_name}, Metric: ROC-AUC", fontsize=10, color="#444444")

        fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the caption
        out_path = data.figure_path("robustness.png")
        fig.savefig(out_path)
        plt.close(fig)
        return out_path