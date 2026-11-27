"""Before/after: SmallCNN vs SmallCNN + augmentation under each corruption; mark resample as the unseen corruption.

Data: data.eval_metrics("smallcnn", c) and data.eval_metrics("smallcnn_aug", c) for c in data.CONDITIONS (roc_auc, and optionally miss_rate_fake).
Save: data.figure_path("augmentation.png") and return that path.
Caption in the report must state the split and the metric.
"""
#To make graph
#python -m src.figures augmentation

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.figures import data


def make(split: str = "val"):
    #raise NotImplementedError("augmentation")
    data.apply_style()

    models = ["smallcnn", "smallcnn_aug"]
    conditions = data.CONDITIONS

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    for idx, model in enumerate(models):
        scores = [data.eval_metrics(model, condition, split=split)["roc_auc"] for condition in conditions]
        offset = (idx - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=data.MODEL_LABELS[model], color=data.MODEL_COLORS[model])

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.3f}", ha='center', va='bottom', fontsize=8)

        if "resample" in conditions:
            resample_idx = conditions.index("resample")
            ax.axvspan(resample_idx - 0.5, resample_idx + 0.5, color="#FFF3CD", alpha=0.2, label='Unseen Corruption')
            ax.annotate('Unseen Corruption', xy=(resample_idx, 0.55), xytext=(resample_idx + 0.5, 0.55),
                        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9, color="#444444")

    split_name = data.split_label(split)
    ax.set_title(f"Effect of Augmentation on {split_name} Set")
    ax.set_ylabel("ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([data.CONDITION_LABELS[condition] for condition in conditions])
    ax.set_ylim(0.5, 1.08)
    ax.legend(loc='lower left', fontsize=10)    

    fig.text(0.1, 0.1, f"Data Split: {split_name}, Metric: ROC-AUC", fontsize=10, color="#444444")  

    fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the caption
    out_path = data.figure_path("augmentation.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path