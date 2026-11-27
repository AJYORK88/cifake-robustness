import numpy as np
import matplotlib.pyplot as plt

MODEL_COLORS = {
    "baseline": "#7F7F7F",  # gray
    "smallcnn": "#1F77B4",  # blue
    "resnet18": "#FF7F0E",  # orange
    "smallcnn_aug": "#2CA02C",  # green
}

MODEL_LABELS = {
    "baseline": "Baseline",
    "smallcnn": "SmallCNN",
    "resnet18": "ResNet18",
    "smallcnn_aug": "SmallCNN + Augmentation",
}

CONDITION_LABELS = {
    "none": "No Corruption",
    "jpeg70": "JPEG 70",
    "jpeg40": "JPEG 40",
    "blur": "Blur",
    "resample": "Resample",
}

def apply_style() -> None:
    """Apply a consistent style to all figures."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.sans-serif": "Arial",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

def split_label(split:str) -> str:
    """Human readable label for dataset split."""
    return "Validation" if split == "val" else "Test" if split == "test" else split.upper()
    