"""Qualitative analysis: example images per group (correct, false positive, missed fake, broken by corruption), with their P(FAKE).

Data: data.failure_cases("smallcnn", "blur") (and "smallcnn_aug" for before/after); images via data.image_path(record["path"]). For broken_by_corruption, show the clean image beside its corrupted version (src.corrupt.get_corruption).
Save: data.figure_path("failure_grids.png") and return that path.
Caption in the report must state the split and the metric.
"""
#To make graph
#python -m src.figures failure_grids

from __future__ import annotations

import matplotlib.pyplot as plt
from PIL import Image

from src.figures import data


def _collect_items(payload: dict, limit: int = 6) -> list[dict]:
    """Extract up to `limit` image records from a failure_cases JSON payload."""
    if not isinstance(payload, dict):
        return []

    groups = payload.get("groups")
    if isinstance(groups, dict) and groups:
        # Prioritize fake/blur/miss groups first (clean hit -> blur miss fakes)
        priority = [
            k for k in groups
            if any(w in k.lower() for w in ("fake", "blur", "miss", "flip"))
        ]
        ordered = priority + [k for k in groups if k not in priority]

        out: list[dict] = []
        for k in ordered:
            rows = groups.get(k) or []
            for row in rows:
                if isinstance(row, dict) and row.get("path"):
                    out.append(row)
                    if len(out) >= limit:
                        return out
        if out:
            return out

    for list_key in ("cases", "items", "examples", "samples"):
        rows = payload.get(list_key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict) and r.get("path")][:limit]

    return []


def _cell_label(item: dict) -> str:
    """Format short ground-truth / score text above each image tile."""
    label = ""
    for k in ("true_label", "label", "class_name", "ground_truth", "y_true"):
        if k in item and item[k] is not None:
            label = str(item[k]).upper()
            break

    score = None
    for k in ("prob_fake", "fake_prob", "p_fake", "score", "prob", "y_score"):
        v = item.get(k)
        if isinstance(v, (int, float)):
            score = v
            break

    if label and score is not None:
        return f"{label} ({score:.2f})"
    if label:
        return label
    if score is not None:
        return f"p={score:.2f}"
    return "sample"


def make(split: str = "val", max_per_row: int = 6):
    """Build qualitative blur failure grid comparing smallcnn and smallcnn_aug."""
    fc_base = data.failure_cases("smallcnn", "blur")
    fc_aug = data.failure_cases("smallcnn_aug", "blur")

    base_items = _collect_items(fc_base, limit=max_per_row)
    aug_items = _collect_items(fc_aug, limit=max_per_row)

    cols = max(len(base_items), len(aug_items), 1)
    fig, axes = plt.subplots(2, cols, figsize=(2.1 * cols, 4.8), dpi=300)
    if cols == 1:
        axes = axes.reshape(2, 1)

    rows = [
        (0, base_items, "SmallCNN (blur)", "#1F77B4"),
        (1, aug_items, "SmallCNN + Aug (blur)", "#2CA02C"),
    ]

    for r_idx, items, row_label, color in rows:
        for c_idx in range(cols):
            ax = axes[r_idx, c_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            if c_idx >= len(items):
                ax.axis("off")
                continue

            img_path = data.image_path(items[c_idx]["path"])
            with Image.open(img_path) as im:
                ax.imshow(im.convert("RGB"), interpolation="nearest")

            ax.set_title(_cell_label(items[c_idx]), fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(1.5)

            if c_idx == 0:
                ax.set_ylabel(row_label, fontsize=8.5, fontweight="bold", color=color)

    split_name = "Validation" if split == "val" else "Test"
    fig.suptitle(f"Blur Failure Examples: SmallCNN vs. Augmented ({split_name} Split)")
    fig.text(
        0.01,
        0.01,
        f"Split: {split_name} | Metric: ROC-AUC (blur failure cases; nearest-neighbor pixel view).",
        fontsize=8,
        color="#444444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path = data.figure_path("failure_grids.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path