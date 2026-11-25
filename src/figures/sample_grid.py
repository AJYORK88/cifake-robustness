"""Sample images: a few REAL and a few FAKE validation images.

Data: data/splits/val.txt (read with src.dataset.read_split, labels via src.dataset.label_from_path); open images with data.image_path(...). Pick them reproducibly with random.Random(42).
Save: data.figure_path("sample_grid.png") and return that path.
Caption in the report must state the split and the metric.
"""

from __future__ import annotations

from src.figures import data


def make():
    raise NotImplementedError("sample_grid")
