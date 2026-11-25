"""Qualitative analysis: example images per group (correct, false positive, missed fake, broken by corruption), with their P(FAKE).

Data: data.failure_cases("smallcnn", "blur") (and "smallcnn_aug" for before/after); images via data.image_path(record["path"]). For broken_by_corruption, show the clean image beside its corrupted version (src.corrupt.get_corruption).
Save: data.figure_path("failure_grids.png") and return that path.
Caption in the report must state the split and the metric.
"""

from __future__ import annotations

from src.figures import data


def make():
    raise NotImplementedError("failure_grids")
