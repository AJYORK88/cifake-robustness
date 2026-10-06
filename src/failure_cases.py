"""Qualitative analysis: save grids of representative successes and failures.

    python -m src.failure_cases --checkpoint results/smallcnn_best.pt --corrupt jpeg40

The course guidelines (section 2.4) require qualitative analysis with successes and failure cases.
Suggested outputs in results/figures/ (PNG figures are small; commit the ones used in the report):
  - most confident correct REAL / correct FAKE
  - most confident WRONG: real called fake (false positives), fake called real (misses)
  - the same images under each corruption, to show what the corruption broke

STATUS: TODO for the team.
"""

from __future__ import annotations

import numpy as np


def select_examples(y_true: np.ndarray, y_score: np.ndarray, k: int = 8, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """Indices for each group: 'tp_fake', 'tn_real', 'fp_real', 'fn_fake', ranked by confidence.

    TODO(team): e.g. fp_real = indices where y_true==0 and score>=threshold, sorted by score descending, top k.
    """
    raise NotImplementedError("failure_cases.select_examples")


def save_grid(dataset, indices: np.ndarray, scores: np.ndarray, title: str, path) -> None:
    """Plot images (dataset[i][0] is a CHW tensor in [0, 1]) with their P(FAKE) and save to `path`.

    TODO(team): matplotlib subplots; img.permute(1, 2, 0).numpy(); interpolation="nearest" for 32px.
    """
    raise NotImplementedError("failure_cases.save_grid")


def main() -> None:
    # TODO(team): argparse like src/evaluate.py; reuse load_checkpoint + predict_scores; val split by default.
    raise NotImplementedError("failure_cases is TODO")


if __name__ == "__main__":
    main()
