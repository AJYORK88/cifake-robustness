"""Regenerate every report figure from saved results:

    python -m src.figures                 # all figures
    python -m src.figures robustness      # just one

A figure whose input results don't exist yet is skipped with a note, so this is safe to run any time.
"""

from __future__ import annotations

import importlib
import sys

FIGURE_MODULES = ("sample_grid", "training_curves", "robustness", "augmentation", "failure_grids", "gradcam")


def main() -> None:
    wanted = sys.argv[1:] or list(FIGURE_MODULES)
    unknown = [w for w in wanted if w not in FIGURE_MODULES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURE_MODULES)}")
    for name in wanted:
        module = importlib.import_module(f"src.figures.{name}")
        try:
            saved = module.make()
            print(f"[figures] {name}: saved {saved}")
        except NotImplementedError:
            print(f"[figures] {name}: TODO (not written yet)")
        except FileNotFoundError as e:
            print(f"[figures] {name}: skipped, {e}")


if __name__ == "__main__":
    main()
