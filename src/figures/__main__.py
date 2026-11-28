"""Regenerate every report figure from saved results:

    python -m src.figures                 # all figures
    python -m src.figures robustness      # just one

A figure whose input results don't exist yet is skipped with a note, so this is safe to run any time.
A figure whose code raises an error is reported (with the traceback) and the others still build; the
command then exits with status 1.
"""

from __future__ import annotations

import importlib
import sys
import traceback

FIGURE_MODULES = ("sample_grid", "training_curves", "robustness", "augmentation", "failure_grids", "gradcam")


def main() -> None:
    wanted = sys.argv[1:] or list(FIGURE_MODULES)
    unknown = [w for w in wanted if w not in FIGURE_MODULES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGURE_MODULES)}")
    failed = []
    for name in wanted:
        try:
            module = importlib.import_module(f"src.figures.{name}")
            saved = module.make()
            print(f"[figures] {name}: saved {saved}")
        except NotImplementedError:
            print(f"[figures] {name}: TODO (not written yet)")
        except FileNotFoundError as e:
            print(f"[figures] {name}: skipped, {e}")
        except Exception:
            failed.append(name)
            print(f"[figures] {name}: FAILED\n{traceback.format_exc()}")
    if failed:
        raise SystemExit(f"[figures] figure code raised errors: {failed}")


if __name__ == "__main__":
    main()
