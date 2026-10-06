# AI Tool Usage

Guidelines §6 require that permitted AI use be acknowledged, and that code we wrote be clearly
distinguished from code we adapted. Keep this file current: whenever AI assistance touches a file, add a row.
Confirm that this usage is allowed under the course syllabus.

## Code

| Date | Tool | Files | What it did |
|---|---|---|---|
| 2026-10-06 | Grok (xAI) | none | Drafted the project scaffold specification (repo layout, split protocol, hyper-parameters, file list). |
| 2026-10-06 | Claude Code (Anthropic) | `configs/default.yaml`, `src/config.py`, `src/dataset.py`, `src/metrics.py`, `src/inference.py`, `src/evaluate.py`, `tests/*`, `README.md`, `.gitignore`, `.gitattributes`, `requirements.txt`, `pyproject.toml` | Wrote the shared infrastructure (config, frozen split, data loading, metrics, checkpoint format, evaluation CLI) and the tests. Generated the train/val split by running `src.dataset --make-split`. |
| 2026-10-06 | Claude Code (Anthropic) | `src/baseline.py`, `src/corrupt.py`, `src/model.py`, `src/train.py`, `src/failure_cases.py`, `notebooks/01_explore.ipynb` | Created **empty stubs only**: function signatures and requirements copied from the project spec, with no implementation hints. AI-written plumbing inside these files: `corruption_names` / `get_corruption` (corrupt.py), `build_model` (model.py), the argument-parsing/config/seed setup in `train.py` and `baseline.py`, and the notebook's setup cell. Everything marked `TODO(team)` is written by the team. |

## Ideas

| Date | Tool | Idea |
|---|---|---|
| 2026-10-06 | Claude Code (Anthropic) | Suggested checking whether REAL and FAKE files differ in JPEG compression, as a possible confounder (notebook section 4). |

## Team-owned

The `TODO(team)` implementations, the experiments, the analysis, the interpretation of results, and the
report and presentation text. If AI help is used on any of these (explanations, reviews, debugging,
or code), add a row above saying what it was.
