# AI Tool Usage

Guidelines §6 require that permitted AI use be acknowledged, and that code we wrote be clearly
distinguished from code we adapted. Keep this file current: whenever AI assistance touches a file, add a row.
Confirm that this usage is allowed under the course syllabus.

| Date | Tool | Files | What it did |
|---|---|---|---|
| 2026-10-06 | Claude Code (Anthropic) | `configs/default.yaml`, `src/config.py`, `src/dataset.py`, `src/metrics.py`, `src/inference.py`, `src/evaluate.py`, `tests/*`, `README.md`, `.gitignore`, `requirements.txt`, `pyproject.toml` | Generated the shared infrastructure (config, frozen split, data loading, metrics, evaluation CLI) and the tests, from a team-written spec. |
| 2026-10-06 | Claude Code (Anthropic) | `src/baseline.py`, `src/corrupt.py`, `src/model.py`, `src/train.py`, `src/failure_cases.py`, `notebooks/01_explore.ipynb` | Generated **stubs only**: function signatures, docstrings, and TODO notes. The registry in `corrupt.py` and `build_model` in `model.py` are AI-written; the TODO bodies are to be written by the team. |
| 2026-10-06 | Grok (xAI) | — | Drafted the initial project scaffold specification that the above was built from. |

Not AI-generated (team-owned): the experimental question, the analysis, the interpretation of results,
and the report text.
