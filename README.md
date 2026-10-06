# Robust Detection of Diffusion-Generated Images under Common Corruptions

TCU COSC 50523 Deep Learning, Fall 2026 · Path A (Applied Deep Learning) · Team of two

**Question.** Can a small CNN separate real CIFAR-10 photos from Stable Diffusion v1.4 fakes on the
official CIFAKE split, and how much of that accuracy survives JPEG compression, blur, and resampling
applied **only at test time**?

| | |
|---|---|
| Input | 32×32 RGB image, float in [0, 1] |
| Output | P(FAKE), one logit per image (REAL = 0, FAKE = 1) |
| Objective | Binary cross-entropy (`BCEWithLogitsLoss`) |
| Baseline | Logistic regression on color histograms + LBP + FFT high-frequency energy |
| Neural models | SmallCNN (3 conv blocks) vs. frozen ImageNet ResNet18 + linear head |
| Metrics | ROC-AUC, accuracy, macro-F1, false-positive rate on real images (threshold 0.5) |
| Robustness | JPEG q=40 / q=70, Gaussian blur σ=1.0, 2× down/up-sampling (bilinear) |

---

## Status

| File | Status | Owner |
|---|---|---|
| `configs/default.yaml`, `src/config.py` | ✅ implemented | shared |
| `src/dataset.py` (loader, layout detection, frozen split) | ✅ implemented | shared |
| `src/metrics.py`, `src/inference.py`, `src/evaluate.py` | ✅ implemented | shared |
| `src/baseline.py` (features + logistic regression) | 🟡 TODO stub | _TBD_ |
| `src/corrupt.py` (jpeg / blur / resample) | 🟡 TODO stub | _TBD_ |
| `src/model.py` (SmallCNN, FrozenResNet18) | 🟡 TODO stub | _TBD_ |
| `src/train.py` (training loop) | 🟡 TODO stub | _TBD_ |
| `src/failure_cases.py` (qualitative analysis) | 🟡 TODO stub | _TBD_ |
| `notebooks/01_explore.ipynb` (stats, grid, leakage checks) | 🟡 TODO stub | _TBD_ |

Stubs raise `NotImplementedError` and include docstrings with the expected behavior and known gotchas.
`tests/` holds acceptance tests for the stubs that **skip** until a stub is implemented, then enforce it.
See [AI_USAGE.md](AI_USAGE.md) for which code was AI-assisted (required by guidelines §6).

---

## Setup (Windows PowerShell)

Requires Python 3.11+ (tested on 3.14). CPU works; CUDA is used automatically if available.
Keep the repo **outside OneDrive/Dropbox**: the dataset is 120,000 small files.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked by execution policy, call .\.venv\Scripts\python.exe directly instead
python -m pip install -r requirements.txt
python -m pytest                      # plumbing tests pass; stub tests show as "skipped"
```

macOS/Linux: `source .venv/bin/activate` instead of `Activate.ps1`.

## Data: CIFAKE

- Source: <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images>
- 120,000 images, 32×32 RGB, balanced. Official release: **100,000 train** (50k REAL / 50k FAKE) and
  **20,000 test** (10k / 10k). REAL = CIFAR-10; FAKE = Stable Diffusion v1.4 generations mirroring CIFAR-10 classes.
- License: check the Kaggle dataset page and record it in the report (our proposal lists MIT, the same terms as CIFAR-10).
- Images and checkpoints are **never committed** (see `.gitignore`). Only the split lists in `data/splits/` are.

### Download (Kaggle CLI)

1. Kaggle → Settings → API → *Generate New Token*, and copy the token string. Save it, with nothing else in the file,
   as `%USERPROFILE%\.kaggle\access_token` (Windows; `access_token.txt` also works) or `~/.kaggle/access_token`
   (macOS/Linux). Create the `.kaggle` folder if it doesn't exist:
   `New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"`, then `notepad "$env:USERPROFILE\.kaggle\access_token.txt"`.
   Alternatively, set the `KAGGLE_API_TOKEN` environment variable. A legacy `kaggle.json` also still works. Never commit the token.
   Test it with: `kaggle datasets files birdy654/cifake-real-and-ai-generated-synthetic-images`
2. Download and unzip:

```powershell
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data
Rename-Item data\cifake-real-and-ai-generated-synthetic-images.zip cifake.zip
New-Item -ItemType Directory -Force data\raw | Out-Null
tar -xf data\cifake.zip -C data\raw   # built into Windows 10+; much faster than Expand-Archive for 120k files
python -m src.dataset --check         # prints the detected folder mapping and per-class counts
```

Expected layout: `data/raw/{train,test}/{REAL,FAKE}`. `src/dataset.py` finds these folders
case-insensitively, up to 3 levels deep, so a nested `data/raw/<something>/train/...` also works. `--check`
prints the mapping it found. **Both partners must unzip the same way**, because split files store paths relative to `data/raw`.

### Split (frozen. Do not change.)

```powershell
python -m src.dataset --make-split
```

- The **official test set is frozen.** It is used only by `src/evaluate.py --split test`, once, after
  model selection. Each test run is logged to `results/test_runs.log`.
- Validation is a stratified holdout from the **official train pool only**: 5,000 REAL and 5,000 FAKE, seed 42.
- `data/splits/train.txt` (90,000 paths) and `data/splits/val.txt` (10,000 paths) are committed, so both
  partners use byte-identical splits. Paths are relative, use `/`, and are sorted before the seeded shuffle,
  so the split does not depend on OS or directory-listing order. The script refuses to overwrite an existing split without `--force`.
- Corruptions are **test-time only**. `build_dataset` refuses to corrupt the training split.

## Running experiments

```powershell
python -m src.baseline                                    # -> results/baseline_val.json
python -m src.train                                       # SmallCNN -> results/smallcnn_best.pt, _history.json, _config.yaml
python -m src.train --model resnet18                      # frozen ResNet18 + linear head

python -m src.evaluate --checkpoint results/smallcnn_best.pt                     # val, clean
python -m src.evaluate --checkpoint results/smallcnn_best.pt --corrupt jpeg40    # val, corrupted
#   --corrupt: none | jpeg40 | jpeg70 | blur | resample
python -m src.evaluate --checkpoint results/smallcnn_best.pt --split test        # FINAL REPORTING ONLY
```

Each evaluation writes `results/eval_<model>_<split>_<corrupt>.json`, and every printed line names the model, split, and corruption.

## Reproducibility

- All hyper-parameters live in `configs/default.yaml`: seed 42, 32 px, batch 128, 15 epochs, Adam lr 1e-3,
  BCE, early stopping on val AUC with patience 3, dropout 0.3, channels [32, 64, 128]. For a new experiment, copy the
  file rather than editing values in code. `train.py` writes the exact config used next to each checkpoint.
- `src.config.set_seed` seeds Python, NumPy, and PyTorch, and requests deterministic cuDNN. GPU runs can still
  differ slightly in the last digits.
- Large files (`data/raw`, `*.jpg/*.png` under `data/`, `results/*.pt`) are gitignored. Small result
  JSONs and report figures *should* be committed so every number in the report traces to a file.

## Repository layout

```
configs/default.yaml      all hyper-parameters, corruption params, paths
data/splits/              train.txt / val.txt (committed); data/raw is gitignored
src/config.py             config loading, seeding, device
src/dataset.py            layout detection, frozen split, Dataset/DataLoader
src/metrics.py            shared metric definitions (baseline and CNNs use the same ones)
src/inference.py          checkpoint format, batched scoring
src/evaluate.py           val/test x corruption evaluation CLI
src/baseline.py           [TODO] hand-crafted features + logistic regression
src/corrupt.py            [TODO functions] JPEG, blur, resample; registry implemented
src/model.py              [TODO models] SmallCNN, FrozenResNet18; factory implemented
src/train.py              [TODO loop] training with early stopping
src/failure_cases.py      [TODO] success/failure image grids for qualitative analysis
notebooks/01_explore.ipynb  [TODO] class counts, sample grid, leakage + JPEG-table checks
tests/                    pytest: plumbing tests + acceptance tests for stubs
results/                  outputs (checkpoints gitignored)
```

## Not yet in scope

Grad-CAM, W&B experiment tracking, and an inference demo are deliberately left out for now. Each is worth
1.25% bonus (guidelines §5) and can be added later without restructuring: hook into `train.py` per epoch,
and load the model through `src.inference.load_checkpoint`.

## References

Verify every citation before it goes in the report (guidelines §6).

- J. J. Bird and A. Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated
  Synthetic Images," *IEEE Access*, vol. 12, pp. 15642–15650, 2024.
- A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., University of Toronto, 2009.
  (CIFAR-10; often cited as Krizhevsky & Hinton, but the report itself lists Krizhevsky as the author.)
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent
  Diffusion Models," *CVPR*, 2022. (Stable Diffusion)
- K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016. (ResNet18)
