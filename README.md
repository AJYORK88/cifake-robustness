# Robust Detection of Diffusion-Generated Images under Common Corruptions

TCU COSC 50523 Deep Learning, Fall 2026 · Path A (Applied Deep Learning) · Team: Andrew York & Angelette Munoz

**Question.** Can a small CNN separate real CIFAR-10 photos from Stable Diffusion v1.4 fakes on the
official CIFAKE split, and how much of that accuracy survives JPEG compression, blur, and resampling
applied **only at test time**? As an extension, does training with corrupted images recover the lost robustness?

| | |
|---|---|
| Input | 32×32 RGB image, float in [0, 1] |
| Output | P(FAKE), one logit per image (REAL = 0, FAKE = 1) |
| Objective | Binary cross-entropy (`BCEWithLogitsLoss`) |
| Baseline | Logistic regression on 108 hand-crafted features (color histograms, LBP, FFT high-frequency energy) |
| Neural models | SmallCNN (3 conv blocks, 93,825 parameters); frozen ImageNet ResNet18 + linear head |
| Extension | SmallCNN trained with random JPEG/blur augmentation (resampling held out as an unseen corruption) |
| Metrics | ROC-AUC, accuracy, macro-F1, false-positive rate on real images, miss rate on fakes (threshold 0.5) |
| Test-time corruptions | JPEG q=70 and q=40, Gaussian blur σ=1.0, 2× down/up-sampling (bilinear) |

## Results

**Official test set** (n = 20,000, used once after all model choices were frozen), ROC-AUC:

| Model | Clean | JPEG-70 | JPEG-40 | Blur | Resample |
|---|---|---|---|---|---|
| Baseline (logistic regression) | 0.927 | 0.931 | 0.884 | 0.673 | 0.661 |
| SmallCNN | **0.990** | **0.990** | 0.960 | 0.759 | 0.770 |
| Frozen ResNet18 | 0.964 | 0.962 | 0.939 | 0.850 | 0.803 |
| SmallCNN + augmentation | 0.987 | 0.988 | **0.976** | **0.946** | **0.938** |

Every number comes from `results/eval_<model>_<split>_<corruption>.json` (accuracy, macro-F1 and both error
rates are in the same files). Validation results agree with test within 0.015 everywhere.

---

## Setup (Windows PowerShell)

Requires Python 3.11+ (tested on 3.14, CPU only; CUDA is used automatically if available). Keep the repo
**outside OneDrive/Dropbox**: the dataset is 120,000 small files.

```powershell
cd C:\dev
git clone https://github.com/AJYORK88/cifake-robustness.git
cd cifake-robustness
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked by execution policy, use .\.venv\Scripts\python.exe instead of python below
python -m pip install -r requirements.txt
```

macOS/Linux: `source .venv/bin/activate` instead of `Activate.ps1`.

### Data: CIFAKE

- Source: <https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images> (Bird & Lotfi, 2024)
- 120,000 images, 32×32 RGB, balanced. Official release: **100,000 train** (50k REAL / 50k FAKE) and
  **20,000 test** (10k / 10k). REAL = CIFAR-10; FAKE = Stable Diffusion v1.4 generations mirroring CIFAR-10 classes.
- License: Kaggle lists it as **"other"**; see the dataset page for the terms.
- Images, zips and model checkpoints are **never committed** (see `.gitignore`).

Get the zip one of three ways and save it as `data\cifake.zip`:
- **Kaggle website:** log in, click **Download** on the dataset page, rename the file to `cifake.zip`.
- **Kaggle CLI:** Kaggle → Settings → API → *Generate New Token*; save the token string alone in
  `%USERPROFILE%\.kaggle\access_token.txt` (never commit it), then:
  `kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data` and rename the zip.
- **From a teammate** (104.5 MB).

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
tar -xf data\cifake.zip -C data\raw    # about 2 minutes; much faster than Expand-Archive
python -m src.dataset --check          # expect 50000 / 50000 / 10000 / 10000 and both split files "OK"
python -m pytest                       # all tests should pass
```

`src/dataset.py` finds the `train/` and `test/` folders even if the zip unpacked into a subfolder.

### Split and protocol (frozen: do not change)

- `data/splits/train.txt` (89,487 images) and `data/splits/val.txt` (10,000) are committed, so everyone uses
  byte-identical splits. **Do not run `--make-split`.**
- Validation is a stratified holdout from the **official train pool only** (5,000 REAL + 5,000 FAKE, seed 42).
- **Duplicate removal.** CIFAKE contains 1,336 pairs of byte-identical FAKE images stored under different
  names; 544 pairs span splits (train+test 347, train+val 166, val+test 31). The 513 training copies whose twin
  is in val or test are removed (`data/splits/removed_duplicates.txt`), so no held-out image is ever trained on.
- The **official test set** was evaluated once, after every model choice was frozen. Each test run is logged
  in `results/test_runs.log`.
- **Corruptions are test-time only** for the main experiments. The augmentation experiment
  (`configs/smallcnn_aug.yaml`) is the one deliberate exception and is labeled as such.

---

## Reproducing the results

Times are for a CPU-only laptop. Run from the repo root with the venv active.

```powershell
# 1. Baseline: fits once on train, scores val under every corruption                       (~5-40 min)
python -m src.baseline --corrupt all

# 2. SmallCNN and the augmentation experiment (15 epochs each)                              (~35 min each)
python -m src.train
python -m src.train --config configs/smallcnn_aug.yaml

# 3. Frozen ResNet18: caches backbone features once, trains the linear head, scores val    (~2-3 h)
python -m src.resnet_head

# 4. Validation robustness for the two CNNs                                                 (~1 min each)
foreach ($m in "smallcnn", "smallcnn_aug") {
  foreach ($c in "none", "jpeg70", "jpeg40", "blur", "resample") {
    python -m src.evaluate --checkpoint "results/${m}_best.pt" --corrupt $c
  }
}

# 5. Analysis (validation only)
python -m src.train_vs_val --checkpoint results/smallcnn_best.pt                     # memorization check
python -m src.failure_cases --checkpoint results/smallcnn_best.pt --corrupt blur      # success/failure examples
python -m src.failure_cases --checkpoint results/smallcnn_aug_best.pt --corrupt blur
python -m src.gradcam                                                                # Grad-CAM heatmaps

# 6. Figures: regenerated from the result files in seconds                       -> results/figures/
python -m src.figures
```

**Official test set: final reporting only, once, after all of the above is frozen** (~3 h, mostly ResNet18):

```powershell
python -m src.baseline --split test --corrupt all
foreach ($m in "smallcnn", "smallcnn_aug") {
  foreach ($c in "none", "jpeg70", "jpeg40", "blur", "resample") {
    python -m src.evaluate --checkpoint "results/${m}_best.pt" --split test --corrupt $c
  }
}
python -m src.resnet_head --split test
```

Every printed line names the model, split and corruption. Outputs in `results/`:

| File | Contents |
|---|---|
| `eval_<model>_<split>_<corrupt>.json` | all metrics for one model × split × corruption |
| `<run>_config.yaml`, `<run>_history.json` | exact config and per-epoch log of each trained model |
| `<run>_train_vs_val.json` | train vs validation metrics (memorization check) |
| `failure_cases_<run>_<corrupt>.json` | selected success/failure examples |
| `gradcam_<corrupt>_<target>.npz/.json` | Grad-CAM heatmaps and summary |
| `test_runs.log` | every use of the official test set |
| `figures/` | report figures |
| `*.pt`, `features/` | checkpoints and cached ResNet18 features (gitignored; regenerate with the commands above) |

## Demo

```powershell
python -m src.demo            # opens http://127.0.0.1:7860
```

Pick an example (validation images) or upload an image, choose a corruption, and compare each model's
P(FAKE). Try the first example with `none` and then `blur`: SmallCNN flips to REAL, the augmented model does not.
Needs the trained checkpoints from steps 2-3. `--share` creates a temporary public link (uploads then pass
through Gradio's servers).

## Reproducibility

- All hyper-parameters are in `configs/default.yaml` (seed 42, 32 px, batch 128, up to 15 epochs, Adam lr 1e-3,
  BCE, early stopping on val ROC-AUC with patience 3, dropout 0.3, channels [32, 64, 128]). Experiment configs
  such as `configs/smallcnn_aug.yaml` list only what differs (`base: configs/default.yaml`).
- `src.config.set_seed` seeds Python, NumPy and PyTorch and requests deterministic cuDNN; GPU runs can still
  differ slightly in the last digits.
- Each training run saves its exact config next to its checkpoint, and every reported number traces to a
  committed JSON file in `results/`.

## Repository layout

```
configs/default.yaml        all hyper-parameters, corruption params, paths
configs/smallcnn_aug.yaml   augmentation experiment (only the differences from default)
data/splits/                train.txt / val.txt / removed_duplicates.txt (committed); data/raw is gitignored
notebooks/01_explore.ipynb  class counts, leakage checks (paths and file contents), JPEG-table check
src/config.py               config loading (with base: inheritance), seeding, device
src/dataset.py              layout detection, frozen split + duplicate removal, Dataset/DataLoader
src/corrupt.py              test-time corruptions; TrainAugment for the augmentation experiment
src/metrics.py              shared metric definitions (every model uses the same ones)
src/model.py                SmallCNN, FrozenResNet18
src/train.py                training loop with early stopping on val ROC-AUC
src/resnet_head.py          ResNet18 via cached backbone features
src/inference.py            checkpoint format, batched scoring
src/evaluate.py             model x split x corruption evaluation
src/baseline.py             hand-crafted features + logistic regression
src/train_vs_val.py         memorization check
src/failure_cases.py        selects success/failure examples (no plotting)
src/gradcam.py              Grad-CAM heatmaps (no plotting)
src/demo.py                 Gradio demo
src/figures/                figure code: one make() per figure; python -m src.figures
tests/                      pytest suite
results/                    outputs (see table above)
```

AI assistance is disclosed file by file in [AI_USAGE.md](AI_USAGE.md) (guidelines §6).

## References

- J. J. Bird and A. Lotfi, "CIFAKE: Image Classification and Explainable Identification of AI-Generated
  Synthetic Images," *IEEE Access*, vol. 12, pp. 15642–15650, 2024.
- A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Rep., University of Toronto, 2009.
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-Resolution Image Synthesis with Latent
  Diffusion Models," *CVPR*, 2022.
- K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
- R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-Based Localization," *ICCV*, 2017.
