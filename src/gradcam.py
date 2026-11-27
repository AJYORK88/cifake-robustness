"""Grad-CAM heatmaps for model interpretation. Computes and saves the heatmaps; does no plotting.

    python -m src.gradcam                       # SmallCNN vs SmallCNN + augmentation, clean vs blurred
    python -m src.gradcam --corrupt resample
    python -m src.gradcam --target predicted    # explain each model's own decision instead of FAKE

Grad-CAM (Selvaraju et al., ICCV 2017) shows WHERE in an image a CNN found the evidence for its decision:
  1. Run the image forward and keep the last convolutional block's output: 128 feature maps of 8x8 (after ReLU,
     before the final max-pool, so the map keeps 8x8 resolution).
  2. Backpropagate the decision score to those maps. The average gradient over each map = how much that map
     (pattern detector) pushed the decision.
  3. Weight each map by that importance, sum, keep only positive evidence (ReLU), upsample to 32x32, scale to [0, 1].
Which decision is explained (--target):
  fake (default)  always the evidence for FAKE (the logit). Same question for every image, model and condition,
                  so clean and corrupted heatmaps are directly comparable. A blank map = no FAKE evidence found.
  predicted       each model's own decision: the FAKE logit if P(FAKE) >= threshold, otherwise the negated logit.

Images (VALIDATION only) come from results/failure_cases_smallcnn_<corrupt>.json (run src.failure_cases first):
fakes broken by the corruption, confident correct reals, and reals wrongly called fake. Each image is explained
clean and corrupted, by both SmallCNN and SmallCNN + augmentation (same architecture, so the maps are comparable).

Outputs:
  results/gradcam_<corrupt>_<target>.npz    images (n, 2, 32, 32, 3) uint8 as the models saw them [clean, corrupted],
                                   cams (models, n, 2, 32, 32) in [0, 1], p_fake (models, n, 2),
                                   models, conditions, groups, paths, labels
  results/gradcam_<corrupt>_<target>.json   summary: per model, mean P(FAKE) per group/condition, and how similar each
                                   image's clean and corrupted heatmaps are (correlation; 1 = same regions)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.transforms.functional import to_tensor

from src.config import load_config, repo_path, set_seed
from src.corrupt import corruption_names, get_corruption
from src.dataset import data_root
from src.inference import load_checkpoint
from src.metrics import save_json

MODELS = ("smallcnn", "smallcnn_aug")
GROUPS = {"broken_by_corruption": 8, "correct_real": 4, "false_positive_real": 4}   # images taken per group


def last_conv_activation(model: nn.Module) -> nn.Module:
    """SmallCNN's last ReLU: the output of the final conv block, before its max-pool (128 x 8 x 8)."""
    return [m for m in model.features if isinstance(m, nn.ReLU)][-1]


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module, threshold: float = 0.5, target: str = "fake"):
        if target not in ("fake", "predicted"):
            raise ValueError("target must be 'fake' or 'predicted'")
        self.model, self.layer, self.threshold, self.target = model.eval(), layer, threshold, target

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        """x: (1, 3, H, W) in [0, 1]. Returns (heatmap H x W in [0, 1], P(FAKE))."""
        store = {}

        def keep_activation(module, inputs, output):
            store["activation"] = output
            output.register_hook(lambda grad: store.__setitem__("gradient", grad))

        handle = self.layer.register_forward_hook(keep_activation)
        try:
            self.model.zero_grad(set_to_none=True)
            logit = self.model(x)[0]
            p_fake = torch.sigmoid(logit).item()
            explain_real = self.target == "predicted" and p_fake < self.threshold
            score = -logit if explain_real else logit
            score.backward()
        finally:
            handle.remove()

        activation, gradient = store["activation"][0], store["gradient"][0]   # (channels, h, w)
        weights = gradient.mean(dim=(1, 2))                                    # importance of each feature map
        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy(), p_fake


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two heatmaps; NaN if either is blank."""
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--corrupt", default="blur")
    ap.add_argument("--target", choices=["fake", "predicted"], default="fake")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.corrupt not in corruption_names(cfg) or args.corrupt == "none":
        ap.error(f"--corrupt must be one of {corruption_names(cfg)[1:]}")
    set_seed(cfg["seed"])
    results = repo_path(cfg["output"]["results_dir"])
    threshold = cfg["eval"]["threshold"]

    explainers = {}
    for name in MODELS:
        model, _ = load_checkpoint(results / f"{name}_best.pt", "cpu")
        explainers[name] = GradCAM(model, last_conv_activation(model), threshold, args.target)

    with open(results / f"failure_cases_smallcnn_{args.corrupt}.json", encoding="utf-8") as f:
        groups = json.load(f)["groups"]
    selected = [(g, rec["path"], rec["label"]) for g, k in GROUPS.items() for rec in groups.get(g, [])[:k]]

    root = data_root(repo_path(cfg["data"]["raw_dir"]))
    corrupt_fn = get_corruption(args.corrupt, cfg)
    n = len(selected)
    images = np.zeros((n, 2, 32, 32, 3), dtype=np.uint8)
    cams = np.zeros((len(MODELS), n, 2, 32, 32), dtype=np.float32)
    p_fake = np.zeros((len(MODELS), n, 2), dtype=np.float32)
    for i, (_, path, _) in enumerate(selected):
        with Image.open(root / path) as im:
            clean = im.convert("RGB")
        for j, version in enumerate((clean, corrupt_fn(clean))):
            images[i, j] = np.asarray(version)
            x = to_tensor(version).unsqueeze(0)
            for m, name in enumerate(MODELS):
                cams[m, i, j], p_fake[m, i, j] = explainers[name](x)

    group_names = [g for g, _, _ in selected]
    summary = {"corrupt": args.corrupt, "target": args.target, "split": "val", "n_images": n, "models": {}}
    for m, name in enumerate(MODELS):
        sims = [_similarity(cams[m, i, 0], cams[m, i, 1]) for i in range(n)]
        summary["models"][name] = {
            "mean_heatmap_similarity_clean_vs_corrupted": float(np.nanmean(sims)),
            "by_group": {
                g: {"mean_p_fake_clean": float(p_fake[m, idx, 0].mean()),
                    "mean_p_fake_corrupted": float(p_fake[m, idx, 1].mean()),
                    "mean_heatmap_similarity": float(np.nanmean([sims[i] for i in idx]))}
                for g in GROUPS if (idx := [i for i, gn in enumerate(group_names) if gn == g])
            },
        }

    out = results / f"gradcam_{args.corrupt}_{args.target}"
    np.savez_compressed(out.with_suffix(".npz"), images=images, cams=cams, p_fake=p_fake,
                        models=np.array(MODELS), conditions=np.array(["none", args.corrupt]),
                        groups=np.array(group_names), paths=np.array([p for _, p, _ in selected]),
                        labels=np.array([lab for _, _, lab in selected]))
    save_json(summary, out.with_suffix(".json"))
    for name, s in summary["models"].items():
        print(f"[gradcam {args.corrupt} target={args.target}] {name}: clean-vs-{args.corrupt} heatmap similarity "
              f"{s['mean_heatmap_similarity_clean_vs_corrupted']:.2f}")
    print(f"wrote {out.with_suffix('.npz')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
