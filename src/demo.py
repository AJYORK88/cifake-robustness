"""Runnable demonstration: pick or upload an image, choose a corruption, and see what each trained model predicts.

    python -m src.demo            # opens http://127.0.0.1:7860 in your browser
    python -m src.demo --share    # also prints a temporary public link (uploads then pass through Gradio's servers)

Needs the trained checkpoints in results/ (not in git, so train them first): smallcnn_best.pt,
smallcnn_aug_best.pt and resnet18_best.pt. Any that are missing are skipped. The baseline is not included:
it saves no model file and refits on the training set each run.

The example images are VALIDATION images, never test images. Uploaded images are resized to 32x32, the size
every model was trained on; photos unlike CIFAR-10 (large, detailed, different subjects) are outside what the
models learned, so treat those predictions with caution.
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.config import get_device, load_config, repo_path
from src.corrupt import corruption_names, get_corruption
from src.dataset import data_root, label_from_path, read_split
from src.inference import load_checkpoint

DEMO_MODELS = {
    "SmallCNN": "smallcnn_best.pt",
    "SmallCNN + augmentation": "smallcnn_aug_best.pt",
    "Frozen ResNet18": "resnet18_best.pt",
}


def load_models(cfg: dict, device) -> dict:
    """{display name: trained model} for every checkpoint present in results/."""
    results = repo_path(cfg["output"]["results_dir"])
    models = {}
    for name, filename in DEMO_MODELS.items():
        if (results / filename).exists():
            models[name], _ = load_checkpoint(results / filename, device)
    if not models:
        raise FileNotFoundError(f"No trained checkpoints in {results}; train the models first (see README).")
    return models


@torch.no_grad()
def predict(models: dict, cfg: dict, device, image: Image.Image, corrupt: str):
    """Return (the image the models see, enlarged 8x for display; rows of [model, P(FAKE), prediction])."""
    size = cfg["data"]["image_size"]
    img = image.convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    corrupt_fn = get_corruption(corrupt, cfg)
    if corrupt_fn is not None:
        img = corrupt_fn(img)
    x = to_tensor(img).unsqueeze(0).to(device)          # same preprocessing as CifakeDataset: [0, 1], (1, 3, 32, 32)
    rows = []
    for name, model in models.items():
        p_fake = float(torch.sigmoid(model(x))[0])
        rows.append([name, round(p_fake, 4), "FAKE" if p_fake >= cfg["eval"]["threshold"] else "REAL"])
    shown = img.resize((size * 8, size * 8), Image.Resampling.NEAREST)   # nearest keeps the real pixels visible
    return shown, rows


def example_rows(cfg: dict) -> list[list[str]]:
    """Clickable examples (validation images only). Prefers the failure-case selections, so the demo can show a
    fake that SmallCNN catches when clean but misses when blurred."""
    root = data_root(repo_path(cfg["data"]["raw_dir"]))
    failure_file = repo_path(cfg["output"]["results_dir"]) / "failure_cases_smallcnn_blur.json"
    rows = []
    if failure_file.exists():
        groups = json.loads(failure_file.read_text(encoding="utf-8"))["groups"]
        for group, corrupt in (("broken_by_corruption", "none"), ("broken_by_corruption", "blur"),
                               ("correct_real", "none"), ("correct_real", "blur"), ("missed_fake", "none")):
            if groups.get(group):
                rows.append([str(root / groups[group][0]["path"]), corrupt])
    if not rows:
        val = read_split(repo_path(cfg["data"]["splits_dir"]) / "val.txt")
        for label in (0, 1):
            rows.append([str(root / next(p for p in val if label_from_path(p) == label)), "none"])
    return rows


def build_app(cfg: dict, models: dict, device):
    import gradio as gr

    def run(image, corrupt):
        if image is None:
            raise gr.Error("Upload an image or click one of the examples first.")
        return predict(models, cfg, device, image, corrupt)

    return gr.Interface(
        fn=run,
        inputs=[gr.Image(type="pil", label="Image (resized to 32x32)"),
                gr.Dropdown(corruption_names(cfg), value="none", label="Test-time corruption")],
        outputs=[gr.Image(type="pil", label="What the models see (32x32, enlarged)"),
                 gr.Dataframe(headers=["Model", "P(FAKE)", "Prediction"], label="Predictions")],
        examples=example_rows(cfg),
        cache_examples=False,
        title="CIFAKE: real photo or diffusion-generated?",
        description=("Each model outputs P(FAKE); 0.5 or above is called FAKE. Try the first example with "
                     "'none' and then 'blur': SmallCNN changes its answer, the augmented model does not."),
        flagging_mode="never",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--share", action="store_true", help="also create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    models = load_models(cfg, device)
    print(f"[demo] loaded: {', '.join(models)}")
    app = build_app(cfg, models, device)
    app.launch(share=args.share, server_port=args.port, inbrowser=True,
               allowed_paths=[str(data_root(repo_path(cfg["data"]["raw_dir"])))])


if __name__ == "__main__":
    main()
