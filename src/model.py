"""Models. Both output ONE LOGIT per image (shape (N,)); apply torch.sigmoid only at inference.

Pair with nn.BCEWithLogitsLoss in training -- it is the numerically stable form of
"sigmoid head + binary cross-entropy".

STATUS: build_model() is implemented (shared plumbing used by train.py / evaluate.py).
        SmallCNN and FrozenResNet18 are TODO for the team.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Three conv blocks -> global average pool -> dropout -> linear logit.

    Each block: Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2).
    Channels from config: [32, 64, 128]  => spatial 32 -> 16 -> 8 -> 4.
    Head: AdaptiveAvgPool2d(1) -> flatten -> Dropout(p) -> Linear(128, 1).

    TODO(team): implement __init__ and forward. forward must return shape (N,), e.g. .squeeze(1).
    Report the parameter count: sum(p.numel() for p in model.parameters()).
    """

    def __init__(self, channels: list[int] = (32, 64, 128), kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        raise NotImplementedError("SmallCNN.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, 3, 32, 32) in [0, 1]
        raise NotImplementedError("SmallCNN.forward")


class FrozenResNet18(nn.Module):
    """ImageNet ResNet18 as a frozen feature extractor + a trainable linear head.

    TODO(team): implement. Gotchas that will silently wreck results if missed:
      1. Weights: torchvision.models.resnet18(weights=ResNet18_Weights[cfg weights]).
         First run downloads ~45 MB from download.pytorch.org into the torch cache.
      2. Upsample 32 -> 224 INSIDE forward (F.interpolate, bilinear, align_corners=False).
         The dataset stays 32x32 for every model.
      3. Normalize with ImageNet mean/std INSIDE forward, after upsampling. Inputs are [0, 1];
         the pretrained backbone expects mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
         Register them as buffers so .to(device) moves them.
      4. Freeze: requires_grad=False on every backbone param; replace backbone.fc with
         nn.Identity() and put a separate nn.Linear(512, 1) head on top.
      5. BatchNorm: requires_grad=False does NOT stop BN running stats from updating in train mode.
         Override train() so the backbone always stays in eval():
             def train(self, mode=True):
                 super().train(mode); self.backbone.eval(); return self
      6. Only pass self.head.parameters() to the optimizer.
      7. CPU cost: 224px forward passes over 90k images x 15 epochs is slow. Because the backbone
         is frozen, consider caching its 512-d features once (train/val) and training the head
         on the cache. Document whichever you do.
    """

    def __init__(self, weights: str = "IMAGENET1K_V1", input_size: int = 224, freeze_backbone: bool = True):
        super().__init__()
        raise NotImplementedError("FrozenResNet18.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, 3, 32, 32) in [0, 1]
        raise NotImplementedError("FrozenResNet18.forward")


MODEL_NAMES = ("smallcnn", "resnet18")


def build_model(name: str, cfg: dict) -> nn.Module:
    """Construct a model from its config block. Used by train.py and evaluate.py."""
    if name == "smallcnn":
        m = cfg["model"]["smallcnn"]
        return SmallCNN(m["channels"], m["kernel_size"], m["dropout"])
    if name == "resnet18":
        m = cfg["model"]["resnet18"]
        return FrozenResNet18(m["weights"], m["input_size"], m["freeze_backbone"])
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
