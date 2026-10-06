"""Models. Both output ONE LOGIT per image, shape (N,). They are trained with BCEWithLogitsLoss.

STATUS: build_model() is implemented (shared plumbing used by train.py / evaluate.py).
        SmallCNN and FrozenResNet18 are TODO for the team.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Small CNN trained from scratch.

    Requirements (from the project spec):
      - Three conv blocks with 3x3 kernels, ReLU, batch norm, and max-pooling.
      - Channel widths from the config: [32, 64, 128].
      - Then global average pooling, dropout (p from config), and a linear head to one logit.
      - Input (N, 3, 32, 32) floats in [0, 1]. Output (N,) logits.
    """

    def __init__(self, channels: list[int] = (32, 64, 128), kernel_size: int = 3, dropout: float = 0.3):
        super().__init__()
        raise NotImplementedError("SmallCNN.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SmallCNN.forward")


class FrozenResNet18(nn.Module):
    """ImageNet-pretrained ResNet18 used as a fixed feature extractor.

    Requirements (from the project spec):
      - Load torchvision ResNet18 with ImageNet weights (weight name from the config).
      - The backbone is frozen and must not change during training.
      - Replace the classification head with a linear layer producing one logit.
      - Accept the same (N, 3, 32, 32) inputs in [0, 1] as SmallCNN. Any resizing to the
        backbone's input size (config: 224) happens inside this class only.
      - Output (N,) logits.
    """

    def __init__(self, weights: str = "IMAGENET1K_V1", input_size: int = 224, freeze_backbone: bool = True):
        super().__init__()
        raise NotImplementedError("FrozenResNet18.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
