"""Models. Both output ONE LOGIT per image, shape (N,). They are trained with BCEWithLogitsLoss.

STATUS: build_model() is implemented (shared plumbing used by train.py / evaluate.py).
        SmallCNN is implemented.
        FrozenResNet18 is implemented.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    """Small CNN trained from scratch.

    Requirements (from the project spec):
      - Three conv blocks with 3x3 kernels, ReLU, batch norm, and max-pooling.
      - Channel widths from the config: [32, 64, 128].
      - Then global average pooling, dropout (p from config), and a linear head to one logit.
      - Input (N, 3, 32, 32) floats in [0, 1]. Output (N,) logits.
    """

    def __init__(
        self,
        channels: list[int] = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        blocks: list[nn.Module] = []
        in_channels = 3

        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(1)


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
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone

    
        if isinstance(weights, str):
            try:
                weights_enum = getattr(models.ResNet18_Weights, weights)
            except AttributeError:
                weights_enum = weights
        else:
            weights_enum = weights

        #this will load the resnet18 model with the specified weights
        self.resnet = models.resnet18(weights=weights_enum)

       # Replace the classification head with a linear layer producing one logit
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 1)

        if self.freeze_backbone:
            # Freeze all parameters in the backbone
            for name, param in self.resnet.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
            # Set backbone to eval mode initially
            self.resnet.eval()

    def train(self, mode: bool = True):
        """Override train to keep frozen backbone in eval mode (protecting BatchNorm running stats)."""
        super().train(mode)
        if self.freeze_backbone:
            self.resnet.eval()
            # Ensure the newly added classification head remains in training mode when model.train() is called
            self.resnet.fc.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )

        #this will normalize the input images using ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        
        x = self.resnet(x)
        return x.squeeze(1)


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