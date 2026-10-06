"""Test-time corruptions. Functions only -- no training, and never applied to the train split.

Each function takes a PIL RGB image (32x32) and returns a PIL RGB image of the SAME size.
CifakeDataset applies the corruption before ToTensor, so PIL in / PIL out is all you need.
(If you also want tensor support, convert with torchvision.transforms.functional.to_pil_image
 at the top and keep the return type PIL.)

Parameters live in configs/default.yaml under `corruptions:` -- don't hard-code them here.

STATUS: get_corruption() is implemented (shared plumbing used by evaluate.py).
        The three corruption functions are TODO for the team.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from PIL import Image


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG at `quality` (1-95) and decode back.

    TODO(team):
      - Save to an io.BytesIO buffer with format="JPEG", quality=quality, then Image.open + convert("RGB").
      - Note for the report: CIFAKE files are ALREADY JPEGs, so this is a second compression.
        Check in the notebook whether REAL and FAKE were originally saved with different
        quantization tables (PIL exposes img.quantization). If they differ, a model could be
        learning compression history instead of diffusion artifacts.
    """
    raise NotImplementedError("corrupt.jpeg")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur with standard deviation `sigma` pixels; output stays 32x32.

    TODO(team):
      - PIL's ImageFilter.GaussianBlur(radius=...) or torchvision.transforms.functional.gaussian_blur.
      - Be explicit in the report about which you used: PIL's `radius` IS the std-dev,
        torchvision needs an odd kernel_size (e.g. 2*ceil(3*sigma)+1) plus sigma.
    """
    raise NotImplementedError("corrupt.gaussian_blur")


def downsample_upsample(img: Image.Image, factor: int = 2, interpolation: str = "bilinear") -> Image.Image:
    """Resize 32 -> 32/factor -> 32 with the given interpolation (destroys high frequencies).

    TODO(team):
      - Map the string to Image.Resampling.BILINEAR / BICUBIC / NEAREST.
      - Use the same interpolation both ways; return exactly the input size.
    """
    raise NotImplementedError("corrupt.downsample_upsample")


_FUNCTIONS = {"jpeg": jpeg, "gaussian_blur": gaussian_blur, "downsample_upsample": downsample_upsample}


def corruption_names(cfg: dict) -> list[str]:
    """Valid --corrupt values: 'none' plus every key under cfg['corruptions']."""
    return ["none", *cfg.get("corruptions", {}).keys()]


def get_corruption(name: str, cfg: dict) -> Callable[[Image.Image], Image.Image] | None:
    """Build the corruption callable named in the config (e.g. 'jpeg40'); 'none' -> None."""
    if name == "none":
        return None
    try:
        spec = dict(cfg["corruptions"][name])
    except KeyError:
        raise ValueError(f"Unknown corruption {name!r}; choose from {corruption_names(cfg)}") from None
    fn = _FUNCTIONS[spec.pop("type")]
    return partial(fn, **spec)
