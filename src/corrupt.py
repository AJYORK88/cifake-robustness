"""Test-time corruptions. Functions only. These are never applied to the training split.

Contract for every corruption function: take a 32x32 PIL RGB image and return a 32x32 PIL RGB image.
Parameter values come from configs/default.yaml under `corruptions:`.

STATUS: corruption_names() / get_corruption() are implemented (shared plumbing used by evaluate.py).
        The three corruption functions are TODO for the team.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from PIL import Image


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Return the image after JPEG compression at the given quality."""
    raise NotImplementedError("corrupt.jpeg")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Return the image after a Gaussian blur with standard deviation `sigma` pixels."""
    raise NotImplementedError("corrupt.gaussian_blur")


def downsample_upsample(img: Image.Image, factor: int = 2, interpolation: str = "bilinear") -> Image.Image:
    """Return the image after shrinking by `factor` and resizing back to its original size."""
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
