"""Test-time corruptions. Functions only. These are never applied to the training split.

Contract for every corruption function: take a 32x32 PIL RGB image and return a 32x32 PIL RGB image.
Parameter values come from configs/default.yaml under `corruptions:`.

Implementation notes (cite these choices in the report):
  - jpeg: Pillow's JPEG encoder (libjpeg) with standard quantization tables scaled by `quality`.
    CIFAKE files are already JPEG (quality 75), so this is a second compression.
  - gaussian_blur: Pillow's GaussianBlur, whose `radius` argument is the standard deviation.
  - downsample_upsample: Pillow resize down by `factor`, then back up, using the same filter both
    ways. Pillow's resize is antialiased when shrinking.

STATUS: implemented.
"""

from __future__ import annotations

import io
from functools import partial
from typing import Callable

from PIL import Image, ImageFilter

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Return the image after JPEG compression at the given quality."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Return the image after a Gaussian blur with standard deviation `sigma` pixels."""
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def downsample_upsample(img: Image.Image, factor: int = 2, interpolation: str = "bilinear") -> Image.Image:
    """Return the image after shrinking by `factor` and resizing back to its original size."""
    try:
        resample = _RESAMPLING[interpolation]
    except KeyError:
        raise ValueError(f"interpolation must be one of {sorted(_RESAMPLING)}, got {interpolation!r}") from None
    img = img.convert("RGB")
    w, h = img.size
    small = img.resize((max(1, w // int(factor)), max(1, h // int(factor))), resample)
    return small.resize((w, h), resample)


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
