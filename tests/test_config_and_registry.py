"""Config sanity + plumbing. Includes ready-made acceptance tests for the TODO stubs:
they SKIP while a stub raises NotImplementedError and start enforcing once implemented."""

import pytest
import torch
from PIL import Image

from src.config import load_config
from src.corrupt import corruption_names, get_corruption
from src.model import build_model


def _skip_if_todo(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except NotImplementedError as e:
        pytest.skip(f"TODO not implemented yet: {e}")


def test_config_records_required_hyperparameters():
    c = load_config("configs/default.yaml")
    assert c["seed"] == 42 and c["data"]["image_size"] == 32
    t = c["train"]
    assert (t["batch_size"], t["epochs"], t["optimizer"], t["early_stopping"]["patience"]) == (128, 15, "adam", 3)
    assert isinstance(t["lr"], float) and t["lr"] == 1e-3      # YAML "1e-3" would be a string
    assert c["model"]["smallcnn"]["channels"] == [32, 64, 128] and c["model"]["smallcnn"]["dropout"] == 0.3


def test_corruption_registry():
    c = load_config("configs/default.yaml")
    assert corruption_names(c) == ["none", "jpeg40", "jpeg70", "blur", "resample"]
    assert get_corruption("none", c) is None
    with pytest.raises(ValueError):
        get_corruption("snow", c)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model("vit", load_config("configs/default.yaml"))


# ---------------------------------------------------------------- acceptance tests for stubs
@pytest.mark.parametrize("name", ["jpeg40", "jpeg70", "blur", "resample"])
def test_corruption_returns_32px_rgb_and_changes_image(name):
    c = load_config("configs/default.yaml")
    img = Image.effect_noise((32, 32), 64).convert("RGB")
    out = _skip_if_todo(get_corruption(name, c), img)
    assert out.mode == "RGB" and out.size == (32, 32)
    assert list(out.getdata()) != list(img.getdata())


def test_smallcnn_outputs_one_logit_per_image():
    model = _skip_if_todo(build_model, "smallcnn", load_config("configs/default.yaml"))
    assert model(torch.rand(4, 3, 32, 32)).shape == (4,)
