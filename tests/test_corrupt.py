import numpy as np
import pytest
from PIL import Image

from src.corrupt import downsample_upsample, gaussian_blur, jpeg


@pytest.fixture
def noisy():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))


def _err(a, b):
    return np.abs(np.asarray(a, float) - np.asarray(b, float)).mean()


def test_jpeg_lower_quality_distorts_more(noisy):
    assert _err(jpeg(noisy, 95), noisy) < _err(jpeg(noisy, 70), noisy) < _err(jpeg(noisy, 40), noisy)


def test_blur_removes_high_frequency_detail(noisy):
    def roughness(im):  # mean absolute difference between horizontal neighbours
        a = np.asarray(im, float)
        return np.abs(np.diff(a, axis=1)).mean()
    assert roughness(gaussian_blur(noisy, 1.0)) < 0.5 * roughness(noisy)
    assert roughness(gaussian_blur(noisy, 2.0)) < roughness(gaussian_blur(noisy, 1.0))


def test_resample_keeps_size_and_rejects_unknown_filter():
    odd = Image.new("RGB", (33, 31), (10, 20, 30))
    assert downsample_upsample(odd, 2).size == (33, 31)
    with pytest.raises(ValueError, match="interpolation"):
        downsample_upsample(odd, 2, "lanczos-ish")


def test_corruptions_are_deterministic(noisy):
    for f in (lambda im: jpeg(im, 40), lambda im: gaussian_blur(im, 1.0), lambda im: downsample_upsample(im, 2)):
        assert np.array_equal(np.asarray(f(noisy)), np.asarray(f(noisy)))


def test_baseline_refuses_to_corrupt_training_features(cfg):
    from src.baseline import build_feature_matrix
    with pytest.raises(ValueError, match="test-time only"):
        build_feature_matrix(cfg, "train", "jpeg40")
