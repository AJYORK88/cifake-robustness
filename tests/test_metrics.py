import math

from src.metrics import compute_metrics, format_metrics, fpr_on_real


def test_fpr_real_is_fake_given_real():
    # 4 REAL: one predicted FAKE -> 0.25. FAKE rows must not affect it.
    assert fpr_on_real([0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 0]) == 0.25


def test_perfect_and_threshold():
    m = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["fpr_real"] == 0.0
    # score == threshold counts as FAKE
    assert compute_metrics([0, 1], [0.5, 0.9])["fpr_real"] == 1.0


def test_single_class_auc_is_nan_not_crash():
    assert math.isnan(compute_metrics([1, 1], [0.2, 0.9])["roc_auc"])


def test_format_names_split_and_corruption():
    line = format_metrics(compute_metrics([0, 1], [0.1, 0.9]), "val", "jpeg40", "smallcnn")
    assert "split=val" in line and "corrupt=jpeg40" in line
