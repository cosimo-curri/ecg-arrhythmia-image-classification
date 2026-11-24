from pathlib import Path

import numpy as np
import pytest

from ecg_classification.data import ImageSample
from ecg_classification.evaluation import evaluate_predictions


def test_confusion_matrix_uses_correct_ground_truth_order():
    test_samples = [
        ImageSample(Path("1.png"), "NOR", "100", 0),
        ImageSample(Path("2.png"), "PVC", "100", 0),
        ImageSample(Path("3.png"), "PVC", "101", 0),
    ]
    predictions = np.zeros((3, 8))
    predictions[0, 0] = 1
    predictions[1, 0] = 1
    predictions[2, 1] = 1
    result = evaluate_predictions(predictions, test_samples)
    assert result["confusion_matrix"].shape == (8, 8)
    assert result["confusion_matrix"][1, 0] == 1  # True PVC, predicted NOR.
    assert result["confusion_matrix"][0, 1] == 0
    assert result["recall"] == pytest.approx(2 / 3)


def test_evaluation_rejects_mismatched_predictions():
    with pytest.raises(ValueError):
        evaluate_predictions(np.zeros((2, 8)), [ImageSample(Path("a"), "NOR", "100", 0)])


def test_evaluation_rejects_nan_scores():
    samples = [ImageSample(Path("a"), "NOR", "100", 0)]
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_predictions(np.array([[float("nan")] + [0.0] * 7]), samples)


def test_per_class_metrics_expose_zero_support_and_macro_average():
    samples = [
        ImageSample(Path("1.png"), "NOR", "100", 0),
        ImageSample(Path("2.png"), "PVC", "101", 0),
    ]
    pred = np.zeros((2, 8), dtype="float32")
    pred[:, 0] = 1.0
    result = evaluate_predictions(pred, samples)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["per_class"]["NOR"]["support"] == 1
    assert result["per_class"]["PVC"]["support"] == 1
    assert result["per_class"]["VFW"]["support"] == 0
    assert result["per_class"]["NOR"]["recall"] == 1.0
    assert result["per_class"]["PVC"]["recall"] == 0.0
    assert result["macro_f1"] == pytest.approx((2/3) / 8)
    assert result["confusion_matrix"].sum() == len(samples)
