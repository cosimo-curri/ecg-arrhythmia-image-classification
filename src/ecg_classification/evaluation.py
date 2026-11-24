"""Multi-class metrics with correctly ordered ground truth and predictions."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from .data import CLASS_NAMES, ImageSample


def evaluate_predictions(predictions: np.ndarray, test: Sequence[ImageSample]) -> dict:
    """Return weighted and macro scores, per-class support, and confusion matrix."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    predictions = np.asarray(predictions)
    if predictions.shape != (len(test), len(CLASS_NAMES)) or not test:
        raise ValueError("Predictions must be an N x 8 matrix matching nonempty test samples")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values")
    y_true = [sample.label for sample in test]
    y_pred = [CLASS_NAMES[int(index)] for index in np.argmax(predictions, axis=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_NAMES), average="weighted", zero_division=0
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASS_NAMES), average=None, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(per_precision[i]),
            "recall": float(per_recall[i]),
            "f1": float(per_f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(CLASS_NAMES)
    }
    # Macro averages include all eight model outputs. Classes with zero true
    # samples are marked by support=0 and must not be interpreted as assessed.
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_precision": float(np.mean(per_precision)),
        "macro_recall": float(np.mean(per_recall)),
        "macro_f1": float(np.mean(per_f1)),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES)),
    }
