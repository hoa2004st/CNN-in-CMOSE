"""Evaluation metrics for CMOSE classification and ordinal scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


def evaluate(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    accuracy = float((preds == labels).mean())
    per_class_acc = []
    for class_id in range(4):
        mask = labels == class_id
        if mask.any():
            per_class_acc.append(float((preds[mask] == class_id).mean()))
    avg_acc = float(np.mean(per_class_acc)) if per_class_acc else 0.0
    return accuracy, avg_acc


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | dict | str]:
    labels = list(range(4))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mae = float(mean_absolute_error(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_normalized = np.divide(cm, row_sums, where=row_sums > 0)
        cm_normalized = np.nan_to_num(cm_normalized)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=["Highly Disengage", "Disengage", "Engage", "Highly Engage"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=["Highly Disengage", "Disengage", "Engage", "Highly Engage"],
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mae": mae,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "classification_report_dict": report_dict,
        "classification_report": report_text,
    }
