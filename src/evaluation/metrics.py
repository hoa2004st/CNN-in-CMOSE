"""Evaluation metrics for CMOSE classification and label-distance scoring."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
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


def label_distance_metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Compute distance metrics by treating class labels as ordinal IDs."""
    confusion = np.asarray(confusion, dtype=np.float64)
    total = float(confusion.sum())
    if total <= 0.0:
        return {"mae": 0.0, "mse": 0.0, "macro_mae": 0.0}

    num_classes = confusion.shape[0]
    class_ids = np.arange(num_classes)
    distances = np.abs(class_ids[:, None] - class_ids[None, :])
    row_sums = confusion.sum(axis=1)
    per_class_mae = np.divide(
        (confusion * distances).sum(axis=1),
        row_sums,
        out=np.full(num_classes, np.nan, dtype=np.float64),
        where=row_sums > 0,
    )

    return {
        "mae": float((confusion * distances).sum() / total),
        "mse": float((confusion * distances**2).sum() / total),
        "macro_mae": float(np.nanmean(per_class_mae)) if np.any(row_sums > 0) else 0.0,
    }


def agreement_metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    """Compute unweighted and quadratic weighted Cohen's kappa from a confusion matrix."""
    confusion = np.asarray(confusion, dtype=np.float64)
    return {
        "cohen_kappa": _kappa_from_confusion(confusion),
        "quadratic_weighted_kappa": _kappa_from_confusion(confusion, weights="quadratic"),
    }


def _kappa_from_confusion(confusion: np.ndarray, *, weights: str | None = None) -> float:
    total = float(confusion.sum())
    if total <= 0.0:
        return 0.0

    num_classes = confusion.shape[0]
    if weights is None:
        weight_matrix = np.ones_like(confusion, dtype=np.float64)
        np.fill_diagonal(weight_matrix, 0.0)
    elif weights == "quadratic":
        if num_classes <= 1:
            weight_matrix = np.zeros_like(confusion, dtype=np.float64)
        else:
            class_ids = np.arange(num_classes, dtype=np.float64)
            weight_matrix = ((class_ids[:, None] - class_ids[None, :]) ** 2) / float((num_classes - 1) ** 2)
    else:
        raise ValueError(f"Unsupported kappa weights: {weights}")

    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / total
    observed_disagreement = float((weight_matrix * confusion).sum() / total)
    expected_disagreement = float((weight_matrix * expected).sum() / total)
    if expected_disagreement <= 0.0:
        return 1.0 if observed_disagreement <= 0.0 else 0.0
    return float(1.0 - observed_disagreement / expected_disagreement)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | dict | str]:
    labels = list(range(4))
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))

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
        "mse": mse,
        **label_distance_metrics_from_confusion(cm),
        **agreement_metrics_from_confusion(cm),
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "classification_report_dict": report_dict,
        "classification_report": report_text,
    }
