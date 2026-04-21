"""Evaluation utilities for the CNN engagement detection pipeline.

Computes and (optionally) plots:
* Overall accuracy
* Per-class and macro / weighted F1 scores
* Confusion matrix
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)

ENGAGEMENT_LABELS = ["Not Engaged", "Barely Engaged", "Engaged", "Highly Engaged"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run inference and return predicted class indices.

    Parameters
    ----------
    model:
        Trained :class:`~src.model.EngagementCNN`.
    X:
        Feature array of shape ``(n_samples, 1, grid_h, grid_w)``.
    batch_size:
        Mini-batch size for inference.
    device:
        Torch device.  Defaults to CUDA if available, else CPU.

    Returns
    -------
    numpy.ndarray of shape ``(n_samples,)`` with integer class predictions.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_t = torch.from_numpy(X).float()
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            batch = X_t[start : start + batch_size].to(device)
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] = ENGAGEMENT_LABELS,
    output_dir: Optional[str | Path] = None,
) -> dict:
    """Compute classification metrics and optionally save a confusion-matrix plot.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.
    label_names:
        Human-readable class names (index → name).
    output_dir:
        If given, a ``confusion_matrix.png`` is saved in this directory.

    Returns
    -------
    dict with keys:
    * ``accuracy``   – overall accuracy (float).
    * ``f1_macro``   – macro-averaged F1 (float).
    * ``f1_weighted`` – weighted-average F1 (float).
    * ``report``     – full classification report as a string.
    * ``conf_matrix`` – confusion matrix as a 2-D numpy.ndarray.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    logger.info("Accuracy      : %.4f", acc)
    logger.info("F1 (macro)    : %.4f", f1_macro)
    logger.info("F1 (weighted) : %.4f", f1_weighted)
    logger.info("\n%s", report)

    if output_dir is not None:
        _save_confusion_matrix(cm, label_names, Path(output_dir))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "conf_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib / seaborn not available; skipping confusion-matrix plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix – Engagement Detection")
    fig.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", out_path)
