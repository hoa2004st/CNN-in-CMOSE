"""Confidence-threshold pseudo-labeling strategy."""

from __future__ import annotations

import numpy as np


THRESHOLDS = (-0.5, 0.0, 0.5)


def score_to_class(score: float) -> int:
    if score < -0.5:
        return 0
    if score < 0.0:
        return 1
    if score < 0.5:
        return 2
    return 3


def is_confident(score: float, *, margin: float = 0.3) -> bool:
    min_distance = min(abs(float(score) - threshold) for threshold in THRESHOLDS)
    return bool(min_distance > float(margin))


def select_confident_pseudo_labels(
    scores: np.ndarray,
    *,
    margin: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.array([is_confident(float(score), margin=margin) for score in scores], dtype=bool)
    labels = np.array([score_to_class(float(score)) for score in scores], dtype=np.int64)
    return mask, labels
