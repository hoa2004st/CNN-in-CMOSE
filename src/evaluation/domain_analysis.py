"""Domain-shift analysis helpers for CMOSE vs private datasets."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance


def feature_group_wasserstein(source: np.ndarray, target: np.ndarray) -> float:
    """Mean 1-D Wasserstein distance over aligned feature dimensions."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("Expected aligned 2-D feature arrays with the same feature dimension")
    distances = [
        wasserstein_distance(source[:, feature_idx], target[:, feature_idx])
        for feature_idx in range(source.shape[1])
    ]
    return float(np.mean(distances)) if distances else 0.0


def centroid_cosine_distance(source_embeddings: np.ndarray, target_embeddings: np.ndarray) -> float:
    """Cosine distance between source and target embedding centroids."""
    source_center = source_embeddings.mean(axis=0)
    target_center = target_embeddings.mean(axis=0)
    return float(cosine(source_center, target_center))


def compute_domain_gap_score(
    openface_wasserstein_groups: list[float],
    i3d_centroid_cosine_distance: float,
) -> float:
    """Composite score used in the implementation spec."""
    if not openface_wasserstein_groups:
        return float(i3d_centroid_cosine_distance)
    return float(np.mean(openface_wasserstein_groups) + i3d_centroid_cosine_distance)
