"""Preprocessing utilities for the narrowed CMOSE comparison pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from tqdm.auto import tqdm


def fit_feature_normalizer(
    matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-feature train-set mean/std statistics on ``n x frames x features`` arrays."""
    if matrices.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {matrices.shape}")

    mean = matrices.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = matrices.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std <= 0.0] = 1.0
    return mean, std


def normalize_dataset_per_feature(
    matrices: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    progress_desc: str | None = None,
    chunk_size: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Normalize every sample using train-fit per-feature mean/std statistics."""
    if matrices.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {matrices.shape}")
    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Expected 1-D mean/std arrays for per-feature normalization")
    if matrices.shape[2] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError(
            "Feature statistics shape mismatch: "
            f"matrices={matrices.shape}, mean={mean.shape}, std={std.shape}"
        )

    total_samples = matrices.shape[0]
    if total_samples == 0:
        return matrices.astype(np.float32, copy=True)

    normalized = np.empty_like(matrices, dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    desc = progress_desc or "Normalizing samples"
    mean_reshaped = mean.reshape(1, 1, -1)
    std_reshaped = std.reshape(1, 1, -1)

    for start_idx in tqdm(
        range(0, total_samples, chunk_size),
        desc=desc,
        unit="chunk",
        leave=False,
    ):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk = matrices[start_idx:end_idx].astype(np.float32, copy=False)

        chunk_out = normalized[start_idx:end_idx]
        np.subtract(chunk, mean_reshaped, out=chunk_out)
        np.divide(chunk_out, std_reshaped, out=chunk_out)

        if progress_callback is not None:
            progress_callback(end_idx, total_samples)

    return normalized
