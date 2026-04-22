"""Preprocessing utilities for the strict CMOSE comparison pipeline."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def reduce_sample_matrix(
    matrix: np.ndarray,
    *,
    method: str = "svd",
    n_components: int = 300,
) -> tuple[np.ndarray, float]:
    """Reduce one ``frames x features`` matrix to ``frames x n_components``."""
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "svd":
        reducer = TruncatedSVD(n_components=n_components)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    reduced = reducer.fit_transform(matrix)
    explained = float(np.sum(getattr(reducer, "explained_variance_ratio_", 0.0)))
    return reduced.astype(np.float32), explained


def minmax_normalize_per_sample(matrix: np.ndarray) -> np.ndarray:
    """Normalize one matrix to the paper's [0, 1] range."""
    min_value = float(matrix.min())
    max_value = float(matrix.max())
    if max_value <= min_value:
        return np.zeros_like(matrix, dtype=np.float32)
    normalized = (matrix - min_value) / (max_value - min_value)
    return normalized.astype(np.float32)


def normalize_dataset_per_sample(
    matrices: np.ndarray,
    *,
    progress_desc: str | None = None,
    chunk_size: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Normalize every sample matrix independently to [0, 1]."""
    if matrices.ndim != 3:
        raise ValueError(f"Expected a 3-D array, got shape {matrices.shape}")

    total_samples = matrices.shape[0]
    if total_samples == 0:
        return matrices.astype(np.float32, copy=True)

    normalized = np.empty_like(matrices, dtype=np.float32)
    chunk_size = max(1, int(chunk_size))
    desc = progress_desc or "Normalizing samples"

    for start_idx in tqdm(
        range(0, total_samples, chunk_size),
        desc=desc,
        unit="chunk",
        leave=False,
    ):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk = matrices[start_idx:end_idx].astype(np.float32, copy=False)
        min_values = chunk.min(axis=(1, 2), keepdims=True)
        max_values = chunk.max(axis=(1, 2), keepdims=True)
        ranges = max_values - min_values

        chunk_out = normalized[start_idx:end_idx]
        np.subtract(chunk, min_values, out=chunk_out)

        zero_mask = ranges <= 0.0
        safe_ranges = ranges.copy()
        safe_ranges[zero_mask] = 1.0
        np.divide(chunk_out, safe_ranges, out=chunk_out)

        if np.any(zero_mask):
            zero_mask_2d = zero_mask.reshape(-1)
            chunk_out[zero_mask_2d] = 0.0

        if progress_callback is not None:
            progress_callback(end_idx, total_samples)

    return normalized


def preprocess_dataset(
    matrices: np.ndarray,
    *,
    method: str = "svd",
    n_components: int = 300,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply paper-style reduction and normalization to all samples."""
    processed = []
    explained = []
    for matrix in tqdm(
        matrices,
        desc=progress_desc or f"{method.upper()} reduction",
        unit="sample",
        leave=False,
    ):
        reduced, evr = reduce_sample_matrix(matrix, method=method, n_components=n_components)
        processed.append(minmax_normalize_per_sample(reduced))
        explained.append(evr)
    return np.stack(processed, axis=0), np.array(explained, dtype=np.float32)


def flatten_matrices(matrices: np.ndarray) -> np.ndarray:
    """Flatten ``n x h x w`` matrices for SMOTE."""
    return matrices.reshape(matrices.shape[0], -1)


def reshape_flattened_samples(flattened: np.ndarray, *, side: int = 300) -> np.ndarray:
    """Restore flattened samples to ``n x 1 x side x side`` for CNN input."""
    return flattened.reshape(flattened.shape[0], 1, side, side).astype(np.float32)


def add_channel_dim(matrices: np.ndarray) -> np.ndarray:
    """Convert ``n x h x w`` inputs into ``n x 1 x h x w`` for 2-D CNNs."""
    return matrices[:, np.newaxis, :, :].astype(np.float32)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a lightweight multi-class SMOTE implementation.

    The paper balances every class to the majority count before the final split.
    """
    rng = np.random.default_rng(random_state)
    class_counts = Counter(int(label) for label in y.tolist())
    target_count = max(class_counts.values())

    synthetic_samples = [X]
    synthetic_labels = [y]

    for label, count in sorted(class_counts.items()):
        if count >= target_count:
            continue

        class_samples = X[y == label]
        if len(class_samples) < 2:
            raise ValueError(f"SMOTE requires at least 2 samples for class {label}")

        n_neighbors = min(k_neighbors, len(class_samples) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(class_samples)
        neighbor_indices = nn.kneighbors(class_samples, return_distance=False)[:, 1:]

        needed = target_count - count
        generated = np.empty((needed, X.shape[1]), dtype=np.float32)
        for idx in range(needed):
            base_idx = int(rng.integers(0, len(class_samples)))
            neighbor_pool = neighbor_indices[base_idx]
            neighbor_idx = int(neighbor_pool[rng.integers(0, len(neighbor_pool))])
            lam = float(rng.random())
            generated[idx] = class_samples[base_idx] + lam * (
                class_samples[neighbor_idx] - class_samples[base_idx]
            )

        synthetic_samples.append(generated)
        synthetic_labels.append(np.full(needed, label, dtype=np.int64))

    return np.concatenate(synthetic_samples, axis=0), np.concatenate(synthetic_labels, axis=0)
