"""k-NN pseudo-label propagation in embedding space."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def knn_label_propagation(
    source_embeddings: np.ndarray,
    source_labels: np.ndarray,
    target_embeddings: np.ndarray,
    *,
    n_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    knn = KNeighborsClassifier(n_neighbors=int(n_neighbors), metric="cosine")
    knn.fit(source_embeddings, source_labels)
    pseudo_labels = knn.predict(target_embeddings)
    pseudo_probs = knn.predict_proba(target_embeddings)
    return pseudo_labels.astype(np.int64), pseudo_probs.astype(np.float32)
