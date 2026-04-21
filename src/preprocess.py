"""Preprocessing utilities: feature normalisation, PCA / SVD, and CNN reshaping.

Pipeline
--------
1. Fit a ``StandardScaler`` on the training features and apply it to all splits.
2. Fit a ``PCA`` (or truncated SVD) on the scaled training features.
3. Project all splits into the PCA subspace.
4. Reshape each projected vector into a 2-D grid that can be fed to a CNN as a
   single-channel image: ``(batch, 1, height, width)``.
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Preprocessor:
    """Combines StandardScaler + PCA (or SVD) for the CNN-in-CMOSE pipeline.

    Parameters
    ----------
    n_components:
        Number of PCA / SVD components to keep.  The components are then
        reshaped into a 2-D grid for CNN input; ``n_components`` must equal
        ``grid_h * grid_w`` (see :meth:`reshape_for_cnn`).
    use_svd:
        If ``True`` use :class:`sklearn.decomposition.TruncatedSVD` instead of
        PCA.  SVD does not centre the data, which can be useful when the data
        has already been scaled, but PCA (the default) generally works better.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 64,
        use_svd: bool = False,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.use_svd = use_svd
        self.random_state = random_state

        self.scaler: StandardScaler = StandardScaler()
        if use_svd:
            self.reducer: TruncatedSVD | PCA = TruncatedSVD(
                n_components=n_components,
                random_state=random_state,
            )
        else:
            self.reducer = PCA(
                n_components=n_components,
                random_state=random_state,
            )
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "Preprocessor":
        """Fit the scaler and the PCA / SVD on training data.

        Parameters
        ----------
        X_train:
            Feature matrix of shape ``(n_samples, n_raw_features)``.

        Returns
        -------
        self
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.reducer.fit(X_scaled)
        self._fitted = True
        explained = self._explained_variance_ratio()
        if explained is not None:
            logger.info(
                "PCA fitted: %d components explain %.1f%% of variance.",
                self.n_components,
                explained * 100,
            )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale and project ``X`` into the PCA / SVD subspace.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_raw_features)``.

        Returns
        -------
        numpy.ndarray of shape ``(n_samples, n_components)``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X_scaled = self.scaler.transform(X)
        return self.reducer.transform(X_scaled)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Convenience method: fit on ``X_train`` and return its projection."""
        self.fit(X_train)
        return self.transform(X_train)

    # ------------------------------------------------------------------
    # Reshape for CNN
    # ------------------------------------------------------------------

    @staticmethod
    def reshape_for_cnn(
        X_reduced: np.ndarray,
        grid_h: Optional[int] = None,
        grid_w: Optional[int] = None,
    ) -> np.ndarray:
        """Reshape projected features into a 4-D tensor for PyTorch CNN input.

        Each row (a 1-D vector of ``n_components`` values) is reshaped into a
        ``(grid_h, grid_w)`` 2-D feature map.  The output tensor has shape
        ``(n_samples, 1, grid_h, grid_w)`` – one channel per sample.

        If ``grid_h`` and ``grid_w`` are not given the method tries to make the
        grid as square as possible.  ``n_components`` must be factorisable into
        ``grid_h * grid_w``.

        Parameters
        ----------
        X_reduced:
            Array of shape ``(n_samples, n_components)``.
        grid_h, grid_w:
            Desired grid dimensions.  Product must equal ``n_components``.

        Returns
        -------
        numpy.ndarray of shape ``(n_samples, 1, grid_h, grid_w)``.
        """
        n_samples, n_components = X_reduced.shape
        if grid_h is None or grid_w is None:
            grid_h, grid_w = _square_factors(n_components)
        if grid_h * grid_w != n_components:
            raise ValueError(
                f"grid_h * grid_w ({grid_h} * {grid_w} = {grid_h * grid_w}) "
                f"must equal n_components ({n_components})."
            )
        return X_reduced.reshape(n_samples, 1, grid_h, grid_w).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the fitted preprocessor to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Load a previously saved preprocessor from a file."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected Preprocessor, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _explained_variance_ratio(self) -> Optional[float]:
        evr = getattr(self.reducer, "explained_variance_ratio_", None)
        if evr is not None:
            return float(np.sum(evr))
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Preprocessor(n_components={self.n_components}, "
            f"use_svd={self.use_svd}, fitted={self._fitted})"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _square_factors(n: int) -> Tuple[int, int]:
    """Return the most square (h, w) factorisation of ``n`` with h <= w."""
    h = int(math.isqrt(n))
    while h >= 1:
        if n % h == 0:
            return h, n // h
        h -= 1
    return 1, n  # fallback: (1, n) always works
