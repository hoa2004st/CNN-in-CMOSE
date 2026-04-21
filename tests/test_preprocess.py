"""Tests for src/preprocess.py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.preprocess import Preprocessor, _square_factors


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((100, 50)).astype(np.float32)
    X_test = rng.standard_normal((20, 50)).astype(np.float32)
    return X_train, X_test


# ---------------------------------------------------------------------------
# _square_factors
# ---------------------------------------------------------------------------

class TestSquareFactors:
    @pytest.mark.parametrize("n,expected_product", [
        (64, 64),
        (1, 1),
        (49, 49),
        (17, 17),
        (100, 100),
    ])
    def test_product_equals_n(self, n: int, expected_product: int) -> None:
        h, w = _square_factors(n)
        assert h * w == expected_product

    def test_h_le_w(self) -> None:
        for n in (16, 25, 36, 48, 64):
            h, w = _square_factors(n)
            assert h <= w

    def test_square_numbers(self) -> None:
        h, w = _square_factors(64)
        assert h == 8 and w == 8

        h, w = _square_factors(36)
        assert h == 6 and w == 6


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_fit_transform_shape(self, dummy_data: tuple) -> None:
        X_train, _ = dummy_data
        prep = Preprocessor(n_components=16)
        X_r = prep.fit_transform(X_train)
        assert X_r.shape == (100, 16)

    def test_transform_shape(self, dummy_data: tuple) -> None:
        X_train, X_test = dummy_data
        prep = Preprocessor(n_components=16)
        prep.fit(X_train)
        X_r = prep.transform(X_test)
        assert X_r.shape == (20, 16)

    def test_unfitted_raises(self, dummy_data: tuple) -> None:
        _, X_test = dummy_data
        prep = Preprocessor(n_components=16)
        with pytest.raises(RuntimeError, match="Call fit()"):
            prep.transform(X_test)

    def test_svd_variant(self, dummy_data: tuple) -> None:
        X_train, X_test = dummy_data
        prep = Preprocessor(n_components=10, use_svd=True)
        X_r = prep.fit_transform(X_train)
        assert X_r.shape == (100, 10)

    def test_reshape_for_cnn_square(self, dummy_data: tuple) -> None:
        X_train, _ = dummy_data
        prep = Preprocessor(n_components=16)
        X_r = prep.fit_transform(X_train)
        cnn_input = Preprocessor.reshape_for_cnn(X_r, grid_h=4, grid_w=4)
        assert cnn_input.shape == (100, 1, 4, 4)
        assert cnn_input.dtype == np.float32

    def test_reshape_for_cnn_auto(self, dummy_data: tuple) -> None:
        X_train, _ = dummy_data
        prep = Preprocessor(n_components=16)
        X_r = prep.fit_transform(X_train)
        cnn_input = Preprocessor.reshape_for_cnn(X_r)
        assert cnn_input.ndim == 4
        assert cnn_input.shape[0] == 100
        assert cnn_input.shape[1] == 1
        assert cnn_input.shape[2] * cnn_input.shape[3] == 16

    def test_reshape_mismatched_dims_raises(self, dummy_data: tuple) -> None:
        X_train, _ = dummy_data
        prep = Preprocessor(n_components=16)
        X_r = prep.fit_transform(X_train)
        with pytest.raises(ValueError, match="grid_h \\* grid_w"):
            Preprocessor.reshape_for_cnn(X_r, grid_h=3, grid_w=4)

    def test_save_and_load(self, dummy_data: tuple, tmp_path: Path) -> None:
        X_train, X_test = dummy_data
        prep = Preprocessor(n_components=8)
        X_r_before = prep.fit_transform(X_train)

        save_path = tmp_path / "prep.pkl"
        prep.save(save_path)
        loaded = Preprocessor.load(save_path)

        X_r_after = loaded.transform(X_train)
        np.testing.assert_allclose(X_r_before, X_r_after, rtol=1e-5)

    def test_load_wrong_type_raises(self, tmp_path: Path) -> None:
        import pickle
        p = tmp_path / "bad.pkl"
        with open(p, "wb") as fh:
            pickle.dump({"not": "a preprocessor"}, fh)
        with pytest.raises(TypeError, match="Expected Preprocessor"):
            Preprocessor.load(p)
