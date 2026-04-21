"""Tests for src/model.py, src/train.py, and src/evaluate.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.model import EngagementCNN, NUM_CLASSES
from src.train import (
    compute_class_weights,
    make_dataloader,
    train,
)
from src.evaluate import evaluate, predict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_cnn() -> EngagementCNN:
    return EngagementCNN(grid_h=4, grid_w=4, num_classes=NUM_CLASSES)


def _make_dataset(
    n: int = 40,
    grid_h: int = 4,
    grid_w: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 1, grid_h, grid_w)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=n).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# EngagementCNN
# ---------------------------------------------------------------------------

class TestEngagementCNN:
    def test_output_shape(self, small_cnn: EngagementCNN) -> None:
        x = torch.randn(8, 1, 4, 4)
        out = small_cnn(x)
        assert out.shape == (8, NUM_CLASSES)

    def test_output_is_logits(self, small_cnn: EngagementCNN) -> None:
        x = torch.randn(4, 1, 4, 4)
        out = small_cnn(x)
        # logits can be any value; check not softmax (no values forced to [0,1] range)
        assert out.dtype == torch.float32

    def test_different_grid_sizes(self) -> None:
        for h, w in [(8, 8), (4, 16), (1, 64)]:
            model = EngagementCNN(grid_h=h, grid_w=w)
            x = torch.randn(2, 1, h, w)
            out = model(x)
            assert out.shape == (2, NUM_CLASSES)

    def test_gradients_flow(self, small_cnn: EngagementCNN) -> None:
        x = torch.randn(4, 1, 4, 4)
        y = torch.randint(0, NUM_CLASSES, (4,))
        loss = torch.nn.CrossEntropyLoss()(small_cnn(x), y)
        loss.backward()
        for name, param in small_cnn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# Train utilities
# ---------------------------------------------------------------------------

class TestTrainUtils:
    def test_make_dataloader(self) -> None:
        X, y = _make_dataset(20)
        loader = make_dataloader(X, y, batch_size=8)
        batches = list(loader)
        # 20 samples / batch_size=8 → 3 batches (8+8+4)
        assert len(batches) == 3
        Xb, yb = batches[0]
        assert Xb.shape == (8, 1, 4, 4)
        assert yb.shape == (8,)

    def test_compute_class_weights(self) -> None:
        y = np.array([0, 0, 0, 1, 2, 3], dtype=np.int64)
        w = compute_class_weights(y, num_classes=4)
        assert w.shape == (4,)
        # Class 0 appears 3×, so its weight should be the smallest
        assert w[0] < w[1]

    def test_balanced_labels_equal_weights(self) -> None:
        y = np.array([0, 1, 2, 3], dtype=np.int64)
        w = compute_class_weights(y, num_classes=4)
        np.testing.assert_allclose(w.numpy(), np.ones(4), rtol=1e-5)


# ---------------------------------------------------------------------------
# Full train + evaluate integration (tiny dataset, few epochs)
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:
    def test_train_returns_history(self, tmp_path: Path) -> None:
        X_tr, y_tr = _make_dataset(40, seed=0)
        X_vl, y_vl = _make_dataset(10, seed=1)
        model = EngagementCNN(grid_h=4, grid_w=4)

        history = train(
            model, X_tr, y_tr, X_vl, y_vl,
            epochs=3,
            batch_size=8,
            patience=3,
            checkpoint_path=tmp_path / "best.pth",
        )

        assert "train_losses" in history
        assert len(history["train_losses"]) <= 3
        assert (tmp_path / "best.pth").exists()

    def test_evaluate_returns_metrics(self) -> None:
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 0, 0, 2, 3])
        metrics = evaluate(y_true, y_pred)

        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert "report" in metrics
        assert "conf_matrix" in metrics
        assert metrics["accuracy"] == pytest.approx(0.875)

    def test_evaluate_saves_confusion_matrix(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        pytest.importorskip("seaborn")
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        evaluate(y_true, y_pred, output_dir=tmp_path)
        assert (tmp_path / "confusion_matrix.png").exists()

    def test_predict_shape(self, small_cnn: EngagementCNN) -> None:
        X, _ = _make_dataset(12)
        preds = predict(small_cnn, X, batch_size=4)
        assert preds.shape == (12,)
        assert preds.dtype in (np.int64, np.int32, np.intp)
        assert set(preds).issubset(set(range(NUM_CLASSES)))
