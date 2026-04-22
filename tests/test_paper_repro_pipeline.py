"""Tests for the paper-faithful CMOSE reproduction modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from main import resolve_output_dir
from src.paper_repro_data import (
    load_cmose_metadata,
    resample_frames,
    select_paper_style_subset,
)
from src.paper_repro_model import build_model
from src.paper_repro_preprocess import (
    add_channel_dim,
    fit_feature_normalizer,
    normalize_dataset_per_feature,
)
from src.paper_repro_train import build_loss, compute_class_weights


def _make_openface_csv(path: Path, *, rows: int = 20) -> None:
    meta_cols = ["frame", "face_id", "timestamp", "confidence", "success"]
    feature_cols = [f"f_{i}" for i in range(709)]
    records = []
    for frame in range(rows):
        base = {
            "frame": frame,
            "face_id": 0,
            "timestamp": frame / 30.0,
            "confidence": 0.8 + (frame % 3) * 0.05,
            "success": 1,
        }
        row = {**base, **{name: float(frame + idx) for idx, name in enumerate(feature_cols)}}
        records.append(row)
    pd.DataFrame.from_records(records, columns=meta_cols + feature_cols).to_csv(path, index=False)


def test_resample_frames_changes_length() -> None:
    matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
    out = resample_frames(matrix, target_frames=30)
    assert out.shape == (30, 2)


def test_select_paper_style_subset_balances_classes(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels = {}
    specs = {
        "Highly Disengage": [("video1_1_person0", "train"), ("video1_2_person0", "test")],
        "Disengage": [
            ("video2_1_person0", "train"),
            ("video2_2_person0", "train"),
            ("video2_3_person0", "test"),
        ],
        "Engage": [
            ("video3_1_person0", "train"),
            ("video3_2_person0", "train"),
            ("video3_3_person0", "test"),
            ("video3_4_person0", "test"),
        ],
        "Highly Engage": [
            ("video4_1_person0", "train"),
            ("video4_2_person0", "test"),
            ("video4_3_person0", "test"),
        ],
    }
    for label, items in specs.items():
        for sample_id, split in items:
            labels[sample_id] = {"label": label, "split": split}
            _make_openface_csv(feature_dir / f"{sample_id}.csv", rows=10)

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")

    records = load_cmose_metadata(labels_path, feature_dir)
    selected = select_paper_style_subset(records)
    counts = {}
    for record in selected:
        counts[record.label_name] = counts.get(record.label_name, 0) + 1

    assert counts["Highly Disengage"] == 2
    assert counts["Disengage"] == 2
    assert counts["Engage"] == 2
    assert counts["Highly Engage"] == 2


def test_normalize_dataset_and_add_channel_dim_shapes() -> None:
    rng = np.random.default_rng(0)
    train_samples = rng.normal(size=(8, 16, 16)).astype(np.float32)
    mean, std = fit_feature_normalizer(train_samples)

    assert mean.shape == (16,)
    assert std.shape == (16,)
    assert np.all(std > 0.0)

    normalized = normalize_dataset_per_feature(
        train_samples,
        mean=mean,
        std=std,
        progress_desc="test",
    )
    assert normalized.shape == (8, 16, 16)
    np.testing.assert_allclose(
        normalized.mean(axis=(0, 1)),
        np.zeros(16, dtype=np.float32),
        atol=1e-5,
    )

    cnn_input = add_channel_dim(normalized)
    assert cnn_input.shape == (8, 1, 16, 16)


def test_model_factory_output_shapes() -> None:
    import torch

    square_model, square_spec = build_model("paper_cnn", input_size=32)
    square_out = square_model(torch.from_numpy(np.random.randn(2, 1, 32, 32).astype(np.float32)))
    assert square_spec.input_kind == "square_matrix"
    assert square_out.shape == (2, 4)

    temporal_model, temporal_spec = build_model("temporal_cnn", input_features=32)
    temporal_out = temporal_model(torch.from_numpy(np.random.randn(2, 20, 32).astype(np.float32)))
    assert temporal_spec.input_kind == "sequence"
    assert temporal_out.shape == (2, 4)

    rectangular_model, rectangular_spec = build_model("rectangular_cnn")
    rectangular_out = rectangular_model(
        torch.from_numpy(np.random.randn(2, 1, 20, 32).astype(np.float32))
    )
    assert rectangular_spec.input_kind == "frame_feature_map"
    assert rectangular_out.shape == (2, 4)

    lstm_model, lstm_spec = build_model("lstm", input_features=32)
    lstm_out = lstm_model(torch.from_numpy(np.random.randn(2, 20, 32).astype(np.float32)))
    assert lstm_spec.input_kind == "sequence"
    assert lstm_out.shape == (2, 4)

    transformer_model, transformer_spec = build_model("transformer", input_features=32)
    transformer_out = transformer_model(
        torch.from_numpy(np.random.randn(2, 20, 32).astype(np.float32))
    )
    assert transformer_spec.input_kind == "sequence"
    assert transformer_out.shape == (2, 4)


def test_loss_factory_builds_weighted_focal_and_ordinal_losses() -> None:
    import torch

    y_train = np.array([0, 0, 1, 2, 2, 2, 3], dtype=np.int64)
    weights = compute_class_weights(y_train)

    assert weights.shape == (4,)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[1] > weights[2]

    weighted_ce, ce_weights = build_loss(
        y_train,
        loss_name="weighted_cross_entropy",
        focal_gamma=2.0,
        device=torch.device("cpu"),
    )
    assert isinstance(weighted_ce, torch.nn.CrossEntropyLoss)
    assert ce_weights is not None
    np.testing.assert_allclose(np.array(ce_weights), weights)

    focal_loss, focal_weights = build_loss(
        y_train,
        loss_name="focal",
        focal_gamma=2.0,
        device=torch.device("cpu"),
    )
    assert focal_loss.__class__.__name__ == "FocalLoss"
    assert focal_weights is not None

    ordinal_loss, ordinal_weights = build_loss(
        y_train,
        loss_name="ordinal",
        focal_gamma=2.0,
        device=torch.device("cpu"),
    )
    assert ordinal_loss.__class__.__name__ == "OrdinalEMDLoss"
    assert ordinal_weights is not None


def test_resolve_output_dir_uses_loss_specific_default_paths() -> None:
    assert resolve_output_dir(
        None,
        model_name="temporal_cnn",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    ) == Path("outputs/temporal_cnn_cross_entropy")
    assert resolve_output_dir(
        None,
        model_name="temporal_cnn",
        loss_name="weighted_cross_entropy",
        focal_gamma=2.0,
    ) == Path("outputs/temporal_cnn_weighted_cross_entropy")
    assert resolve_output_dir(
        None,
        model_name="transformer",
        loss_name="focal",
        focal_gamma=1.5,
    ) == Path("outputs/transformer_focal_g1p5")
    assert resolve_output_dir(
        None,
        model_name="lstm",
        loss_name="ordinal",
        focal_gamma=2.0,
    ) == Path("outputs/lstm_ordinal")
