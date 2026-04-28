"""Tests for the narrowed CMOSE reproduction modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from main import _has_materialized_i3d_features, resolve_output_dir, split_cmose_records_by_usage
from src.paper_repro_data import (
    LABEL_MAP,
    SampleMeta,
    get_openface_feature_columns,
    load_cmose_metadata,
    load_i3d_dataset_matrices,
    load_i3d_matrix,
    materialize_i3d_features_from_json,
    resample_frames,
)
from src.paper_repro_model import build_model
from src.paper_repro_preprocess import fit_feature_normalizer, normalize_dataset_per_feature
from src.paper_repro_train import (
    build_loss,
    compute_class_weights,
    evaluate_predictions,
    predict,
    train_model,
)


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


def _make_i3d_npy(path: Path, *, steps: int = 12, features: int = 16) -> None:
    values = np.arange(steps * features, dtype=np.float32).reshape(steps, features)
    np.save(path, values)


def _make_labels_with_embeds(path: Path) -> None:
    payload = {
        "sample_a": {"split": "train", "label": "Engage", "agreement": 1.0, "embeds": [0.1] * 8},
        "sample_b": {"split": "test", "label": "Disengage", "agreement": 0.9, "embeds": []},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resample_frames_changes_length() -> None:
    matrix = np.arange(20, dtype=np.float32).reshape(10, 2)
    out = resample_frames(matrix, target_frames=30)
    assert out.shape == (30, 2)


def test_split_cmose_records_by_usage_treats_test_key_as_unlabeled(tmp_path: Path) -> None:
    train_record = SampleMeta(
        sample_id="video1_person0",
        base_video_id="video1",
        person_id="0",
        label_name="Engage",
        label_id=LABEL_MAP["Engage"],
        split="train",
        csv_path=tmp_path / "video1_person0.csv",
    )
    unlabeled_record = SampleMeta(
        sample_id="video2_person0",
        base_video_id="video2",
        person_id="0",
        label_name="Disengage",
        label_id=LABEL_MAP["Disengage"],
        split="test",
        csv_path=tmp_path / "video2_person0.csv",
    )

    train_records, unlabeled_records = split_cmose_records_by_usage(
        [train_record, unlabeled_record]
    )

    assert train_records == [train_record]
    assert unlabeled_records == [unlabeled_record]


def test_load_cmose_metadata_filters_by_available_csv(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _make_openface_csv(feature_dir / "video1_person0.csv", rows=10)

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "video1_person0": {"split": "train", "label": "Engage"},
                "video2_person0": {"split": "test", "label": "Disengage"},
            }
        ),
        encoding="utf-8",
    )

    records = load_cmose_metadata(labels_path, feature_dir)

    assert len(records) == 1
    assert records[0].sample_id == "video1_person0"


def test_normalize_dataset_shapes() -> None:
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


def test_get_openface_feature_columns(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    csv_path = feature_dir / "sample.csv"
    _make_openface_csv(csv_path, rows=10)

    feature_columns = get_openface_feature_columns(csv_path)
    assert len(feature_columns) == 709
    assert "frame" not in feature_columns
    assert feature_columns[0] == "f_0"


def test_load_i3d_matrix_and_dataset_alignment(tmp_path: Path) -> None:
    feature_dir = tmp_path / "i3d"
    feature_dir.mkdir()
    _make_i3d_npy(feature_dir / "sample_a.npy", steps=10, features=8)
    _make_i3d_npy(feature_dir / "sample_b.npy", steps=14, features=8)

    sample_a = load_i3d_matrix(feature_dir / "sample_a.npy", target_frames=6)
    assert sample_a.shape == (6, 8)
    assert sample_a.dtype == np.float32

    stacked = load_i3d_dataset_matrices(
        ["sample_a", "sample_b"],
        feature_dir=feature_dir,
        target_frames=5,
        progress_desc="test i3d",
    )
    assert stacked.shape == (2, 5, 8)
    assert stacked.dtype == np.float32


def test_materialize_i3d_features_from_json(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    output_dir = tmp_path / "i3d"
    _make_labels_with_embeds(labels_path)

    summary = materialize_i3d_features_from_json(labels_path, output_dir)
    assert summary["written_files"] == 2
    assert summary["embedding_dim"] == 8
    assert summary["replaced_empty_embeddings"] == 1
    assert _has_materialized_i3d_features(output_dir)

    sample_a = load_i3d_matrix(output_dir / "sample_a.npy")
    sample_b = load_i3d_matrix(output_dir / "sample_b.npy")
    assert sample_a.shape == (1, 8)
    assert sample_b.shape == (1, 8)
    np.testing.assert_allclose(sample_b, np.zeros((1, 8), dtype=np.float32))


def test_model_factory_output_shapes() -> None:
    import torch

    openface_mlp_model, openface_mlp_spec = build_model("openface_mlp", input_features=32)
    openface_mlp_out = openface_mlp_model(
        torch.from_numpy(np.random.randn(2, 20, 32).astype(np.float32))
    )
    assert openface_mlp_spec.input_kind == "openface_flat_mlp"
    assert openface_mlp_out.shape == (2, 4)

    temporal_model, temporal_spec = build_model("temporal_cnn", input_features=32)
    temporal_out = temporal_model(torch.from_numpy(np.random.randn(2, 20, 32).astype(np.float32)))
    assert temporal_spec.input_kind == "sequence"
    assert temporal_out.shape == (2, 4)

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

    i3d_mlp_model, i3d_mlp_spec = build_model("i3d_mlp", i3d_input_features=64)
    i3d_mlp_out = i3d_mlp_model(torch.from_numpy(np.random.randn(2, 15, 64).astype(np.float32)))
    assert i3d_mlp_spec.input_kind == "i3d_flat_mlp"
    assert i3d_mlp_out.shape == (2, 4)

    fusion_model, fusion_spec = build_model(
        "openface_tcn_i3d_fusion",
        input_features=32,
        i3d_input_features=64,
    )
    fusion_out = fusion_model(
        torch.from_numpy(np.random.randn(2, 15, 32).astype(np.float32)),
        torch.from_numpy(np.random.randn(2, 15, 64).astype(np.float32)),
    )
    fusion_logits, fusion_aux = fusion_model.forward_with_aux(
        torch.from_numpy(np.random.randn(2, 15, 32).astype(np.float32)),
        torch.from_numpy(np.random.randn(2, 15, 64).astype(np.float32)),
    )
    assert fusion_spec.input_kind == "multimodal_sequence"
    assert fusion_out.shape == (2, 4)
    assert fusion_logits.shape == (2, 4)
    assert fusion_aux.ndim == 0
    assert float(fusion_aux.item()) >= 0.0


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


def test_evaluate_predictions_reports_macro_accuracy_and_f1_scores() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)

    metrics = evaluate_predictions(y_true, y_pred)

    assert "accuracy" in metrics
    assert "macro_accuracy" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert 0.0 <= metrics["macro_accuracy"] <= 1.0


def test_resolve_output_dir_uses_one_folder_per_model() -> None:
    assert resolve_output_dir(
        None,
        model_name="openface_mlp",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    ) == Path("outputs/openface_mlp")
    assert resolve_output_dir(
        None,
        model_name="temporal_cnn",
        loss_name="weighted_cross_entropy",
        focal_gamma=2.0,
    ) == Path("outputs/temporal_cnn")
    assert resolve_output_dir(
        None,
        model_name="transformer",
        loss_name="focal",
        focal_gamma=1.5,
    ) == Path("outputs/transformer")
    assert resolve_output_dir(
        None,
        model_name="lstm",
        loss_name="ordinal",
        focal_gamma=2.0,
    ) == Path("outputs/lstm")
    assert resolve_output_dir(
        None,
        model_name="openface_tcn_i3d_fusion",
        loss_name="cross_entropy",
        focal_gamma=2.0,
    ) == Path("outputs/openface_tcn_i3d_fusion")


def test_train_and_predict_support_multimodal_batches(tmp_path: Path) -> None:
    import torch

    rng = np.random.default_rng(0)
    X_openface_train = rng.normal(size=(8, 12, 6)).astype(np.float32)
    X_i3d_train = rng.normal(size=(8, 12, 10)).astype(np.float32)
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

    X_openface_eval = rng.normal(size=(4, 12, 6)).astype(np.float32)
    X_i3d_eval = rng.normal(size=(4, 12, 10)).astype(np.float32)
    y_eval = np.array([0, 1, 2, 3], dtype=np.int64)

    model, _ = build_model(
        "openface_tcn_i3d_fusion",
        input_features=6,
        i3d_input_features=10,
    )

    history = train_model(
        model,
        (X_openface_train, X_i3d_train),
        y_train,
        (X_openface_eval, X_i3d_eval),
        y_eval,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        patience=2,
        loss_name="cross_entropy",
        focal_gamma=2.0,
        checkpoint_path=tmp_path / "fusion_model.pth",
        device=torch.device("cpu"),
        num_workers=0,
        use_amp=False,
    )
    assert history["best_epoch"] >= 1
    assert len(history["eval_macro_accuracies"]) == len(history["eval_losses"])
    assert len(history["eval_f1_macros"]) == len(history["eval_losses"])
    assert len(history["eval_f1_weighteds"]) == len(history["eval_losses"])

    preds = predict(
        model,
        (X_openface_eval, X_i3d_eval),
        batch_size=2,
        device=torch.device("cpu"),
        num_workers=0,
        use_amp=False,
    )
    assert preds.shape == (4,)


def test_train_and_predict_support_flat_mlp_batches(tmp_path: Path) -> None:
    import torch

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(8, 12, 6)).astype(np.float32)
    y_train = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    X_eval = rng.normal(size=(4, 12, 6)).astype(np.float32)
    y_eval = np.array([0, 1, 2, 3], dtype=np.int64)

    model, _ = build_model("openface_mlp", input_features=6)

    history = train_model(
        model,
        X_train,
        y_train,
        X_eval,
        y_eval,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        patience=2,
        loss_name="cross_entropy",
        focal_gamma=2.0,
        checkpoint_path=tmp_path / "openface_mlp_model.pth",
        device=torch.device("cpu"),
        num_workers=0,
        use_amp=False,
    )
    assert history["best_epoch"] >= 1
    assert len(history["eval_macro_accuracies"]) == len(history["eval_losses"])
    assert len(history["eval_f1_macros"]) == len(history["eval_losses"])
    assert len(history["eval_f1_weighteds"]) == len(history["eval_losses"])

    preds = predict(
        model,
        X_eval,
        batch_size=2,
        device=torch.device("cpu"),
        num_workers=0,
        use_amp=False,
    )
    assert preds.shape == (4,)
