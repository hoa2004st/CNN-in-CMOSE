"""Tests for the paper-faithful CMOSE reproduction modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.paper_repro_data import (
    load_cmose_metadata,
    resample_frames,
    select_paper_style_subset,
)
from src.paper_repro_model import PaperEngagementCNN
from src.paper_repro_preprocess import (
    flatten_matrices,
    minmax_normalize_per_sample,
    reshape_flattened_samples,
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


def test_minmax_flatten_and_reshape_shapes() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(16, 16)).astype(np.float32)
    processed = minmax_normalize_per_sample(matrix)
    assert processed.shape == (16, 16)
    assert np.all((processed >= 0.0) & (processed <= 1.0))

    samples = np.stack([processed + idx for idx in range(8)], axis=0)
    X_flat = flatten_matrices(samples)
    assert X_flat.shape == (8, 256)

    cnn_input = reshape_flattened_samples(X_flat, side=16)
    assert cnn_input.shape[1:] == (1, 16, 16)


def test_paper_model_output_shape() -> None:
    model = PaperEngagementCNN(input_size=32)
    x = np.random.randn(2, 1, 32, 32).astype(np.float32)
    import torch

    out = model(torch.from_numpy(x))
    assert out.shape == (2, 4)
