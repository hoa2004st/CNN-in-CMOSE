"""Tests for src/data_loader.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import (
    _load_openface_csv,
    aggregate_clip_features,
    load_split,
    DEFAULT_FEATURE_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def openface_csv(tmp_path: Path) -> Path:
    """Write a minimal OpenFace CSV and return its path."""
    cols = ["frame", "timestamp", "confidence", "success"] + DEFAULT_FEATURE_COLS
    header = ",".join(cols)
    n_frames = 10
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        vals = [i, i * 0.033, 0.95, 1]
        vals += rng.standard_normal(len(DEFAULT_FEATURE_COLS)).tolist()
        rows.append(",".join(str(v) for v in vals))
    content = header + "\n" + "\n".join(rows) + "\n"
    p = tmp_path / "clip001.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def cmose_root(tmp_path: Path, openface_csv: Path) -> Path:
    """Create a minimal CMOSE directory structure."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    of_dir = tmp_path / "openface"
    of_dir.mkdir()

    # Copy the sample OpenFace CSV into the openface directory
    (of_dir / "clip001.csv").write_bytes(openface_csv.read_bytes())

    for split in ("train", "val", "test"):
        (labels_dir / f"{split}_labels.csv").write_text("clip_id,label\nclip001,2\n")

    return tmp_path


# ---------------------------------------------------------------------------
# _load_openface_csv
# ---------------------------------------------------------------------------

class TestLoadOpenfaceCSV:
    def test_returns_array_on_valid_file(self, openface_csv: Path) -> None:
        arr = _load_openface_csv(openface_csv, DEFAULT_FEATURE_COLS)
        assert arr is not None
        assert arr.ndim == 2
        assert arr.shape[1] == len(DEFAULT_FEATURE_COLS)

    def test_filters_low_confidence(self, tmp_path: Path) -> None:
        cols = ["frame", "confidence", "success"] + DEFAULT_FEATURE_COLS[:3]
        row = ",".join(["0", "0.5", "1"] + ["0"] * len(DEFAULT_FEATURE_COLS[:3]))
        content = ",".join(cols) + "\n" + row + "\n"
        p = tmp_path / "low_conf.csv"
        p.write_text(content)
        arr = _load_openface_csv(p, DEFAULT_FEATURE_COLS[:3])
        assert arr is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        arr = _load_openface_csv(tmp_path / "nonexistent.csv", DEFAULT_FEATURE_COLS)
        assert arr is None

    def test_filters_failed_tracking(self, tmp_path: Path) -> None:
        cols = ["frame", "confidence", "success"] + DEFAULT_FEATURE_COLS[:2]
        row = ",".join(["0", "0.95", "0"] + ["0.0"] * len(DEFAULT_FEATURE_COLS[:2]))
        content = ",".join(cols) + "\n" + row + "\n"
        p = tmp_path / "fail.csv"
        p.write_text(content)
        arr = _load_openface_csv(p, DEFAULT_FEATURE_COLS[:2])
        assert arr is None


# ---------------------------------------------------------------------------
# aggregate_clip_features
# ---------------------------------------------------------------------------

class TestAggregateClipFeatures:
    @pytest.fixture()
    def frames(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.standard_normal((20, 10)).astype(np.float32)

    def test_mean(self, frames: np.ndarray) -> None:
        out = aggregate_clip_features(frames, stat="mean")
        np.testing.assert_allclose(out, frames.mean(axis=0))

    def test_std(self, frames: np.ndarray) -> None:
        out = aggregate_clip_features(frames, stat="std")
        np.testing.assert_allclose(out, frames.std(axis=0))

    def test_mean_std(self, frames: np.ndarray) -> None:
        out = aggregate_clip_features(frames, stat="mean_std")
        assert out.shape == (frames.shape[1] * 2,)
        np.testing.assert_allclose(out[: frames.shape[1]], frames.mean(axis=0))
        np.testing.assert_allclose(out[frames.shape[1] :], frames.std(axis=0))

    def test_max(self, frames: np.ndarray) -> None:
        out = aggregate_clip_features(frames, stat="max")
        np.testing.assert_allclose(out, frames.max(axis=0))

    def test_invalid_stat_raises(self, frames: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation stat"):
            aggregate_clip_features(frames, stat="invalid")


# ---------------------------------------------------------------------------
# load_split
# ---------------------------------------------------------------------------

class TestLoadSplit:
    def test_loads_train_split(self, cmose_root: Path) -> None:
        X, y, clip_ids = load_split(cmose_root, "train")
        assert X.ndim == 2
        assert len(y) == len(clip_ids)
        assert y[0] == 2

    def test_loads_all_splits(self, cmose_root: Path) -> None:
        for split in ("train", "val", "test"):
            X, y, ids = load_split(cmose_root, split)
            assert len(X) > 0
            assert len(X) == len(y) == len(ids)

    def test_missing_labels_raises(self, tmp_path: Path) -> None:
        (tmp_path / "openface").mkdir()
        (tmp_path / "labels").mkdir()
        with pytest.raises(FileNotFoundError, match="Labels file"):
            load_split(tmp_path, "train")

    def test_missing_openface_dir_raises(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "train_labels.csv").write_text("clip_id,label\nclip001,1\n")
        with pytest.raises(FileNotFoundError, match="OpenFace feature directory"):
            load_split(tmp_path, "train")

    def test_bad_columns_raises(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "train_labels.csv").write_text("video,engagement\nclip001,1\n")
        (tmp_path / "openface").mkdir()
        with pytest.raises(ValueError, match="'clip_id' and 'label'"):
            load_split(tmp_path, "train")

    def test_aggregation_option(self, cmose_root: Path) -> None:
        X_ms, _, _ = load_split(cmose_root, "train", aggregation="mean_std")
        X_m, _, _ = load_split(cmose_root, "train", aggregation="mean")
        # mean_std should have twice the features of mean
        assert X_ms.shape[1] == X_m.shape[1] * 2
