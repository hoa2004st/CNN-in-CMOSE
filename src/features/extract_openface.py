"""CMOSE data loading for the narrowed OpenFace/I3D comparison pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


LABEL_MAP = {
    "Highly Disengage": 0,
    "Disengage": 1,
    "Engage": 2,
    "Highly Engage": 3,
}

ID_TO_LABEL = {value: key for key, value in LABEL_MAP.items()}

OPENFACE_META_COLS = ["frame", "face_id", "timestamp", "confidence", "success"]
OPENFACE_BASELINE_GROUPS = {
    "gaze": [
        "gaze_0_x",
        "gaze_0_y",
        "gaze_0_z",
        "gaze_1_x",
        "gaze_1_y",
        "gaze_1_z",
        "gaze_angle_x",
        "gaze_angle_y",
    ],
    "head_pose": ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"],
    "au_intensity": [
        "AU01_r",
        "AU02_r",
        "AU04_r",
        "AU05_r",
        "AU06_r",
        "AU07_r",
        "AU09_r",
        "AU10_r",
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "AU17_r",
        "AU20_r",
        "AU23_r",
        "AU25_r",
        "AU26_r",
        "AU45_r",
    ],
    "au_presence": [
        "AU01_c",
        "AU02_c",
        "AU04_c",
        "AU05_c",
        "AU06_c",
        "AU07_c",
        "AU09_c",
        "AU10_c",
        "AU12_c",
        "AU14_c",
        "AU15_c",
        "AU17_c",
        "AU20_c",
        "AU23_c",
        "AU25_c",
        "AU26_c",
        "AU28_c",
        "AU45_c",
    ],
}
OPENFACE_BASELINE_FEATURE_COLS = [
    *OPENFACE_BASELINE_GROUPS["gaze"],
    *OPENFACE_BASELINE_GROUPS["head_pose"],
    *OPENFACE_BASELINE_GROUPS["au_intensity"],
    *OPENFACE_BASELINE_GROUPS["au_presence"],
]


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    base_video_id: str
    person_id: str
    label_name: str
    label_id: int
    split: str
    csv_path: Path


def load_cmose_metadata(
    labels_path: str | Path,
    feature_dir: str | Path,
    *,
    allowed_splits: Iterable[str] = ("train", "test"),
) -> list[SampleMeta]:
    """Load CMOSE labels and align them with extracted OpenFace CSV files."""
    labels_path = Path(labels_path)
    feature_dir = Path(feature_dir)
    allowed_splits = set(allowed_splits)

    def resolve_csv_path(sample_id: str) -> Path | None:
        direct = feature_dir / f"{sample_id}.csv"
        if direct.exists():
            return direct
        nested = feature_dir / "secondFeature" / f"{sample_id}.csv"
        if nested.exists():
            return nested
        return None

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    records: list[SampleMeta] = []
    for sample_id, meta in raw.items():
        split = meta.get("split")
        label_name = meta.get("label")
        if split not in allowed_splits or label_name not in LABEL_MAP:
            continue

        csv_path = resolve_csv_path(sample_id)
        if csv_path is None:
            continue

        base_video_id, person_suffix = sample_id.rsplit("_person", 1)
        records.append(
            SampleMeta(
                sample_id=sample_id,
                base_video_id=base_video_id,
                person_id=person_suffix,
                label_name=label_name,
                label_id=LABEL_MAP[label_name],
                split=split,
                csv_path=csv_path,
            )
        )
    return records


def describe_selection(records: list[SampleMeta]) -> dict[str, dict[str, int]]:
    """Return simple counts for reporting and debugging."""
    label_counts = Counter(record.label_name for record in records)
    split_counts = Counter(record.split for record in records)
    base_counts = Counter(record.base_video_id for record in records)
    return {
        "labels": dict(label_counts),
        "splits": dict(split_counts),
        "base_videos": {"unique": len(base_counts)},
        "samples": {"total": len(records)},
    }


def get_openface_feature_columns(csv_path: str | Path) -> list[str]:
    """Return the ordered OpenFace feature columns for one CMOSE CSV."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, nrows=0)
    df.columns = df.columns.str.strip()

    if not set(OPENFACE_META_COLS).issubset(df.columns):
        raise ValueError(f"Missing OpenFace metadata columns in {csv_path}")

    feature_cols = [column for column in df.columns if column not in OPENFACE_META_COLS]
    if len(feature_cols) != 709:
        raise ValueError(
            f"Expected 709 OpenFace features in {csv_path}, found {len(feature_cols)}"
        )
    return feature_cols


def load_openface_matrix(
    csv_path: str | Path,
    *,
    target_frames: int = 300,
) -> np.ndarray:
    """Load one CMOSE/OpenFace CSV as a fixed-size frame-feature matrix.

    The first five columns are treated as metadata, leaving the 709 OpenFace
    features described in the paper summary.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if not set(OPENFACE_META_COLS).issubset(df.columns):
        raise ValueError(f"Missing OpenFace metadata columns in {csv_path}")

    # Paper stage 2: when multiple detections exist in one frame, keep the
    # highest-confidence row.
    frame_best_idx = (
        df.sort_values(["frame", "confidence"], ascending=[True, False])
        .groupby("frame", sort=False)["confidence"]
        .idxmax()
        .to_numpy()
    )
    df = df.loc[frame_best_idx].sort_values("frame").reset_index(drop=True).copy()

    feature_cols = get_openface_feature_columns(csv_path)
    if len(feature_cols) != 709:
        raise ValueError(
            f"Expected 709 OpenFace features in {csv_path}, found {len(feature_cols)}"
        )

    matrix = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    return resample_frames(matrix, target_frames=target_frames)


def resample_frames(matrix: np.ndarray, *, target_frames: int = 300) -> np.ndarray:
    """Resample a variable-length frame sequence to a fixed frame count."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {matrix.shape}")

    n_frames, n_features = matrix.shape
    if n_frames == target_frames:
        return matrix.astype(np.float32, copy=False)
    if n_frames == 0:
        raise ValueError("Cannot resample an empty frame matrix")
    if n_frames == 1:
        return np.repeat(matrix.astype(np.float32), target_frames, axis=0)

    source_positions = np.linspace(0.0, 1.0, num=n_frames, dtype=np.float64)
    target_positions = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float64)

    resampled = np.empty((target_frames, n_features), dtype=np.float32)
    for feature_idx in range(n_features):
        resampled[:, feature_idx] = np.interp(
            target_positions,
            source_positions,
            matrix[:, feature_idx].astype(np.float64),
        )
    return resampled


def load_dataset_matrices(
    records: list[SampleMeta],
    *,
    target_frames: int = 300,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all selected samples into a 3-D array."""
    if not records:
        raise ValueError("No records provided to load_dataset_matrices")
    matrices = [
        load_openface_matrix(record.csv_path, target_frames=target_frames)
        for record in tqdm(
            records,
            desc=progress_desc or "Loading samples",
            unit="sample",
            leave=False,
        )
    ]
    sample_ids = [record.sample_id for record in records]
    labels = np.array([record.label_id for record in records], dtype=np.int64)
    return np.stack(matrices, axis=0), labels, sample_ids


def select_openface_baseline_features(df: pd.DataFrame) -> np.ndarray:
    """Select paper-aligned 49 OpenFace features in a stable order."""
    missing_cols = [name for name in OPENFACE_BASELINE_FEATURE_COLS if name not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required OpenFace baseline columns: "
            + ", ".join(missing_cols[:8])
            + ("..." if len(missing_cols) > 8 else "")
        )
    matrix = df[OPENFACE_BASELINE_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    if matrix.shape[1] != 49:
        raise ValueError(f"Expected 49 baseline OpenFace features, got {matrix.shape[1]}")
    return matrix


def chunk_openface_baseline(frames: np.ndarray, *, chunk_count: int = 10) -> np.ndarray:
    """Convert frame-level 49-d OpenFace features to paper chunk stats (147, T)."""
    if frames.ndim != 2 or frames.shape[1] != 49:
        raise ValueError(f"Expected OpenFace frame matrix with shape (n, 49), got {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("Cannot chunk an empty OpenFace matrix")

    standard_frames = 250
    while frames.shape[0] < standard_frames:
        frames = np.concatenate([frames, frames], axis=0)
    frames = frames[:standard_frames].astype(np.float32, copy=False)

    chunks = np.array_split(frames, int(chunk_count), axis=0)
    stats: list[np.ndarray] = []
    for chunk in chunks:
        stats.append(
            np.concatenate(
                [
                    chunk.min(axis=0),
                    chunk.max(axis=0),
                    chunk.var(axis=0),
                ],
                axis=0,
            )
        )
    return np.stack(stats, axis=1).astype(np.float32, copy=False)


def load_baseline_openface_chunk_matrix(
    csv_path: str | Path,
    *,
    chunk_count: int = 10,
) -> np.ndarray:
    """Load one OpenFace CSV into paper baseline chunk tensor of shape (147, T)."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if not set(OPENFACE_META_COLS).issubset(df.columns):
        raise ValueError(f"Missing OpenFace metadata columns in {csv_path}")

    frame_best_idx = (
        df.sort_values(["frame", "confidence"], ascending=[True, False])
        .groupby("frame", sort=False)["confidence"]
        .idxmax()
        .to_numpy()
    )
    df = df.loc[frame_best_idx].sort_values("frame").reset_index(drop=True).copy()
    frame_matrix = select_openface_baseline_features(df)
    return chunk_openface_baseline(frame_matrix, chunk_count=chunk_count)


def load_baseline_openface_dataset_matrices(
    records: list[SampleMeta],
    *,
    chunk_count: int = 10,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all selected samples as baseline OpenFace chunk tensors (N, 147, T)."""
    if not records:
        raise ValueError("No records provided to load_baseline_openface_dataset_matrices")
    matrices = [
        load_baseline_openface_chunk_matrix(record.csv_path, chunk_count=chunk_count)
        for record in tqdm(
            records,
            desc=progress_desc or "Loading baseline OpenFace chunks",
            unit="sample",
            leave=False,
        )
    ]
    sample_ids = [record.sample_id for record in records]
    labels = np.array([record.label_id for record in records], dtype=np.int64)
    return np.stack(matrices, axis=0).astype(np.float32, copy=False), labels, sample_ids
