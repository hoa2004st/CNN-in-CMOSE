"""CMOSE data loading for the paper-faithful reproduction pipeline.

This module adapts the DAiSEE-oriented paper procedure to the CMOSE feature
dump in ``data/CMOSE/secondFeature/secondFeature``:

* labels come from ``final_data_1.json``
* each CSV already represents one person track within a base video
* the paper's pre-balancing selection is approximated by selecting complete
  base-video groups in ascending group-size order until reaching the minority
  class count
* every sample is converted to a fixed ``target_frames x 709`` matrix
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
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

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    records: list[SampleMeta] = []
    for sample_id, meta in raw.items():
        split = meta.get("split")
        label_name = meta.get("label")
        if split not in allowed_splits or label_name not in LABEL_MAP:
            continue

        csv_path = feature_dir / f"{sample_id}.csv"
        if not csv_path.exists():
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


def select_paper_style_subset(records: list[SampleMeta]) -> list[SampleMeta]:
    """Approximate the paper's identity-based selection using CMOSE base videos.

    The original paper selects groups in ascending size until reaching the
    minority-class count. For CMOSE the closest stable grouping is the base
    video id, because each sample file already refers to one tracked person.
    """
    by_label: dict[int, list[SampleMeta]] = defaultdict(list)
    for record in records:
        by_label[record.label_id].append(record)

    if not by_label:
        return []

    minority_count = min(len(items) for items in by_label.values())
    selected: list[SampleMeta] = []

    for label_id in sorted(by_label):
        grouped: dict[str, list[SampleMeta]] = defaultdict(list)
        for record in by_label[label_id]:
            grouped[record.base_video_id].append(record)

        ordered_groups = sorted(grouped.items(), key=lambda item: (len(item[1]), item[0]))
        running = 0
        for _, group_records in ordered_groups:
            if running >= minority_count:
                break
            selected.extend(group_records)
            running += len(group_records)

    return selected


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
    df = (
        df.sort_values(["frame", "confidence"], ascending=[True, False])
        .groupby("frame", as_index=False)
        .first()
        .sort_values("frame")
    )

    feature_cols = [column for column in df.columns if column not in OPENFACE_META_COLS]
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
