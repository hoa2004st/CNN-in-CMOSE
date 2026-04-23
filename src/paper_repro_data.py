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
import torch
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


def resolve_feature_indices(
    feature_columns: list[str],
    *,
    exact_names: list[str] | None = None,
    prefixes: list[str] | None = None,
) -> list[int]:
    """Resolve feature indices by exact names and/or prefixes while preserving order."""
    exact_names = exact_names or []
    prefixes = prefixes or []
    exact_name_set = set(exact_names)

    indices: list[int] = []
    for idx, column in enumerate(feature_columns):
        if column in exact_name_set or any(column.startswith(prefix) for prefix in prefixes):
            indices.append(idx)
    return indices


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


def resolve_i3d_feature_path(
    sample_id: str,
    feature_dir: str | Path,
    *,
    allowed_suffixes: tuple[str, ...] = (".npy", ".npz", ".pt"),
) -> Path:
    """Resolve the precomputed I3D feature file for one sample id."""
    feature_dir = Path(feature_dir)
    for suffix in allowed_suffixes:
        candidate = feature_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing I3D feature file for sample_id={sample_id!r} in {feature_dir}"
    )


def _coerce_i3d_array(array: np.ndarray, *, source_path: Path) -> np.ndarray:
    """Convert loaded I3D arrays to a stable ``time x features`` float32 matrix."""
    matrix = np.asarray(array)
    matrix = np.squeeze(matrix)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D I3D feature matrix in {source_path}, got {matrix.shape}")
    if min(matrix.shape) <= 0:
        raise ValueError(f"Invalid empty I3D feature matrix in {source_path}: {matrix.shape}")

    # I3D embeddings are usually ``time x feature_dim`` with feature_dim much
    # larger than the temporal length. If the opposite is true, assume the file
    # was saved as ``feature_dim x time`` and transpose it.
    if matrix.shape[0] >= matrix.shape[1] * 4:
        matrix = matrix.T
    return matrix.astype(np.float32, copy=False)


def load_i3d_matrix(
    feature_path: str | Path,
    *,
    target_frames: int | None = None,
) -> np.ndarray:
    """Load one precomputed I3D feature file as a ``time x features`` matrix."""
    feature_path = Path(feature_path)

    if feature_path.suffix == ".npy":
        raw = np.load(feature_path, allow_pickle=False)
    elif feature_path.suffix == ".npz":
        archive = np.load(feature_path, allow_pickle=False)
        if "features" in archive:
            raw = archive["features"]
        elif len(archive.files) == 1:
            raw = archive[archive.files[0]]
        else:
            raise ValueError(
                f"Ambiguous NPZ I3D archive at {feature_path}; expected a single array or 'features'"
            )
    elif feature_path.suffix == ".pt":
        raw = torch.load(feature_path, map_location="cpu")
        if isinstance(raw, dict):
            if "features" in raw:
                raw = raw["features"]
            elif len(raw) == 1:
                raw = next(iter(raw.values()))
            else:
                raise ValueError(
                    f"Ambiguous PT I3D tensor container at {feature_path}; expected key 'features'"
                )
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported I3D feature file type: {feature_path.suffix}")

    matrix = _coerce_i3d_array(np.asarray(raw), source_path=feature_path)
    if target_frames is not None:
        return resample_frames(matrix, target_frames=target_frames)
    return matrix


def load_i3d_dataset_matrices(
    sample_ids: list[str],
    *,
    feature_dir: str | Path,
    target_frames: int,
    progress_desc: str | None = None,
) -> np.ndarray:
    """Load aligned precomputed I3D features for a list of sample ids."""
    matrices = [
        load_i3d_matrix(
            resolve_i3d_feature_path(sample_id, feature_dir),
            target_frames=target_frames,
        )
        for sample_id in tqdm(
            sample_ids,
            desc=progress_desc or "Loading I3D features",
            unit="sample",
            leave=False,
        )
    ]
    return np.stack(matrices, axis=0).astype(np.float32, copy=False)


def materialize_i3d_features_from_json(
    labels_path: str | Path,
    output_dir: str | Path,
) -> dict[str, int | str]:
    """Write per-sample I3D feature files from ``final_data_1.json`` embeds."""
    labels_path = Path(labels_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    embed_dim = 0
    for meta in raw.values():
        embeds = meta.get("embeds")
        if isinstance(embeds, list) and embeds:
            embed_dim = len(embeds)
            break
    if embed_dim <= 0:
        raise ValueError(f"Could not infer I3D embedding dimension from {labels_path}")

    written = 0
    replaced_empty = 0
    skipped_invalid = 0
    for sample_id, meta in raw.items():
        embeds = meta.get("embeds")
        if not isinstance(embeds, list):
            skipped_invalid += 1
            continue
        if len(embeds) == 0:
            vector = np.zeros((embed_dim,), dtype=np.float32)
            replaced_empty += 1
        else:
            if len(embeds) != embed_dim:
                raise ValueError(
                    f"Inconsistent I3D embedding length for {sample_id}: "
                    f"expected {embed_dim}, found {len(embeds)}"
                )
            vector = np.asarray(embeds, dtype=np.float32)
        np.save(output_dir / f"{sample_id}.npy", vector)
        written += 1

    return {
        "source_json": str(labels_path),
        "output_dir": str(output_dir),
        "embedding_dim": embed_dim,
        "written_files": written,
        "replaced_empty_embeddings": replaced_empty,
        "skipped_invalid_records": skipped_invalid,
    }
