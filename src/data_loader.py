"""Data loading utilities for the CMOSE dataset.

The CMOSE (Comprehensive Multimodal Open Student Engagement) dataset provides
per-frame OpenFace features extracted from student video recordings, together
with engagement labels (0–3).

Expected dataset layout
-----------------------
<cmose_root>/
    labels/
        train_labels.csv   # columns: clip_id, label
        val_labels.csv
        test_labels.csv
    openface/
        <clip_id>.csv      # per-frame OpenFace features for one clip

Each OpenFace CSV row corresponds to one video frame and contains columns such
as ``frame``, ``timestamp``, ``confidence``, ``success``, gaze angles, head
pose, 2-D/3-D face landmarks, and Action Unit (AU) intensities / occurrences.

Only frames where OpenFace reports ``success == 1`` and
``confidence >= MIN_CONFIDENCE`` are kept.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum OpenFace confidence to accept a frame
MIN_CONFIDENCE: float = 0.80

# OpenFace feature groups used in this pipeline
GAZE_COLS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]
POSE_COLS = [
    "pose_Tx", "pose_Ty", "pose_Tz",
    "pose_Rx", "pose_Ry", "pose_Rz",
]
# Action Units available in OpenFace (intensity .._r and occurrence .._c)
AU_INTENSITY_COLS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
]
AU_OCCURRENCE_COLS = [
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
]

# Default feature set (AU intensities + pose + gaze)
DEFAULT_FEATURE_COLS = AU_INTENSITY_COLS + POSE_COLS + GAZE_COLS


def _load_openface_csv(path: Path, feature_cols: list[str]) -> Optional[np.ndarray]:
    """Load a single OpenFace CSV and return a 2-D feature array (frames × features).

    Frames with low confidence or failed tracking are discarded.  If no valid
    frames remain, ``None`` is returned so the caller can skip this clip.

    Parameters
    ----------
    path:
        Path to the OpenFace CSV file.
    feature_cols:
        List of column names to extract.

    Returns
    -------
    numpy.ndarray of shape (n_valid_frames, n_features), or ``None``.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None

    # Normalise column names (OpenFace sometimes adds leading spaces)
    df.columns = df.columns.str.strip()

    # Filter by tracking success and confidence
    if "success" in df.columns:
        df = df[df["success"] == 1]
    if "confidence" in df.columns:
        df = df[df["confidence"] >= MIN_CONFIDENCE]

    if df.empty:
        logger.debug("No valid frames in %s after filtering.", path)
        return None

    # Keep only the requested feature columns that actually exist in the file
    present = [c for c in feature_cols if c in df.columns]
    if not present:
        logger.warning("None of the requested feature columns found in %s.", path)
        return None

    return df[present].values.astype(np.float32)


def aggregate_clip_features(
    frames: np.ndarray,
    stat: str = "mean_std",
) -> np.ndarray:
    """Aggregate per-frame features into a single clip-level feature vector.

    Parameters
    ----------
    frames:
        Array of shape (n_frames, n_features).
    stat:
        Aggregation strategy.  Supported values:

        * ``"mean"`` – mean across frames.
        * ``"std"``  – standard deviation across frames.
        * ``"mean_std"`` – concatenation of mean and std (default).
        * ``"max"``  – max across frames.

    Returns
    -------
    1-D numpy.ndarray of aggregated features.
    """
    if stat == "mean":
        return frames.mean(axis=0)
    if stat == "std":
        return frames.std(axis=0)
    if stat == "mean_std":
        return np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
    if stat == "max":
        return frames.max(axis=0)
    raise ValueError(f"Unknown aggregation stat '{stat}'.  Choose from: mean, std, mean_std, max.")


def load_split(
    cmose_root: str | Path,
    split: str,
    feature_cols: list[str] = DEFAULT_FEATURE_COLS,
    aggregation: str = "mean_std",
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load one data split (train / val / test) from the CMOSE dataset.

    Parameters
    ----------
    cmose_root:
        Root directory of the CMOSE dataset (must contain ``labels/`` and
        ``openface/`` sub-directories).
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    feature_cols:
        OpenFace columns to extract (default: AU intensities + pose + gaze).
    aggregation:
        How to aggregate per-frame features into a clip vector (see
        :func:`aggregate_clip_features`).

    Returns
    -------
    X : numpy.ndarray
        Feature matrix of shape (n_clips, n_features).
    y : numpy.ndarray
        Integer label vector of shape (n_clips,).
    clip_ids : list[str]
        Clip identifiers corresponding to each row of ``X`` / ``y``.
    """
    cmose_root = Path(cmose_root)
    labels_path = cmose_root / "labels" / f"{split}_labels.csv"
    openface_dir = cmose_root / "openface"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not openface_dir.is_dir():
        raise FileNotFoundError(f"OpenFace feature directory not found: {openface_dir}")

    labels_df = pd.read_csv(labels_path)
    if not {"clip_id", "label"}.issubset(labels_df.columns):
        raise ValueError(
            f"Labels CSV must contain 'clip_id' and 'label' columns. "
            f"Found: {list(labels_df.columns)}"
        )

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    clip_ids: list[str] = []

    skipped = 0
    for _, row in labels_df.iterrows():
        clip_id = str(row["clip_id"])
        label = int(row["label"])
        csv_path = openface_dir / f"{clip_id}.csv"

        frames = _load_openface_csv(csv_path, feature_cols)
        if frames is None:
            logger.warning("Skipping clip %s (no valid frames or file missing).", clip_id)
            skipped += 1
            continue

        clip_feat = aggregate_clip_features(frames, stat=aggregation)
        X_list.append(clip_feat)
        y_list.append(label)
        clip_ids.append(clip_id)

    if skipped:
        logger.info("Skipped %d clips due to missing / invalid data.", skipped)

    if not X_list:
        raise RuntimeError(f"No valid clips loaded for split='{split}'.  Check your data paths.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    logger.info("Loaded split='%s': %d clips, feature dim=%d", split, len(y), X.shape[1])
    return X, y, clip_ids
