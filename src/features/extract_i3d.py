"""I3D feature loading/materialization helpers for the CMOSE pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.features.extract_openface import resample_frames


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
