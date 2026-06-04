"""Generate the per-clip CMOSE-test predictions table consumed by the thesis clip UI.

Sole purpose: run each CMOSE-trained checkpoint on the CMOSE test split and write a
single wide CSV — ``outputs/model_assessment/naive/predictions_by_clip.csv`` — with one
row per clip and, for every model/loss run, the columns ``predicted_label__<run>``,
``predicted_id__<run>`` and ``confidence__<run>`` (plus ``clip_id`` and the dataset
``true_id``/``true_label``). The thesis clip browser
(``src/manual_label_ui/thesis_clips_ui.py``) launches this module as a subprocess to
create that file when it is missing.

No metrics, distributions, reports, or summaries are produced here.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.feature_extraction.extract_i3d import load_i3d_dataset_matrices
from src.feature_extraction.extract_openface import (
    ID_TO_LABEL,
    SampleMeta,
    load_cmose_metadata,
    load_dataset_matrices,
    load_openface_matrix,
    resample_frames,
)
from src.models.models import build_model
from src.output_paths import (
    CMOSE_TESTSET_ASSESSMENT_DIR,
    training_run_dir,
    training_run_key,
)
from src.visualization.style import COMPARISON_LOSS_SLUGS, MODEL_ORDER


MODEL_NAMES = list(MODEL_ORDER)
LOSS_SLUGS = list(COMPARISON_LOSS_SLUGS)
OPENFACE_MODELS = {"openface_mlp", "temporal_cnn", "lstm", "transformer"}
I3D_MODELS = {"i3d_mlp"}
FUSION_MODELS = {"openface_tcn_i3d_fusion"}

MODEL_RUNS: dict[str, dict[str, str]] = {}
for _model in MODEL_NAMES:
    for _loss_slug in LOSS_SLUGS:
        _run_dir = training_run_dir(model_name=_model, loss_name=_loss_slug, dataset="cmose")
        _config = {"model": _model, "checkpoint": str(_run_dir / "best_model.pth")}
        MODEL_RUNS[training_run_key(_model, _loss_slug)] = _config
        MODEL_RUNS.setdefault(f"{_model}/{_loss_slug}", _config)
DEFAULT_RUNS = [
    training_run_key(model, loss_slug_name)
    for model in MODEL_NAMES
    for loss_slug_name in LOSS_SLUGS
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the per-clip CMOSE-test predictions table for the thesis clip UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels_json", default="data/CMOSE/final_data_1.json")
    parser.add_argument("--cmose_openface_dir", default="data/CMOSE/features/openface")
    parser.add_argument("--cmose_i3d_dir", default="data/CMOSE/features/i3d")
    parser.add_argument("--output_dir", default=str(CMOSE_TESTSET_ASSESSMENT_DIR))
    parser.add_argument("--target_frames", type=int, default=300)
    parser.add_argument("--fusion_frames", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run keys to predict with. Defaults to all configured model/loss runs.",
    )
    return parser


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def resolve_cmose_openface_dir(path: str | Path) -> Path:
    requested = Path(path)
    candidates = [
        requested,
        Path("data/CMOSE/secondFeature"),
        Path("data/CMOSE/secondFeature/secondFeature"),
        Path("data/CMOSE/openface-features/secondFeature"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find CMOSE OpenFace directory. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def fit_matrix_normalizer_streaming(items: Any, *, load_fn: Any) -> tuple[np.ndarray, np.ndarray]:
    feature_sum: np.ndarray | None = None
    feature_sumsq: np.ndarray | None = None
    count = 0
    for _sample_id, path in tqdm(items, desc="Fitting normalizer", unit="sample", leave=False):
        matrix = load_fn(path).astype(np.float64, copy=False)
        if feature_sum is None:
            feature_sum = np.zeros(matrix.shape[1], dtype=np.float64)
            feature_sumsq = np.zeros(matrix.shape[1], dtype=np.float64)
        feature_sum += matrix.sum(axis=0)
        feature_sumsq += np.square(matrix).sum(axis=0)
        count += matrix.shape[0]

    if feature_sum is None or feature_sumsq is None or count == 0:
        raise RuntimeError("Cannot fit a normalizer on an empty stream")
    mean = feature_sum / float(count)
    variance = np.maximum(feature_sumsq / float(count) - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std <= 0.0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def fit_openface_normalizer_streaming(
    records: list[SampleMeta], *, target_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    return fit_matrix_normalizer_streaming(
        ((record.sample_id, record.csv_path) for record in records),
        load_fn=lambda path: load_openface_matrix(path, target_frames=target_frames),
    )


def fit_i3d_normalizer_streaming(
    sample_ids: list[str], *, feature_dir: str | Path, target_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    from src.feature_extraction.extract_i3d import load_i3d_matrix, resolve_i3d_feature_path

    return fit_matrix_normalizer_streaming(
        ((sample_id, resolve_i3d_feature_path(sample_id, feature_dir)) for sample_id in sample_ids),
        load_fn=lambda path: load_i3d_matrix(path, target_frames=target_frames),
    )


def normalize(matrices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((matrices.astype(np.float32) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(
        np.float32, copy=False
    )


def predict_probabilities(
    model: torch.nn.Module,
    features: np.ndarray | tuple[np.ndarray, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    arrays = features if isinstance(features, tuple) else (features,)
    dataset = TensorDataset(*(torch.from_numpy(array).float() for array in arrays))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model.to(device)
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", unit="batch", leave=False):
            tensors = [tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch]
            logits = model(*tensors) if len(tensors) > 1 else model(tensors[0])
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def load_model_for_run(
    run_key: str,
    run_cfg: dict[str, str],
    *,
    i3d_input_features: int | None,
    device: torch.device,
) -> torch.nn.Module:
    model_name = run_cfg["model"]
    model, _ = build_model(
        model_name,
        input_features=709,
        i3d_input_features=i3d_input_features if model_name in FUSION_MODELS else None,
        num_classes=4,
    )
    checkpoint_path = Path(run_cfg["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {run_key}: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    return model


def predictions_by_clip_frame(
    sample_ids: list[str],
    true_ids: list[int],
    run_probs: dict[str, np.ndarray],
    *,
    expected_runs: list[str],
) -> pd.DataFrame:
    """One row per clip; each run contributes predicted_id/label/confidence columns."""
    frame = pd.DataFrame(
        {
            "clip_id": sample_ids,
            "true_id": true_ids,
            "true_label": [ID_TO_LABEL[int(label)] for label in true_ids],
        }
    )
    for run_key in expected_runs:
        probs = run_probs[run_key]
        pred_ids = probs.argmax(axis=1).astype(int)
        frame[f"predicted_id__{run_key}"] = pred_ids
        frame[f"predicted_label__{run_key}"] = [ID_TO_LABEL[int(pred)] for pred in pred_ids]
        frame[f"confidence__{run_key}"] = probs.max(axis=1).astype(float)
    return frame


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    unknown_runs = [run for run in args.runs if run not in MODEL_RUNS]
    if unknown_runs:
        raise ValueError(f"Unknown run key(s): {unknown_runs}. Available: {sorted(MODEL_RUNS)}")

    print(f"Using device: {device}")
    cmose_openface_dir = resolve_cmose_openface_dir(args.cmose_openface_dir)
    records = load_cmose_metadata(args.labels_json, cmose_openface_dir, allowed_splits=("train", "test"))
    train_records = [record for record in records if record.split == "train"]
    test_records = [record for record in records if record.split == "test"]
    if not train_records:
        raise RuntimeError("No CMOSE train records were found for normalization")
    if not test_records:
        raise RuntimeError("No CMOSE test records were found")
    sample_ids = [record.sample_id for record in test_records]
    true_ids = [int(record.label_id) for record in test_records]
    print(f"CMOSE test clips: {len(test_records)}")

    need_openface = any(MODEL_RUNS[run]["model"] in OPENFACE_MODELS | FUSION_MODELS for run in args.runs)
    need_i3d = any(MODEL_RUNS[run]["model"] in I3D_MODELS | FUSION_MODELS for run in args.runs)

    openface_300: np.ndarray | None = None
    openface_fusion: np.ndarray | None = None
    i3d: np.ndarray | None = None
    i3d_input_features: int | None = None

    if need_openface:
        print("Fitting OpenFace normalizer on CMOSE train split")
        openface_mean, openface_std = fit_openface_normalizer_streaming(
            train_records, target_frames=args.target_frames
        )
        openface_raw, _labels, _loaded_ids = load_dataset_matrices(
            test_records, target_frames=args.target_frames, progress_desc="Loading CMOSE OpenFace"
        )
        openface_300 = normalize(openface_raw, openface_mean, openface_std)
        openface_fusion = np.stack(
            [
                resample_frames(sample, target_frames=args.fusion_frames)
                for sample in tqdm(openface_300, desc="Resampling OpenFace for fusion", unit="sample", leave=False)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        del openface_raw
        gc.collect()

    if need_i3d:
        print("Fitting I3D normalizer on CMOSE train split")
        train_sample_ids = [record.sample_id for record in train_records]
        i3d_mean, i3d_std = fit_i3d_normalizer_streaming(
            train_sample_ids, feature_dir=args.cmose_i3d_dir, target_frames=args.fusion_frames
        )
        i3d_raw = load_i3d_dataset_matrices(
            sample_ids,
            feature_dir=args.cmose_i3d_dir,
            target_frames=args.fusion_frames,
            progress_desc="Loading CMOSE I3D",
        )
        i3d = normalize(i3d_raw, i3d_mean, i3d_std)
        i3d_input_features = int(i3d.shape[-1])
        del i3d_raw
        gc.collect()

    run_probs: dict[str, np.ndarray] = {}
    for run_key in args.runs:
        run_cfg = MODEL_RUNS[run_key]
        model_name = run_cfg["model"]
        print(f"Predicting CMOSE test clips with {run_key}")
        model = load_model_for_run(run_key, run_cfg, i3d_input_features=i3d_input_features, device=device)
        if model_name in FUSION_MODELS:
            if openface_fusion is None or i3d is None:
                raise RuntimeError("Fusion inputs were not prepared")
            features: np.ndarray | tuple[np.ndarray, ...] = (openface_fusion, i3d)
        elif model_name in I3D_MODELS:
            if i3d is None:
                raise RuntimeError("I3D inputs were not prepared")
            features = i3d
        else:
            if openface_300 is None:
                raise RuntimeError("OpenFace inputs were not prepared")
            features = openface_300
        run_probs[run_key] = predict_probabilities(
            model, features, batch_size=args.batch_size, device=device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = predictions_by_clip_frame(sample_ids, true_ids, run_probs, expected_runs=args.runs)
    output_path = output_dir / "predictions_by_clip.csv"
    predictions.to_csv(output_path, index=False)
    print(f"Saved per-clip predictions to {output_path}  ({len(predictions)} clips, {len(args.runs)} runs)")


if __name__ == "__main__":
    main()
