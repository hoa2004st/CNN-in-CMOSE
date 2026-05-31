"""Write raw CMOSE-test and private-set predictions for trained checkpoints.

The output focus is the direct model result: one prediction row per model per
sample, plus a wide per-sample CSV that shows every model's prediction side by
side. Evaluation visualizations are intentionally not generated here.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
from src.evaluation.metrics import agreement_metrics_from_confusion, label_distance_metrics_from_confusion
from src.models.models import build_model
from src.output_paths import (
    CMOSE_TESTSET_ASSESSMENT_DIR,
    MANUAL_LABELS_CSV,
    MODEL_ASSESSMENT_DIR,
    RAW_PREDICTION_DIR,
    PRIVATE_ASSESSMENT_DIR,
    training_run_dir,
    training_run_key,
)
from src.visualization.style import (
    CLASS_LABELS as VIS_CLASS_LABELS,
    COMPARISON_LOSS_SLUGS,
    MODEL_ORDER,
)


MODEL_NAMES = list(MODEL_ORDER)
LOSS_SLUGS = list(COMPARISON_LOSS_SLUGS)
MODEL_RUNS = {}
for _model in MODEL_NAMES:
    for _loss_slug in LOSS_SLUGS:
        _run_dir = training_run_dir(model_name=_model, loss_name=_loss_slug, dataset="cmose")
        _config = {
            "model": _model,
            "checkpoint": str(_run_dir / "best_model.pth"),
            "metrics": str(_run_dir / "metrics.json"),
        }
        MODEL_RUNS[training_run_key(_model, _loss_slug)] = _config
        _legacy_key = f"{_model}/{_loss_slug}"
        if _legacy_key not in MODEL_RUNS:
            MODEL_RUNS[_legacy_key] = _config
DEFAULT_RUNS = [
    training_run_key(model, loss_slug_name)
    for model in MODEL_NAMES
    for loss_slug_name in LOSS_SLUGS
]
OPENFACE_MODELS = {"openface_mlp", "temporal_cnn", "lstm", "transformer"}
I3D_MODELS = {"i3d_mlp"}
FUSION_MODELS = {"openface_tcn_i3d_fusion"}
CLASS_IDS = list(range(4))
CLASS_LABELS = list(VIS_CLASS_LABELS)


@dataclass(frozen=True)
class PrivateSample:
    clip_id: str
    clip_path: Path
    openface_csv: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write raw predictions for CMOSE test clips and accepted private clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--accepted_csv", default="data/private/accepted.csv")
    parser.add_argument("--private_i3d_dir", default="data/private/features/i3d")
    parser.add_argument("--labels_json", default="data/CMOSE/final_data_1.json")
    parser.add_argument("--cmose_openface_dir", default="data/CMOSE/features/openface")
    parser.add_argument("--cmose_i3d_dir", default="data/CMOSE/features/i3d")
    parser.add_argument("--output_dir", default=str(RAW_PREDICTION_DIR))
    parser.add_argument("--dataset_output_root", default=str(MODEL_ASSESSMENT_DIR))
    parser.add_argument(
        "--document_path",
        default=str(RAW_PREDICTION_DIR / "raw_prediction_report.md"),
    )
    parser.add_argument(
        "--manual_labels_csv",
        default=str(MANUAL_LABELS_CSV),
        help="Optional manual private labels CSV. Rows with notes containing 'Delete' are excluded from labels only.",
    )
    parser.add_argument("--target_frames", type=int, default=300)
    parser.add_argument("--fusion_frames", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run keys to analyze. Defaults to all configured model/loss runs.",
    )
    parser.add_argument(
        "--from_existing",
        action="store_true",
        help="Regenerate raw CSV summaries/report from existing prediction CSV outputs without rerunning inference.",
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


def _manifest_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def load_private_samples(accepted_csv: str | Path) -> list[PrivateSample]:
    accepted_path = Path(accepted_csv)
    df = pd.read_csv(accepted_path)
    required = {"clip_id", "clip_path", "openface_csv", "is_accepted"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{accepted_path} is missing columns: {missing}")

    accepted = df[df["is_accepted"].astype(str) == "1"].copy()
    samples: list[PrivateSample] = []
    for row in accepted.itertuples(index=False):
        openface_csv = _manifest_path(row.openface_csv)
        if not openface_csv.exists():
            raise FileNotFoundError(f"Accepted OpenFace CSV does not exist: {openface_csv}")
        samples.append(
            PrivateSample(
                clip_id=str(row.clip_id),
                clip_path=_manifest_path(row.clip_path),
                openface_csv=openface_csv,
            )
        )
    if not samples:
        raise RuntimeError(f"No accepted private clips found in {accepted_path}")
    return samples


def load_private_manual_labels(labels_csv: str | Path) -> dict[str, int]:
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        return {}
    df = pd.read_csv(labels_path)
    required = {"clip_id", "manual_label_id", "notes"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{labels_path} is missing columns: {missing}")

    labeled = df[df["manual_label_id"].notna()].copy()
    notes = labeled["notes"].fillna("").astype(str)
    labeled = labeled[~notes.str.contains("Delete", case=False, na=False)]
    return {
        str(row.clip_id): int(row.manual_label_id)
        for row in labeled.itertuples(index=False)
        if str(row.manual_label_id) != ""
    }


def split_cmose_records(records: list[SampleMeta]) -> tuple[list[SampleMeta], list[SampleMeta]]:
    train_records = [record for record in records if record.split == "train"]
    test_records = [record for record in records if record.split == "test"]
    if not train_records:
        raise RuntimeError("No CMOSE train records were found for normalization")
    if not test_records:
        raise RuntimeError("No CMOSE test records were found for source-distribution reference")
    return train_records, test_records


def fit_openface_normalizer_streaming(
    records: list[SampleMeta],
    *,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    return fit_matrix_normalizer_streaming(
        ((record.sample_id, record.csv_path) for record in records),
        load_fn=lambda path: load_openface_matrix(path, target_frames=target_frames),
    )


def fit_i3d_normalizer_streaming(
    sample_ids: list[str],
    *,
    feature_dir: str | Path,
    target_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    from src.feature_extraction.extract_i3d import load_i3d_matrix, resolve_i3d_feature_path

    return fit_matrix_normalizer_streaming(
        ((sample_id, resolve_i3d_feature_path(sample_id, feature_dir)) for sample_id in sample_ids),
        load_fn=lambda path: load_i3d_matrix(path, target_frames=target_frames),
    )


def fit_matrix_normalizer_streaming(
    items: Any,
    *,
    load_fn: Any,
) -> tuple[np.ndarray, np.ndarray]:
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


def normalize(matrices: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((matrices.astype(np.float32) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(
        np.float32,
        copy=False,
    )


def load_private_openface(
    samples: list[PrivateSample],
    *,
    target_frames: int,
) -> np.ndarray:
    matrices = [
        load_openface_matrix(sample.openface_csv, target_frames=target_frames)
        for sample in tqdm(samples, desc="Loading private OpenFace", unit="sample", leave=False)
    ]
    return np.stack(matrices, axis=0).astype(np.float32, copy=False)


def load_private_i3d(
    sample_ids: list[str],
    *,
    feature_dir: str | Path,
    target_frames: int,
) -> np.ndarray:
    return load_i3d_dataset_matrices(
        sample_ids,
        feature_dir=feature_dir,
        target_frames=target_frames,
        progress_desc="Loading private I3D",
    )


def load_cmose_i3d(
    sample_ids: list[str],
    *,
    feature_dir: str | Path,
    target_frames: int,
) -> np.ndarray:
    return load_i3d_dataset_matrices(
        sample_ids,
        feature_dir=feature_dir,
        target_frames=target_frames,
        progress_desc="Loading CMOSE I3D",
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
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


def prediction_frame(
    run_key: str,
    sample_ids: list[str],
    probs: np.ndarray,
    *,
    true_labels: list[int | None] | np.ndarray | None = None,
) -> pd.DataFrame:
    pred_ids = probs.argmax(axis=1).astype(int)
    sorted_probs = np.sort(probs, axis=1)
    confidences = probs.max(axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / math.log(probs.shape[1])

    rows = []
    for idx, clip_id in enumerate(sample_ids):
        row = {
            "run": run_key,
            "clip_id": clip_id,
            "predicted_id": int(pred_ids[idx]),
            "predicted_label": ID_TO_LABEL[int(pred_ids[idx])],
            "confidence": float(confidences[idx]),
            "margin": float(margins[idx]),
            "normalized_entropy": float(entropy[idx]),
        }
        if true_labels is not None:
            true_value = true_labels[idx]
            if true_value is None or pd.isna(true_value):
                row["true_id"] = np.nan
                row["true_label"] = ""
                row["is_correct"] = np.nan
            else:
                true_id = int(true_value)
                row["true_id"] = true_id
                row["true_label"] = ID_TO_LABEL[true_id]
                row["is_correct"] = bool(pred_ids[idx] == true_id)
        for class_id in CLASS_IDS:
            row[f"prob_{class_id}_{ID_TO_LABEL[class_id]}"] = float(probs[idx, class_id])
        rows.append(row)
    return pd.DataFrame(rows)


def attach_true_labels(
    predictions: pd.DataFrame,
    labels_by_clip: dict[str, int],
) -> pd.DataFrame:
    if not labels_by_clip:
        return predictions

    labeled = predictions.copy()
    if "true_id" not in labeled.columns:
        labeled["true_id"] = np.nan
    if "true_label" not in labeled.columns:
        labeled["true_label"] = ""
    if "is_correct" not in labeled.columns:
        labeled["is_correct"] = pd.NA
    else:
        labeled["is_correct"] = labeled["is_correct"].astype("object")

    clip_labels = labeled["clip_id"].astype(str).map(labels_by_clip)
    has_label = clip_labels.notna()
    labeled.loc[has_label, "true_id"] = clip_labels[has_label].astype(int)
    labeled.loc[has_label, "true_label"] = labeled.loc[has_label, "true_id"].astype(int).map(ID_TO_LABEL)
    labeled.loc[has_label, "is_correct"] = (
        labeled.loc[has_label, "predicted_id"].astype(int)
        == labeled.loc[has_label, "true_id"].astype(int)
    )
    return labeled


def distribution_frame(predictions: pd.DataFrame, *, domain: str) -> pd.DataFrame:
    rows = []
    for run_key, group in predictions.groupby("run"):
        total = len(group)
        for class_id in CLASS_IDS:
            class_group = group[group["predicted_id"] == class_id]
            rows.append(
                {
                    "run": run_key,
                    "domain": domain,
                    "class_id": class_id,
                    "class_label": ID_TO_LABEL[class_id],
                    "count": int(len(class_group)),
                    "proportion": float(len(class_group) / total) if total else 0.0,
                    "mean_confidence": float(class_group["confidence"].mean())
                    if len(class_group)
                    else 0.0,
                    "mean_entropy": float(class_group["normalized_entropy"].mean())
                    if len(class_group)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def compute_supervised_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "true_id" not in predictions.columns:
        return pd.DataFrame()
    rows = []
    for run_key, group in predictions.groupby("run"):
        labeled = group[group["true_id"].notna()]
        if labeled.empty:
            continue
        y_true = labeled["true_id"].to_numpy(dtype=int)
        y_pred = labeled["predicted_id"].to_numpy(dtype=int)
        confusion = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
        per_class_accuracy = np.divide(
            confusion.diagonal(),
            confusion.sum(axis=1),
            out=np.zeros(len(CLASS_IDS), dtype=np.float64),
            where=confusion.sum(axis=1) > 0,
        )
        rows.append(
            {
                "run": run_key,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_accuracy": float(per_class_accuracy.mean()),
                "f1_macro": float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0)),
                "f1_weighted": float(
                    f1_score(y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0)
                ),
                **label_distance_metrics_from_confusion(confusion),
                **agreement_metrics_from_confusion(confusion),
            }
        )
    return pd.DataFrame(rows)

def raw_predictions_by_clip_frame(
    predictions: pd.DataFrame,
    *,
    expected_runs: list[str],
) -> pd.DataFrame:
    """Return one row per clip with each run's prediction in separate columns."""
    if predictions.empty:
        return pd.DataFrame()

    base_columns = ["clip_id"]
    for optional in ["true_id", "true_label"]:
        if optional in predictions.columns:
            base_columns.append(optional)
    base = predictions[base_columns].drop_duplicates("clip_id").set_index("clip_id")

    value_columns = [
        "predicted_id",
        "predicted_label",
        "confidence",
        "margin",
        "normalized_entropy",
    ]
    value_columns.extend(column for column in predictions.columns if column.startswith("prob_"))

    for run_key in expected_runs:
        run_rows = predictions[predictions["run"] == run_key].set_index("clip_id")
        if run_rows.empty:
            raise ValueError(f"Missing predictions for run: {run_key}")
        missing_columns = [column for column in value_columns if column not in run_rows.columns]
        if missing_columns:
            raise ValueError(f"Missing prediction column(s) for {run_key}: {missing_columns}")
        renamed = run_rows[value_columns].rename(
            columns={column: f"{column}__{run_key}" for column in value_columns}
        )
        base = base.join(renamed, how="left")

    return base.reset_index()


def true_distribution_from_predictions(predictions: pd.DataFrame, *, domain: str) -> pd.DataFrame:
    if "true_id" not in predictions.columns:
        return pd.DataFrame()
    labeled = predictions[["clip_id", "true_id"]].drop_duplicates("clip_id")
    labeled = labeled[labeled["true_id"].notna()]
    total = len(labeled)
    if total == 0:
        return pd.DataFrame()

    rows = []
    true_ids = labeled["true_id"].to_numpy(dtype=int)
    for class_id in CLASS_IDS:
        count = int(np.sum(true_ids == class_id))
        rows.append(
            {
                "run": "__labels__",
                "domain": domain,
                "class_id": class_id,
                "class_label": ID_TO_LABEL[class_id],
                "count": count,
                "proportion": float(count / total),
                "mean_confidence": np.nan,
                "mean_entropy": np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_dataset_prediction_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    predictions: pd.DataFrame,
    distributions: pd.DataFrame,
    expected_runs: list[str],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    supervised_metrics = compute_supervised_metrics(predictions)
    predictions_by_clip = raw_predictions_by_clip_frame(predictions, expected_runs=expected_runs)
    true_distribution = true_distribution_from_predictions(predictions, domain=f"{dataset_name}_true")

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    predictions_by_clip.to_csv(output_dir / "predictions_by_clip.csv", index=False)
    distributions.to_csv(output_dir / "prediction_distribution.csv", index=False)
    supervised_metrics.to_csv(output_dir / "supervised_metrics.csv", index=False)
    true_distribution.to_csv(
        output_dir / "true_label_distribution.csv",
        index=False,
    )
    dataset_summary = {
        "dataset": dataset_name,
        "num_samples": int(predictions["clip_id"].nunique()),
        "runs": expected_runs,
        "prediction_files": {
            "long": str(output_dir / "predictions.csv"),
            "by_clip": str(output_dir / "predictions_by_clip.csv"),
            "distribution": str(output_dir / "prediction_distribution.csv"),
        },
        "supervised_metrics": supervised_metrics.to_dict(orient="records"),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    return supervised_metrics


def write_top_level_prediction_outputs(
    *,
    output_dir: Path,
    private_predictions: pd.DataFrame,
    cmose_predictions: pd.DataFrame,
    distributions: pd.DataFrame,
    expected_runs: list[str],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_predictions.to_csv(output_dir / "private_predictions.csv", index=False)
    cmose_predictions.to_csv(output_dir / "cmose_test_predictions.csv", index=False)
    raw_predictions_by_clip_frame(private_predictions, expected_runs=expected_runs).to_csv(
        output_dir / "private_predictions_by_clip.csv",
        index=False,
    )
    raw_predictions_by_clip_frame(cmose_predictions, expected_runs=expected_runs).to_csv(
        output_dir / "cmose_test_predictions_by_clip.csv",
        index=False,
    )
    distributions.to_csv(output_dir / "prediction_distribution.csv", index=False)
    (output_dir / "raw_prediction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def build_raw_summary(
    *,
    samples: list[PrivateSample],
    manual_labels: dict[str, int],
    test_record_count: int,
    args: argparse.Namespace,
    device: torch.device,
    run_metrics: dict[str, dict[str, Any]],
    private_distribution: pd.DataFrame,
    cmose_distribution: pd.DataFrame,
    private_supervised_metrics: pd.DataFrame,
    cmose_supervised_metrics: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "accepted_private_clips": len(samples),
        "private_manual_labels_available": int(
            sum(sample.clip_id in manual_labels for sample in samples)
        ),
        "manual_labels_csv": str(args.manual_labels_csv),
        "manual_label_filter": "labels with notes containing 'Delete' are excluded from metric rows",
        "cmose_test_clips": test_record_count,
        "runs": args.runs,
        "device": device.type,
        "target_frames": args.target_frames,
        "fusion_frames": args.fusion_frames,
        "raw_prediction_files": {
            "private_long": str(Path(args.output_dir) / "private_predictions.csv"),
            "private_by_clip": str(Path(args.output_dir) / "private_predictions_by_clip.csv"),
            "cmose_test_long": str(Path(args.output_dir) / "cmose_test_predictions.csv"),
            "cmose_test_by_clip": str(Path(args.output_dir) / "cmose_test_predictions_by_clip.csv"),
        },
        "run_metrics": run_metrics,
        "private_prediction_distribution": private_distribution.to_dict(orient="records"),
        "cmose_test_prediction_distribution": cmose_distribution.to_dict(orient="records"),
        "private_supervised_metrics": private_supervised_metrics.to_dict(orient="records"),
        "cmose_supervised_metrics": cmose_supervised_metrics.to_dict(orient="records"),
    }


def load_run_metrics(run_cfg: dict[str, str]) -> dict[str, float]:
    metrics_path = Path(run_cfg["metrics"])
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    distance_metrics = {
        "mae": metrics.get("mae"),
        "mse": metrics.get("mse"),
        "macro_mae": metrics.get("macro_mae"),
    }
    agreement_metrics = {
        "cohen_kappa": metrics.get("cohen_kappa"),
        "quadratic_weighted_kappa": metrics.get("quadratic_weighted_kappa"),
    }
    if any(value is None for value in distance_metrics.values()) and metrics.get("confusion_matrix"):
        distance_metrics.update(label_distance_metrics_from_confusion(metrics["confusion_matrix"]))
    if any(value is None for value in agreement_metrics.values()) and metrics.get("confusion_matrix"):
        agreement_metrics.update(agreement_metrics_from_confusion(metrics["confusion_matrix"]))
    for key, value in distance_metrics.items():
        if value is None:
            distance_metrics[key] = float("nan")
    for key, value in agreement_metrics.items():
        if value is None:
            agreement_metrics[key] = float("nan")
    return {
        "accuracy": metrics.get("accuracy", float("nan")),
        "macro_accuracy": metrics.get("macro_accuracy", float("nan")),
        "f1_macro": metrics.get("f1_macro", float("nan")),
        "f1_weighted": metrics.get("f1_weighted", float("nan")),
        "mae": distance_metrics.get("mae", float("nan")),
        "mse": distance_metrics.get("mse", float("nan")),
        "macro_mae": distance_metrics.get("macro_mae", float("nan")),
        "cohen_kappa": agreement_metrics.get("cohen_kappa", float("nan")),
        "quadratic_weighted_kappa": agreement_metrics.get("quadratic_weighted_kappa", float("nan")),
    }


def write_markdown_report(
    *,
    output_path: Path,
    output_dir: Path,
    dataset_output_root: Path,
    samples: list[PrivateSample],
    run_metrics: dict[str, dict[str, Any]],
    distributions: pd.DataFrame,
    private_supervised_metrics: pd.DataFrame | None = None,
    cmose_supervised_metrics: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Raw Prediction Outputs",
        "",
        "This report records the direct predictions made by each configured CMOSE-trained run.",
        "",
        f"- Accepted private clips predicted: {len(samples)}",
        f"- Runs predicted: {len(run_metrics)}",
        "",
        "## Raw CSV Files",
        "",
        "| Dataset | Long format | One row per clip | Distribution | Metrics |",
        "|---|---|---|---|---|",
        (
            f"| Private | `{output_dir / 'private_predictions.csv'}` "
            f"| `{output_dir / 'private_predictions_by_clip.csv'}` "
            f"| `{dataset_output_root / PRIVATE_ASSESSMENT_DIR.name / 'prediction_distribution.csv'}` "
            f"| `{dataset_output_root / PRIVATE_ASSESSMENT_DIR.name / 'supervised_metrics.csv'}` |"
        ),
        (
            f"| CMOSE test | `{output_dir / 'cmose_test_predictions.csv'}` "
            f"| `{output_dir / 'cmose_test_predictions_by_clip.csv'}` "
            f"| `{dataset_output_root / CMOSE_TESTSET_ASSESSMENT_DIR.name / 'prediction_distribution.csv'}` "
            f"| `{dataset_output_root / CMOSE_TESTSET_ASSESSMENT_DIR.name / 'supervised_metrics.csv'}` |"
        ),
        "",
        "## Prediction Columns",
        "",
        "- `predictions.csv` / `*_predictions.csv`: one row per `run` and `clip_id`, including predicted class, confidence, margin, normalized entropy, and class probabilities.",
        "- `predictions_by_clip.csv` / `*_predictions_by_clip.csv`: one row per clip, with each run's prediction in separate columns.",
        "",
        "## CMOSE Test Metrics",
        "",
        "| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    cmose_metrics = cmose_supervised_metrics if cmose_supervised_metrics is not None else pd.DataFrame()
    metric_rows = (
        cmose_metrics.sort_values("run").to_dict(orient="records")
        if not cmose_metrics.empty
        else [{"run": run_key, **metrics} for run_key, metrics in sorted(run_metrics.items())]
    )
    for metrics in metric_rows:
        lines.append(
            "| "
            + str(metrics.get("run", ""))
            + f" | {metrics.get('accuracy', float('nan')):.4f}"
            + f" | {metrics.get('macro_accuracy', float('nan')):.4f}"
            + f" | {metrics.get('f1_macro', float('nan')):.4f}"
            + f" | {metrics.get('f1_weighted', float('nan')):.4f}"
            + f" | {metrics.get('mae', float('nan')):.4f}"
            + f" | {metrics.get('mse', float('nan')):.4f} |"
        )

    if private_supervised_metrics is not None and not private_supervised_metrics.empty:
        lines.extend(
            [
                "",
                "## Private Manual-Label Metrics",
                "",
                "| Run | Accuracy | Macro Accuracy | F1 Macro | F1 Weighted | MAE | MSE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in private_supervised_metrics.sort_values("run").itertuples(index=False):
            lines.append(
                f"| {row.run} | {row.accuracy:.4f} | {row.macro_accuracy:.4f} "
                f"| {row.f1_macro:.4f} | {row.f1_weighted:.4f} | {row.mae:.4f} | {row.mse:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Prediction Distribution",
            "",
            "| Dataset | Run | Highly Disengage | Disengage | Engage | Highly Engage |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    predicted_domains = {
        "private_accepted": "Private",
        "cmose_test_predicted": "CMOSE test",
    }
    for domain, label in predicted_domains.items():
        domain_df = distributions[distributions["domain"] == domain]
        for run_key in sorted(domain_df["run"].unique()):
            run_distribution = domain_df[domain_df["run"] == run_key].set_index("class_label")
            props = [
                float(run_distribution.loc[class_label, "proportion"])
                if class_label in run_distribution.index
                else 0.0
                for class_label in CLASS_LABELS
            ]
            lines.append(
                "| "
                + label
                + " | "
                + run_key
                + " | "
                + " | ".join(f"{prop:.3f}" for prop in props)
                + " |"
            )

    lines.extend(["", "## Run Metrics Files", ""])
    for run_key in sorted(run_metrics):
        lines.append(
            f"- `{run_key}`: `{MODEL_RUNS[run_key]['metrics']}`"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root = Path(args.dataset_output_root)
    private_dataset_dir = dataset_output_root / PRIVATE_ASSESSMENT_DIR.name
    cmose_dataset_dir = dataset_output_root / CMOSE_TESTSET_ASSESSMENT_DIR.name
    document_path = Path(args.document_path)
    device = resolve_device(args.device)

    unknown_runs = [run for run in args.runs if run not in MODEL_RUNS]
    if unknown_runs:
        raise ValueError(f"Unknown run key(s): {unknown_runs}. Available: {sorted(MODEL_RUNS)}")

    print(f"Using device: {device}")
    samples = load_private_samples(args.accepted_csv)
    manual_labels = load_private_manual_labels(args.manual_labels_csv)
    sample_ids = [sample.clip_id for sample in samples]
    private_true_labels = [manual_labels.get(sample_id) for sample_id in sample_ids]
    if not samples:
        raise RuntimeError(f"No accepted private clips were found in {args.accepted_csv}")
    print(f"Accepted private clips: {len(samples)}")
    print(f"Private manual labels attached: {sum(label is not None for label in private_true_labels)}")

    if args.from_existing:
        predictions_path = private_dataset_dir / "predictions.csv"
        cmose_predictions_path = cmose_dataset_dir / "predictions.csv"
        for path in (predictions_path, cmose_predictions_path):
            if not path.exists():
                raise FileNotFoundError(f"Cannot use --from_existing; missing {path}")
        predictions = pd.read_csv(predictions_path)
        predictions = attach_true_labels(predictions, manual_labels)
        cmose_predictions = pd.read_csv(cmose_predictions_path)
        private_distribution = distribution_frame(predictions, domain="private_accepted")
        cmose_distribution = distribution_frame(cmose_predictions, domain="cmose_test_predicted")
        distributions = pd.concat([private_distribution, cmose_distribution], ignore_index=True)
        private_supervised_metrics = write_dataset_prediction_outputs(
            output_dir=private_dataset_dir,
            dataset_name="private",
            predictions=predictions,
            distributions=private_distribution,
            expected_runs=args.runs,
        )
        cmose_supervised_metrics = write_dataset_prediction_outputs(
            output_dir=cmose_dataset_dir,
            dataset_name="cmose",
            predictions=cmose_predictions,
            distributions=cmose_distribution,
            expected_runs=args.runs,
        )
        run_metrics = {run_key: load_run_metrics(MODEL_RUNS[run_key]) for run_key in args.runs}
        write_top_level_prediction_outputs(
            output_dir=output_dir,
            private_predictions=predictions,
            cmose_predictions=cmose_predictions,
            distributions=distributions,
            expected_runs=args.runs,
            summary=build_raw_summary(
                samples=samples,
                manual_labels=manual_labels,
                test_record_count=int(cmose_predictions["clip_id"].nunique()),
                args=args,
                device=device,
                run_metrics=run_metrics,
                private_distribution=private_distribution,
                cmose_distribution=cmose_distribution,
                private_supervised_metrics=private_supervised_metrics,
                cmose_supervised_metrics=cmose_supervised_metrics,
            ),
        )
        write_markdown_report(
            output_path=document_path,
            output_dir=output_dir,
            dataset_output_root=dataset_output_root,
            samples=samples,
            run_metrics=run_metrics,
            distributions=distributions,
            private_supervised_metrics=private_supervised_metrics,
            cmose_supervised_metrics=cmose_supervised_metrics,
        )
        print(f"Regenerated raw prediction outputs from existing CSVs under {output_dir}")
        print(f"Saved markdown report to {document_path}")
        return

    cmose_openface_dir = resolve_cmose_openface_dir(args.cmose_openface_dir)
    cmose_records = load_cmose_metadata(
        args.labels_json,
        cmose_openface_dir,
        allowed_splits=("train", "test"),
    )
    train_records, test_records = split_cmose_records(cmose_records)
    cmose_sample_ids = [record.sample_id for record in test_records]
    cmose_true_labels = np.array([record.label_id for record in test_records], dtype=np.int64)
    print(f"CMOSE test clips: {len(test_records)}")

    need_openface = any(MODEL_RUNS[run]["model"] in OPENFACE_MODELS | FUSION_MODELS for run in args.runs)
    need_i3d = any(MODEL_RUNS[run]["model"] in I3D_MODELS | FUSION_MODELS for run in args.runs)

    private_openface_300: np.ndarray | None = None
    private_openface_fusion: np.ndarray | None = None
    private_i3d: np.ndarray | None = None
    cmose_openface_300: np.ndarray | None = None
    cmose_openface_fusion: np.ndarray | None = None
    cmose_i3d: np.ndarray | None = None
    i3d_input_features: int | None = None

    if need_openface:
        print("Fitting OpenFace normalizer on CMOSE train split")
        openface_mean, openface_std = fit_openface_normalizer_streaming(
            train_records,
            target_frames=args.target_frames,
        )
        cmose_openface_raw, _cmose_labels, _loaded_cmose_ids = load_dataset_matrices(
            test_records,
            target_frames=args.target_frames,
            progress_desc="Loading CMOSE OpenFace",
        )
        cmose_openface_300 = normalize(cmose_openface_raw, openface_mean, openface_std)
        cmose_openface_fusion = np.stack(
            [
                resample_frames(sample, target_frames=args.fusion_frames)
                for sample in tqdm(
                    cmose_openface_300,
                    desc="Resampling CMOSE OpenFace for fusion",
                    unit="sample",
                    leave=False,
                )
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        private_openface_raw = load_private_openface(samples, target_frames=args.target_frames)
        private_openface_300 = normalize(private_openface_raw, openface_mean, openface_std)
        private_openface_fusion = np.stack(
            [
                resample_frames(sample, target_frames=args.fusion_frames)
                for sample in tqdm(
                    private_openface_300,
                    desc="Resampling private OpenFace for fusion",
                    unit="sample",
                    leave=False,
                )
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        del cmose_openface_raw, private_openface_raw
        gc.collect()

    if need_i3d:
        print("Fitting I3D normalizer on CMOSE train split")
        train_sample_ids = [record.sample_id for record in train_records]
        i3d_mean, i3d_std = fit_i3d_normalizer_streaming(
            train_sample_ids,
            feature_dir=args.cmose_i3d_dir,
            target_frames=args.fusion_frames,
        )
        private_i3d_raw = load_private_i3d(
            sample_ids,
            feature_dir=args.private_i3d_dir,
            target_frames=args.fusion_frames,
        )
        cmose_i3d_raw = load_cmose_i3d(
            cmose_sample_ids,
            feature_dir=args.cmose_i3d_dir,
            target_frames=args.fusion_frames,
        )
        private_i3d = normalize(private_i3d_raw, i3d_mean, i3d_std)
        cmose_i3d = normalize(cmose_i3d_raw, i3d_mean, i3d_std)
        i3d_input_features = int(cmose_i3d.shape[-1])
        del private_i3d_raw, cmose_i3d_raw
        gc.collect()

    private_prediction_tables = []
    cmose_prediction_tables = []
    run_metrics: dict[str, dict[str, Any]] = {}

    for run_key in args.runs:
        run_cfg = MODEL_RUNS[run_key]
        model_name = run_cfg["model"]
        print(f"Predicting private and CMOSE clips with {run_key}")
        model = load_model_for_run(
            run_key,
            run_cfg,
            i3d_input_features=i3d_input_features,
            device=device,
        )
        if model_name in FUSION_MODELS:
            if (
                private_openface_fusion is None
                or private_i3d is None
                or cmose_openface_fusion is None
                or cmose_i3d is None
            ):
                raise RuntimeError("Fusion inputs were not prepared")
            private_features: np.ndarray | tuple[np.ndarray, ...] = (private_openface_fusion, private_i3d)
            cmose_features: np.ndarray | tuple[np.ndarray, ...] = (cmose_openface_fusion, cmose_i3d)
        elif model_name in I3D_MODELS:
            if private_i3d is None or cmose_i3d is None:
                raise RuntimeError("I3D inputs were not prepared")
            private_features = private_i3d
            cmose_features = cmose_i3d
        else:
            if private_openface_300 is None or cmose_openface_300 is None:
                raise RuntimeError("OpenFace inputs were not prepared")
            private_features = private_openface_300
            cmose_features = cmose_openface_300

        private_probs = predict_probabilities(
            model,
            private_features,
            batch_size=args.batch_size,
            device=device,
        )
        private_prediction_tables.append(
            prediction_frame(run_key, sample_ids, private_probs, true_labels=private_true_labels)
        )
        cmose_probs = predict_probabilities(
            model,
            cmose_features,
            batch_size=args.batch_size,
            device=device,
        )
        cmose_prediction_tables.append(
            prediction_frame(run_key, cmose_sample_ids, cmose_probs, true_labels=cmose_true_labels)
        )
        run_metrics[run_key] = load_run_metrics(run_cfg)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(private_prediction_tables, ignore_index=True)
    cmose_predictions = pd.concat(cmose_prediction_tables, ignore_index=True)
    private_distribution = distribution_frame(predictions, domain="private_accepted")
    cmose_distribution = distribution_frame(cmose_predictions, domain="cmose_test_predicted")
    distributions = pd.concat([private_distribution, cmose_distribution], ignore_index=True)
    private_supervised_metrics = write_dataset_prediction_outputs(
        output_dir=private_dataset_dir,
        dataset_name="private",
        predictions=predictions,
        distributions=private_distribution,
        expected_runs=args.runs,
    )
    cmose_supervised_metrics = write_dataset_prediction_outputs(
        output_dir=cmose_dataset_dir,
        dataset_name="cmose",
        predictions=cmose_predictions,
        distributions=cmose_distribution,
        expected_runs=args.runs,
    )
    write_top_level_prediction_outputs(
        output_dir=output_dir,
        private_predictions=predictions,
        cmose_predictions=cmose_predictions,
        distributions=distributions,
        expected_runs=args.runs,
        summary=build_raw_summary(
            samples=samples,
            manual_labels=manual_labels,
            test_record_count=len(test_records),
            args=args,
            device=device,
            run_metrics=run_metrics,
            private_distribution=private_distribution,
            cmose_distribution=cmose_distribution,
            private_supervised_metrics=private_supervised_metrics,
            cmose_supervised_metrics=cmose_supervised_metrics,
        ),
    )
    write_markdown_report(
        output_path=document_path,
        output_dir=output_dir,
        dataset_output_root=dataset_output_root,
        samples=samples,
        run_metrics=run_metrics,
        distributions=distributions,
        private_supervised_metrics=private_supervised_metrics,
        cmose_supervised_metrics=cmose_supervised_metrics,
    )
    print(f"Saved raw prediction outputs to {output_dir}")
    print(f"Saved markdown report to {document_path}")


if __name__ == "__main__":
    main()
