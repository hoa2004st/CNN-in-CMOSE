"""Run private-domain prediction analysis with retained CMOSE checkpoints.

This script addresses the domain-shift step in ``documents/thesis_direction.md``:
apply CMOSE-trained retained models to the accepted private clips, quantify the
target prediction distributions, and compare them with each model's CMOSE test
behavior where available.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
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


MODEL_RUNS = {
    "openface_mlp/weighted_ce": {
        "model": "openface_mlp",
        "checkpoint": "outputs/openface_mlp/weighted_ce/best_model.pth",
        "metrics": "outputs/openface_mlp/weighted_ce/metrics.json",
    },
    "lstm/weighted_ce": {
        "model": "lstm",
        "checkpoint": "outputs/lstm/weighted_ce/best_model.pth",
        "metrics": "outputs/lstm/weighted_ce/metrics.json",
    },
    "transformer/ce": {
        "model": "transformer",
        "checkpoint": "outputs/transformer/ce/best_model.pth",
        "metrics": "outputs/transformer/ce/metrics.json",
    },
    "temporal_cnn/weighted_ce": {
        "model": "temporal_cnn",
        "checkpoint": "outputs/temporal_cnn/weighted_ce/best_model.pth",
        "metrics": "outputs/temporal_cnn/weighted_ce/metrics.json",
    },
    "i3d_mlp/ce": {
        "model": "i3d_mlp",
        "checkpoint": "outputs/i3d_mlp/ce/best_model.pth",
        "metrics": "outputs/i3d_mlp/ce/metrics.json",
    },
    "openface_tcn_i3d_fusion/ce": {
        "model": "openface_tcn_i3d_fusion",
        "checkpoint": "outputs/openface_tcn_i3d_fusion/ce/best_model.pth",
        "metrics": "outputs/openface_tcn_i3d_fusion/ce/metrics.json",
    },
}
OPENFACE_MODELS = {"openface_mlp", "temporal_cnn", "lstm", "transformer"}
I3D_MODELS = {"i3d_mlp"}
FUSION_MODELS = {"openface_tcn_i3d_fusion"}
CLASS_IDS = list(range(4))
CLASS_LABELS = [ID_TO_LABEL[i] for i in CLASS_IDS]


@dataclass(frozen=True)
class PrivateSample:
    clip_id: str
    clip_path: Path
    openface_csv: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run domain-shift prediction analysis on accepted private clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--accepted_csv", default="data/private/accepted.csv")
    parser.add_argument("--private_i3d_dir", default="data/private/features/i3d")
    parser.add_argument("--labels_json", default="data/CMOSE/final_data_1.json")
    parser.add_argument("--cmose_openface_dir", default="data/CMOSE/features/openface")
    parser.add_argument("--cmose_i3d_dir", default="data/CMOSE/features/i3d")
    parser.add_argument("--output_dir", default="outputs/domain_shift_analysis")
    parser.add_argument("--dataset_output_root", default="outputs/dataset_analysis")
    parser.add_argument("--document_path", default="documents/domain_shift_analysis.md")
    parser.add_argument("--target_frames", type=int, default=300)
    parser.add_argument("--fusion_frames", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(MODEL_RUNS),
        help="Run keys to analyze. Defaults to the best retained run for each model.",
    )
    parser.add_argument(
        "--from_existing",
        action="store_true",
        help="Regenerate plots/report from existing CSV outputs without rerunning inference.",
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
        openface_csv = Path(str(row.openface_csv))
        if not openface_csv.exists():
            raise FileNotFoundError(f"Accepted OpenFace CSV does not exist: {openface_csv}")
        samples.append(
            PrivateSample(
                clip_id=str(row.clip_id),
                clip_path=Path(str(row.clip_path)),
                openface_csv=openface_csv,
            )
        )
    if not samples:
        raise RuntimeError(f"No accepted private clips found in {accepted_path}")
    return samples


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
    true_labels: np.ndarray | None = None,
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
            true_id = int(true_labels[idx])
            row["true_id"] = true_id
            row["true_label"] = ID_TO_LABEL[true_id]
            row["is_correct"] = bool(pred_ids[idx] == true_id)
        for class_id in CLASS_IDS:
            row[f"prob_{class_id}_{ID_TO_LABEL[class_id]}"] = float(probs[idx, class_id])
        rows.append(row)
    return pd.DataFrame(rows)


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


def true_distribution_frame(sample_ids: list[str], true_labels: np.ndarray, *, domain: str) -> pd.DataFrame:
    total = len(sample_ids)
    rows = []
    for class_id in CLASS_IDS:
        count = int(np.sum(true_labels == class_id))
        rows.append(
            {
                "run": "__labels__",
                "domain": domain,
                "class_id": class_id,
                "class_label": ID_TO_LABEL[class_id],
                "count": count,
                "proportion": float(count / total) if total else 0.0,
                "mean_confidence": np.nan,
                "mean_entropy": np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_supervised_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "true_id" not in predictions.columns:
        return pd.DataFrame()
    rows = []
    for run_key, group in predictions.groupby("run"):
        y_true = group["true_id"].to_numpy(dtype=int)
        y_pred = group["predicted_id"].to_numpy(dtype=int)
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
            }
        )
    return pd.DataFrame(rows)


def write_dataset_analysis_outputs(
    *,
    output_dir: Path,
    dataset_name: str,
    predictions: pd.DataFrame,
    distributions: pd.DataFrame,
    expected_runs: list[str],
    true_labels: np.ndarray | None = None,
    sample_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    agreement_per_clip, pairwise_kappa, agreement_summary = compute_model_agreement(
        predictions,
        expected_runs=expected_runs,
    )
    supervised_metrics = compute_supervised_metrics(predictions)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    distributions.to_csv(output_dir / "prediction_distribution.csv", index=False)
    agreement_per_clip.to_csv(output_dir / "model_agreement_per_clip.csv", index=False)
    pairwise_kappa.to_csv(output_dir / "pairwise_cohens_kappa.csv", index=False)
    if not supervised_metrics.empty:
        supervised_metrics.to_csv(output_dir / "supervised_metrics.csv", index=False)
    if true_labels is not None and sample_ids is not None:
        true_distribution_frame(sample_ids, true_labels, domain=f"{dataset_name}_true").to_csv(
            output_dir / "true_label_distribution.csv",
            index=False,
        )
    (output_dir / "agreement_summary.json").write_text(
        json.dumps(agreement_summary, indent=2),
        encoding="utf-8",
    )
    dataset_summary = {
        "dataset": dataset_name,
        "num_samples": int(predictions["clip_id"].nunique()),
        "runs": expected_runs,
        "agreement_summary": agreement_summary,
        "supervised_metrics": supervised_metrics.to_dict(orient="records"),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    plot_prediction_distribution(
        distributions,
        output_dir / "prediction_distribution.png",
        domain=f"{dataset_name}_predicted",
        ylabel=f"{dataset_name.upper()} prediction proportion",
    )
    return agreement_per_clip, pairwise_kappa, agreement_summary


def compute_model_agreement(
    predictions: pd.DataFrame,
    *,
    expected_runs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute per-clip and across-clip agreement for retained model predictions."""
    prediction_pivot = predictions.pivot(
        index="clip_id",
        columns="run",
        values="predicted_id",
    )
    confidence_pivot = predictions.pivot(
        index="clip_id",
        columns="run",
        values="confidence",
    )
    missing_runs = [run for run in expected_runs if run not in prediction_pivot.columns]
    if missing_runs:
        raise ValueError(f"Missing predictions for run(s): {missing_runs}")
    prediction_pivot = prediction_pivot[expected_runs].astype(int)
    confidence_pivot = confidence_pivot[expected_runs].astype(float)

    per_clip_rows = []
    total_pairs = len(expected_runs) * (len(expected_runs) - 1) // 2
    for clip_id, row in prediction_pivot.iterrows():
        votes = row.to_numpy(dtype=int)
        counts = np.bincount(votes, minlength=len(CLASS_IDS))
        vote_probs = counts.astype(np.float64) / float(len(votes))
        nonzero_vote_probs = vote_probs[vote_probs > 0.0]
        vote_entropy = -float(np.sum(nonzero_vote_probs * np.log(nonzero_vote_probs)))
        normalized_vote_entropy = vote_entropy / math.log(len(CLASS_IDS))
        agreeing_pairs = int(sum(count * (count - 1) // 2 for count in counts))
        majority_class_id = int(np.argmax(counts))
        per_clip_row: dict[str, Any] = {
            "clip_id": clip_id,
            "num_models": len(expected_runs),
            "agreement_rate": float(agreeing_pairs / total_pairs) if total_pairs else 1.0,
            "prediction_entropy": normalized_vote_entropy,
            "mean_confidence": float(confidence_pivot.loc[clip_id].mean()),
            "majority_class_id": majority_class_id,
            "majority_label": ID_TO_LABEL[majority_class_id],
            "majority_vote_count": int(counts[majority_class_id]),
        }
        for class_id in CLASS_IDS:
            per_clip_row[f"vote_count_{class_id}_{ID_TO_LABEL[class_id]}"] = int(counts[class_id])
            per_clip_row[f"vote_proportion_{class_id}_{ID_TO_LABEL[class_id]}"] = float(
                vote_probs[class_id]
            )
        for run in expected_runs:
            per_clip_row[f"predicted_id__{run}"] = int(prediction_pivot.loc[clip_id, run])
            per_clip_row[f"predicted_label__{run}"] = ID_TO_LABEL[
                int(prediction_pivot.loc[clip_id, run])
            ]
            per_clip_row[f"confidence__{run}"] = float(confidence_pivot.loc[clip_id, run])
        per_clip_rows.append(per_clip_row)

    pairwise_rows = []
    for left_idx, left_run in enumerate(expected_runs):
        for right_run in expected_runs[left_idx + 1 :]:
            left_values = prediction_pivot[left_run].to_numpy(dtype=int)
            right_values = prediction_pivot[right_run].to_numpy(dtype=int)
            raw_agreement = float(np.mean(left_values == right_values))
            pairwise_rows.append(
                {
                    "run_a": left_run,
                    "run_b": right_run,
                    "cohens_kappa": float(cohen_kappa_score(left_values, right_values, labels=CLASS_IDS)),
                    "raw_agreement": raw_agreement,
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    per_clip = pd.DataFrame(per_clip_rows)
    fleiss = fleiss_kappa_from_votes(prediction_pivot.to_numpy(dtype=int), n_classes=len(CLASS_IDS))
    summary = {
        "num_clips": int(len(per_clip)),
        "num_models": int(len(expected_runs)),
        "agreement_rate_mean": float(per_clip["agreement_rate"].mean()),
        "agreement_rate_median": float(per_clip["agreement_rate"].median()),
        "prediction_entropy_mean": float(per_clip["prediction_entropy"].mean()),
        "prediction_entropy_median": float(per_clip["prediction_entropy"].median()),
        "mean_confidence": float(per_clip["mean_confidence"].mean()),
        "fleiss_kappa": fleiss,
        "pairwise_cohens_kappa_mean": float(pairwise["cohens_kappa"].mean()),
        "pairwise_cohens_kappa_min": float(pairwise["cohens_kappa"].min()),
        "pairwise_cohens_kappa_max": float(pairwise["cohens_kappa"].max()),
        "pairwise_raw_agreement_mean": float(pairwise["raw_agreement"].mean()),
        "majority_label_distribution": per_clip["majority_label"].value_counts().to_dict(),
    }
    return per_clip, pairwise, summary


def fleiss_kappa_from_votes(vote_matrix: np.ndarray, *, n_classes: int) -> float:
    """Compute Fleiss' kappa from an items x raters integer label matrix."""
    if vote_matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D vote matrix, got {vote_matrix.shape}")
    n_items, n_raters = vote_matrix.shape
    if n_items == 0 or n_raters < 2:
        return float("nan")
    counts = np.zeros((n_items, n_classes), dtype=np.float64)
    for item_idx in range(n_items):
        counts[item_idx] = np.bincount(vote_matrix[item_idx], minlength=n_classes)
    p_j = counts.sum(axis=0) / float(n_items * n_raters)
    p_i = (np.square(counts).sum(axis=1) - n_raters) / float(n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_e_bar = float(np.square(p_j).sum())
    if math.isclose(1.0 - p_e_bar, 0.0):
        return float("nan")
    return float((p_bar - p_e_bar) / (1.0 - p_e_bar))


def load_source_reference(run_key: str, run_cfg: dict[str, str]) -> pd.DataFrame:
    metrics_path = Path(run_cfg["metrics"])
    if not metrics_path.exists():
        return pd.DataFrame()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    confusion = np.asarray(metrics.get("confusion_matrix", []), dtype=np.int64)
    rows = []
    if confusion.shape == (4, 4):
        true_counts = confusion.sum(axis=1)
        pred_counts = confusion.sum(axis=0)
        true_total = max(int(true_counts.sum()), 1)
        pred_total = max(int(pred_counts.sum()), 1)
        for class_id in CLASS_IDS:
            rows.append(
                {
                    "run": run_key,
                    "domain": "cmose_test_true",
                    "class_id": class_id,
                    "class_label": ID_TO_LABEL[class_id],
                    "count": int(true_counts[class_id]),
                    "proportion": float(true_counts[class_id] / true_total),
                    "mean_confidence": np.nan,
                    "mean_entropy": np.nan,
                }
            )
            rows.append(
                {
                    "run": run_key,
                    "domain": "cmose_test_predicted",
                    "class_id": class_id,
                    "class_label": ID_TO_LABEL[class_id],
                    "count": int(pred_counts[class_id]),
                    "proportion": float(pred_counts[class_id] / pred_total),
                    "mean_confidence": np.nan,
                    "mean_entropy": np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_shift(distributions: pd.DataFrame, run_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    private = distributions[distributions["domain"] == "private_accepted"]
    source_pred = distributions[distributions["domain"] == "cmose_test_predicted"]
    source_true = distributions[distributions["domain"] == "cmose_test_true"]
    for run_key in sorted(private["run"].unique()):
        private_run = private[private["run"] == run_key].set_index("class_id")
        source_pred_run = source_pred[source_pred["run"] == run_key].set_index("class_id")
        source_true_run = source_true[source_true["run"] == run_key].set_index("class_id")
        metrics = run_metrics.get(run_key, {})
        dominant_class_id = int(private_run["proportion"].idxmax())
        for class_id in CLASS_IDS:
            private_prop = float(private_run.loc[class_id, "proportion"])
            source_pred_prop = (
                float(source_pred_run.loc[class_id, "proportion"])
                if class_id in source_pred_run.index
                else np.nan
            )
            source_true_prop = (
                float(source_true_run.loc[class_id, "proportion"])
                if class_id in source_true_run.index
                else np.nan
            )
            rows.append(
                {
                    "run": run_key,
                    "model": MODEL_RUNS[run_key]["model"],
                    "class_id": class_id,
                    "class_label": ID_TO_LABEL[class_id],
                    "private_proportion": private_prop,
                    "source_predicted_proportion": source_pred_prop,
                    "source_true_proportion": source_true_prop,
                    "private_minus_source_predicted": private_prop - source_pred_prop,
                    "private_minus_source_true": private_prop - source_true_prop,
                    "private_mean_confidence": float(private_run.loc[class_id, "mean_confidence"]),
                    "private_mean_entropy": float(private_run.loc[class_id, "mean_entropy"]),
                    "is_private_dominant": class_id == dominant_class_id,
                    "source_test_accuracy": metrics.get("accuracy"),
                    "source_test_macro_accuracy": metrics.get("macro_accuracy"),
                    "source_test_f1_macro": metrics.get("f1_macro"),
                }
            )
    return pd.DataFrame(rows)


def load_run_metrics(run_cfg: dict[str, str]) -> dict[str, float]:
    metrics_path = Path(run_cfg["metrics"])
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_accuracy": metrics.get("macro_accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
    }


def plot_prediction_distribution(
    distributions: pd.DataFrame,
    output_path: Path,
    *,
    domain: str,
    ylabel: str,
) -> None:
    predicted = distributions[distributions["domain"] == domain]
    pivot = (
        predicted.pivot_table(index="run", columns="class_label", values="proportion", fill_value=0.0)
        .reindex(columns=CLASS_LABELS, fill_value=0.0)
    )
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20c")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("CMOSE-trained run")
    ax.set_ylim(0, 1)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        title="Predicted class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def plot_private_distribution(distributions: pd.DataFrame, output_path: Path) -> None:
    plot_prediction_distribution(
        distributions,
        output_path,
        domain="private_accepted",
        ylabel="Private accepted prediction proportion",
    )


def plot_shift_heatmap(shift: pd.DataFrame, output_path: Path) -> None:
    pivot = shift.pivot(index="run", columns="class_label", values="private_minus_source_predicted")[
        CLASS_LABELS
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(pivot.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Private proportion minus CMOSE test predicted proportion")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    *,
    output_path: Path,
    samples: list[PrivateSample],
    run_metrics: dict[str, dict[str, Any]],
    shift: pd.DataFrame,
    distributions: pd.DataFrame,
    agreement_summary: dict[str, Any] | None = None,
    pairwise_kappa: pd.DataFrame | None = None,
) -> None:
    private = distributions[distributions["domain"] == "private_accepted"].copy()
    lines = [
        "# Domain Shift Analysis",
        "",
        "This report applies retained CMOSE-trained models to the private accepted subset.",
        "",
        f"- Accepted private clips analyzed: {len(samples)}",
        "- Target labels are unavailable, so findings diagnose prediction behavior, confidence, and distribution shift rather than target accuracy.",
        "- Source reference uses each run's saved CMOSE test confusion matrix.",
        "",
        "## CMOSE Test Reference",
        "",
        "| Run | Accuracy | Macro Accuracy | F1 Macro |",
        "|---|---:|---:|---:|",
    ]
    for run_key in sorted(run_metrics):
        metrics = run_metrics[run_key]
        lines.append(
            "| "
            + run_key
            + f" | {metrics.get('accuracy', float('nan')):.4f}"
            + f" | {metrics.get('macro_accuracy', float('nan')):.4f}"
            + f" | {metrics.get('f1_macro', float('nan')):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Private Prediction Distribution",
            "",
            "| Run | HD | DE | EG | HE | Dominant | Mean Confidence | Mean Entropy |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    label_short = {
        "Highly Disengage": "HD",
        "Disengage": "DE",
        "Engage": "EG",
        "Highly Engage": "HE",
    }
    for run_key in sorted(private["run"].unique()):
        run_private = private[private["run"] == run_key].set_index("class_label")
        props = [float(run_private.loc[label, "proportion"]) for label in CLASS_LABELS]
        dominant_label = CLASS_LABELS[int(np.argmax(props))]
        pred_rows = distributions[
            (distributions["domain"] == "private_accepted") & (distributions["run"] == run_key)
        ]
        pred_count = max(int(pred_rows["count"].sum()), 1)
        mean_conf = float((pred_rows["mean_confidence"] * pred_rows["count"]).sum() / pred_count)
        mean_entropy = float((pred_rows["mean_entropy"] * pred_rows["count"]).sum() / pred_count)
        lines.append(
            "| "
            + run_key
            + " | "
            + " | ".join(f"{prop:.3f}" for prop in props)
            + f" | {label_short[dominant_label]} | {mean_conf:.3f} | {mean_entropy:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Largest Distribution Shifts",
            "",
            "Positive values mean the class is predicted more often on private clips than on the same run's CMOSE test predictions.",
            "",
            "| Run | Class | Private - CMOSE Predicted | Private Proportion | CMOSE Predicted Proportion |",
            "|---|---|---:|---:|---:|",
        ]
    )
    top_shift = shift.assign(
        abs_shift=shift["private_minus_source_predicted"].abs()
    ).sort_values("abs_shift", ascending=False)
    for row in top_shift.head(12).itertuples(index=False):
        lines.append(
            f"| {row.run} | {row.class_label} | {row.private_minus_source_predicted:+.3f} "
            f"| {row.private_proportion:.3f} | {row.source_predicted_proportion:.3f} |"
        )

    if agreement_summary is not None and pairwise_kappa is not None:
        lines.extend(
            [
                "",
                "## Cross-Model Agreement",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Mean agreement rate | {agreement_summary['agreement_rate_mean']:.4f} |",
                f"| Median agreement rate | {agreement_summary['agreement_rate_median']:.4f} |",
                f"| Mean prediction entropy | {agreement_summary['prediction_entropy_mean']:.4f} |",
                f"| Median prediction entropy | {agreement_summary['prediction_entropy_median']:.4f} |",
                f"| Mean confidence | {agreement_summary['mean_confidence']:.4f} |",
                f"| Fleiss' kappa | {agreement_summary['fleiss_kappa']:.4f} |",
                f"| Mean pairwise Cohen's kappa | {agreement_summary['pairwise_cohens_kappa_mean']:.4f} |",
                f"| Mean pairwise raw agreement | {agreement_summary['pairwise_raw_agreement_mean']:.4f} |",
                "",
                "Lowest pairwise Cohen's kappa values indicate the model pairs that disagree most often across private clips.",
                "",
                "| Model A | Model B | Cohen's Kappa | Raw Agreement |",
                "|---|---|---:|---:|",
            ]
        )
        for row in pairwise_kappa.sort_values("cohens_kappa").head(8).itertuples(index=False):
            lines.append(
                f"| {row.run_a} | {row.run_b} | {row.cohens_kappa:.4f} | {row.raw_agreement:.4f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The analysis should be read as domain-shift diagnosis, not target-domain performance evaluation.",
            "- A class is treated as unstable when it shows a large private-vs-source prediction proportion shift and/or low private confidence.",
            "- Minority CMOSE classes remain especially hard to interpret without private labels; large HD/HE swings should be discussed as hypothesis-generating evidence.",
            "",
            "## Output Files",
            "",
            "- `outputs/domain_shift_analysis/prediction_distribution.csv`",
            "- `outputs/domain_shift_analysis/domain_shift_by_class.csv`",
            "- `outputs/domain_shift_analysis/domain_shift_summary.json`",
            "- `outputs/domain_shift_analysis/private_vs_source_predicted_shift.png`",
            "- `outputs/dataset_analysis/private/`",
            "- `outputs/dataset_analysis/cmose/`",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root = Path(args.dataset_output_root)
    private_dataset_dir = dataset_output_root / "private"
    cmose_dataset_dir = dataset_output_root / "cmose"
    document_path = Path(args.document_path)
    device = resolve_device(args.device)

    unknown_runs = [run for run in args.runs if run not in MODEL_RUNS]
    if unknown_runs:
        raise ValueError(f"Unknown run key(s): {unknown_runs}. Available: {sorted(MODEL_RUNS)}")

    print(f"Using device: {device}")
    samples = load_private_samples(args.accepted_csv)
    sample_ids = [sample.clip_id for sample in samples]
    print(f"Accepted private clips: {len(samples)}")

    if args.from_existing:
        predictions_path = private_dataset_dir / "predictions.csv"
        distributions_path = output_dir / "prediction_distribution.csv"
        shift_path = output_dir / "domain_shift_by_class.csv"
        for path in (predictions_path, distributions_path, shift_path):
            if not path.exists():
                raise FileNotFoundError(f"Cannot use --from_existing; missing {path}")
        distributions = pd.read_csv(distributions_path)
        shift = pd.read_csv(shift_path)
        predictions = pd.read_csv(predictions_path)
        agreement_per_clip, pairwise_kappa, agreement_summary = compute_model_agreement(
            predictions,
            expected_runs=args.runs,
        )
        run_metrics = {run_key: load_run_metrics(MODEL_RUNS[run_key]) for run_key in args.runs}
        plot_shift_heatmap(shift, output_dir / "private_vs_source_predicted_shift.png")
        write_markdown_report(
            output_path=document_path,
            samples=samples,
            run_metrics=run_metrics,
            shift=shift,
            distributions=distributions,
            agreement_summary=agreement_summary,
            pairwise_kappa=pairwise_kappa,
        )
        print(f"Regenerated plots from {output_dir}")
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
        private_prediction_tables.append(prediction_frame(run_key, sample_ids, private_probs))
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
    cmose_true_distribution = true_distribution_frame(
        cmose_sample_ids,
        cmose_true_labels,
        domain="cmose_test_true",
    )
    source_true_tables = []
    for run_key in args.runs:
        run_true_distribution = cmose_true_distribution.copy()
        run_true_distribution["run"] = run_key
        source_true_tables.append(run_true_distribution)
    source_distribution = pd.concat([cmose_distribution, *source_true_tables], ignore_index=True)
    distributions = pd.concat([private_distribution, source_distribution], ignore_index=True)
    shift = summarize_shift(distributions, run_metrics)
    agreement_per_clip, pairwise_kappa, agreement_summary = write_dataset_analysis_outputs(
        output_dir=private_dataset_dir,
        dataset_name="private",
        predictions=predictions,
        distributions=private_distribution.assign(domain="private_predicted"),
        expected_runs=args.runs,
    )
    write_dataset_analysis_outputs(
        output_dir=cmose_dataset_dir,
        dataset_name="cmose",
        predictions=cmose_predictions,
        distributions=cmose_distribution.assign(domain="cmose_predicted"),
        expected_runs=args.runs,
        true_labels=cmose_true_labels,
        sample_ids=cmose_sample_ids,
    )

    distributions.to_csv(output_dir / "prediction_distribution.csv", index=False)
    shift.to_csv(output_dir / "domain_shift_by_class.csv", index=False)

    summary = {
        "accepted_private_clips": len(samples),
        "cmose_test_clips": len(test_records),
        "runs": args.runs,
        "device": device.type,
        "target_frames": args.target_frames,
        "fusion_frames": args.fusion_frames,
        "run_metrics": run_metrics,
        "private_distribution": private_distribution.to_dict(orient="records"),
        "agreement_summary": agreement_summary,
        "largest_absolute_shifts": shift.assign(
            abs_shift=shift["private_minus_source_predicted"].abs()
        )
        .sort_values("abs_shift", ascending=False)
        .head(12)
        .drop(columns=["abs_shift"])
        .to_dict(orient="records"),
    }
    (output_dir / "domain_shift_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_shift_heatmap(shift, output_dir / "private_vs_source_predicted_shift.png")
    write_markdown_report(
        output_path=document_path,
        samples=samples,
        run_metrics=run_metrics,
        shift=shift,
        distributions=distributions,
        agreement_summary=agreement_summary,
        pairwise_kappa=pairwise_kappa,
    )
    print(f"Saved domain-shift outputs to {output_dir}")
    print(f"Saved markdown report to {document_path}")


if __name__ == "__main__":
    main()
