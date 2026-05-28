"""Evaluate DaiSEE-trained and CMOSE-trained models on the private dataset side-by-side.

Each model group uses its own training-set statistics for normalization,
matching how each model was trained.

Outputs:
  outputs/model_assessment/private_daisee/supervised_metrics.csv
  outputs/model_assessment/private_cmose/supervised_metrics.csv
  outputs/model_assessment/private_comparison.csv   <- combined comparison table

Usage (after both training runs are complete):
  python scripts/evaluate_daisee_vs_cmose.py

Override paths if needed:
  python scripts/evaluate_daisee_vs_cmose.py \\
      --daisee_training_root outputs/training_log/daisee \\
      --daisee_labels_json   data/DaiSEE/final_data_1.json \\
      --daisee_openface_dir  data/DaiSEE/features/openface \\
      --daisee_i3d_dir       data/DaiSEE/features/i3d \\
      --cmose_training_root  outputs/training_log \\
      --cmose_labels_json    data/CMOSE/final_data_1.json \\
      --cmose_openface_dir   data/CMOSE/features/openface \\
      --cmose_i3d_dir        data/CMOSE/features/i3d
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import agreement_metrics_from_confusion, label_distance_metrics_from_confusion
from src.feature_extraction.extract_i3d import load_i3d_matrix, load_i3d_dataset_matrices, resolve_i3d_feature_path
from src.feature_extraction.extract_openface import (
    ID_TO_LABEL,
    load_cmose_metadata,
    load_dataset_matrices,
    load_openface_matrix,
    resample_frames,
)
from src.models.models import build_model
from src.output_paths import (
    MODEL_ASSESSMENT_DIR,
    MANUAL_LABELS_CSV,
    model_output_name,
    loss_slug,
)

MODELS = ["openface_mlp", "temporal_cnn", "lstm", "transformer", "i3d_mlp", "openface_tcn_i3d_fusion"]
LOSSES = ["cross_entropy", "weighted_cross_entropy", "ordinal"]
OPENFACE_MODELS = {"openface_mlp", "temporal_cnn", "lstm", "transformer"}
I3D_MODELS = {"i3d_mlp"}
FUSION_MODELS = {"openface_tcn_i3d_fusion"}
CLASS_IDS = list(range(4))


@dataclass(frozen=True)
class PrivateSample:
    clip_id: str
    clip_path: Path
    openface_csv: Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--daisee_training_root",  default="outputs/training_log/daisee")
    p.add_argument("--daisee_labels_json",    default="data/DaiSEE/final_data_1.json")
    p.add_argument("--daisee_openface_dir",   default="data/DaiSEE/features/openface")
    p.add_argument("--daisee_i3d_dir",        default="data/DaiSEE/features/i3d")
    p.add_argument("--cmose_training_root",   default="outputs/training_log")
    p.add_argument("--cmose_labels_json",     default="data/CMOSE/final_data_1.json")
    p.add_argument("--cmose_openface_dir",    default="data/CMOSE/features/openface")
    p.add_argument("--cmose_i3d_dir",         default="data/CMOSE/features/i3d")
    p.add_argument("--accepted_csv",          default="data/private/accepted.csv")
    p.add_argument("--private_i3d_dir",       default="data/private/features/i3d")
    p.add_argument("--manual_labels_csv",     default=str(MANUAL_LABELS_CSV))
    p.add_argument("--output_dir",            default=str(MODEL_ASSESSMENT_DIR))
    p.add_argument("--target_frames", type=int, default=300)
    p.add_argument("--fusion_frames",  type=int, default=75)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def load_private_samples(accepted_csv: str | Path) -> list[PrivateSample]:
    df = pd.read_csv(accepted_csv)
    accepted = df[df["is_accepted"].astype(str) == "1"]
    samples = []
    for row in accepted.itertuples(index=False):
        csv_path = Path(str(row.openface_csv).replace("\\", "/"))
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        samples.append(PrivateSample(
            clip_id=str(row.clip_id),
            clip_path=Path(str(row.clip_path).replace("\\", "/")),
            openface_csv=csv_path,
        ))
    return samples


def load_manual_labels(labels_csv: str | Path) -> dict[str, int]:
    p = Path(labels_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not {"clip_id", "manual_label_id", "notes"}.issubset(df.columns):
        return {}
    labeled = df[df["manual_label_id"].notna()].copy()
    notes = labeled["notes"].fillna("").astype(str)
    labeled = labeled[~notes.str.contains("Delete", case=False, na=False)]
    return {str(row.clip_id): int(row.manual_label_id) for row in labeled.itertuples(index=False)}


def _fit_openface_norm(records, *, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
    import logging
    feat_sum = feat_sq = None
    count = 0
    skipped = 0
    for rec in tqdm(records, desc="Fitting OpenFace normalizer", unit="sample", leave=False):
        try:
            m = load_openface_matrix(rec.csv_path, target_frames=target_frames).astype(np.float64)
        except Exception:
            skipped += 1
            continue
        if feat_sum is None:
            feat_sum = np.zeros(m.shape[1], dtype=np.float64)
            feat_sq  = np.zeros(m.shape[1], dtype=np.float64)
        feat_sum += m.sum(axis=0);  feat_sq += np.square(m).sum(axis=0);  count += m.shape[0]
    if skipped:
        logging.getLogger(__name__).warning("_fit_openface_norm: skipped %d bad samples", skipped)
    mean = feat_sum / count
    std  = np.sqrt(np.maximum(feat_sq / count - np.square(mean), 0.0))
    std[std <= 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _fit_i3d_norm(sample_ids: list[str], *, feature_dir: str | Path, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
    import logging
    feat_sum = feat_sq = None
    count = 0
    skipped = 0
    for sid in tqdm(sample_ids, desc="Fitting I3D normalizer", unit="sample", leave=False):
        try:
            m = load_i3d_matrix(resolve_i3d_feature_path(sid, feature_dir), target_frames=target_frames).astype(np.float64)
        except Exception:
            skipped += 1
            continue
        if feat_sum is None:
            feat_sum = np.zeros(m.shape[1], dtype=np.float64)
            feat_sq  = np.zeros(m.shape[1], dtype=np.float64)
        feat_sum += m.sum(axis=0);  feat_sq += np.square(m).sum(axis=0);  count += m.shape[0]
    if skipped:
        logging.getLogger(__name__).warning("_fit_i3d_norm: skipped %d missing samples", skipped)
    mean = feat_sum / count
    std  = np.sqrt(np.maximum(feat_sq / count - np.square(mean), 0.0))
    std[std <= 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def predict_probs(
    model: torch.nn.Module,
    features: np.ndarray | tuple[np.ndarray, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    arrays = features if isinstance(features, tuple) else (features,)
    dataset = TensorDataset(*(torch.from_numpy(a).float() for a in arrays))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         pin_memory=device.type == "cuda")
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            ts = [t.to(device, non_blocking=True) for t in batch]
            logits = model(*ts) if len(ts) > 1 else model(ts[0])
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _make_predictions_df(
    run_key: str,
    sample_ids: list[str],
    probs: np.ndarray,
    true_labels: list[int | None] | None,
) -> pd.DataFrame:
    pred_ids  = probs.argmax(axis=1)
    sorted_p  = np.sort(probs, axis=1)
    rows = []
    for i, clip_id in enumerate(sample_ids):
        row: dict[str, Any] = {
            "run": run_key,
            "clip_id": clip_id,
            "predicted_id": int(pred_ids[i]),
            "predicted_label": ID_TO_LABEL[int(pred_ids[i])],
            "confidence": float(probs[i].max()),
            "margin": float(sorted_p[i, -1] - sorted_p[i, -2]),
            "normalized_entropy": float(
                -np.sum(probs[i] * np.log(np.clip(probs[i], 1e-12, 1.0))) / math.log(probs.shape[1])
            ),
        }
        if true_labels is not None:
            tl = true_labels[i]
            if tl is None or (isinstance(tl, float) and np.isnan(tl)):
                row.update(true_id=np.nan, true_label="", is_correct=np.nan)
            else:
                row.update(true_id=int(tl), true_label=ID_TO_LABEL[int(tl)],
                           is_correct=bool(pred_ids[i] == int(tl)))
        for c in CLASS_IDS:
            row[f"prob_{c}_{ID_TO_LABEL[c]}"] = float(probs[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


def _supervised_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run_key, grp in predictions.groupby("run"):
        labeled = grp[grp["true_id"].notna()] if "true_id" in grp.columns else pd.DataFrame()
        if labeled.empty:
            continue
        y_true = labeled["true_id"].to_numpy(dtype=int)
        y_pred = labeled["predicted_id"].to_numpy(dtype=int)
        cm     = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
        per_class_acc = np.divide(
            cm.diagonal(), cm.sum(axis=1),
            out=np.zeros(len(CLASS_IDS), dtype=np.float64),
            where=cm.sum(axis=1) > 0,
        )
        rows.append({
            "run": run_key,
            "accuracy":       float(accuracy_score(y_true, y_pred)),
            "macro_accuracy": float(per_class_acc.mean()),
            "f1_macro":       float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro",    zero_division=0)),
            "f1_weighted":    float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0)),
            **label_distance_metrics_from_confusion(cm),
            **agreement_metrics_from_confusion(cm),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core: evaluate one training group against private dataset
# ---------------------------------------------------------------------------

def evaluate_group(
    *,
    tag: str,
    training_root: str | Path,
    labels_json: str | Path,
    openface_dir: str | Path,
    i3d_dir: str | Path,
    private_samples: list[PrivateSample],
    private_i3d_dir: str | Path,
    manual_labels: dict[str, int],
    target_frames: int,
    fusion_frames: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    """Run inference for all available checkpoints in training_root and return metrics."""
    training_root = Path(training_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids   = [s.clip_id for s in private_samples]
    true_labels  = [manual_labels.get(sid) for sid in sample_ids]

    # Discover available checkpoints
    run_configs: list[dict[str, Any]] = []
    for model in MODELS:
        for loss in LOSSES:
            ckpt = training_root / model_output_name(model) / loss_slug(loss) / "best_model.pth"
            if ckpt.exists():
                run_configs.append({"model": model, "loss": loss, "checkpoint": ckpt,
                                    "run_key": f"{tag}/{model_output_name(model)}/{loss_slug(loss)}"})

    if not run_configs:
        print(f"  [warn] No checkpoints found under {training_root}")
        return pd.DataFrame()

    print(f"  Found {len(run_configs)} checkpoints under {training_root}")

    # Load training records for normalizer fitting
    records = load_cmose_metadata(labels_json, openface_dir, allowed_splits=("train", "test", "unlabel"))
    train_records = [r for r in records if r.split == "train"]
    if not train_records:
        raise RuntimeError(f"No train records found in {labels_json}")

    # Determine which modalities we need
    need_openface = any(cfg["model"] in OPENFACE_MODELS | FUSION_MODELS for cfg in run_configs)
    need_i3d      = any(cfg["model"] in I3D_MODELS      | FUSION_MODELS for cfg in run_configs)

    # --- Load and normalize private OpenFace features ---
    priv_of_300 = priv_of_fusion = priv_i3d = None
    i3d_feat_dim: int | None = None

    if need_openface:
        print(f"  Fitting OpenFace normalizer from {tag} train split ({len(train_records)} samples)")
        of_mean, of_std = _fit_openface_norm(train_records, target_frames=target_frames)
        priv_of_raw = np.stack([
            load_openface_matrix(s.openface_csv, target_frames=target_frames)
            for s in tqdm(private_samples, desc="Loading private OpenFace", leave=False)
        ]).astype(np.float32)
        priv_of_300   = _norm(priv_of_raw, of_mean, of_std)
        priv_of_fusion = np.stack([
            resample_frames(s, target_frames=fusion_frames)
            for s in tqdm(priv_of_300, desc="Resampling for fusion", leave=False)
        ]).astype(np.float32)
        del priv_of_raw; gc.collect()

    if need_i3d:
        train_ids = [r.sample_id for r in train_records]
        print(f"  Fitting I3D normalizer from {tag} train split")
        i3d_mean, i3d_std = _fit_i3d_norm(train_ids, feature_dir=i3d_dir, target_frames=fusion_frames)
        priv_i3d_raw = load_i3d_dataset_matrices(
            sample_ids, feature_dir=private_i3d_dir, target_frames=fusion_frames,
            progress_desc="Loading private I3D",
        )
        priv_i3d     = _norm(priv_i3d_raw, i3d_mean, i3d_std)
        i3d_feat_dim = priv_i3d.shape[-1]
        del priv_i3d_raw; gc.collect()

    # --- Run inference ---
    all_preds: list[pd.DataFrame] = []
    for cfg in run_configs:
        model_name = cfg["model"]
        run_key    = cfg["run_key"]
        print(f"  Predicting [{run_key}]")

        model, _ = build_model(
            model_name,
            input_features=709,
            i3d_input_features=i3d_feat_dim if model_name in FUSION_MODELS else None,
            num_classes=4,
        )
        state = torch.load(cfg["checkpoint"], map_location=device)
        model.load_state_dict(state)

        if model_name in FUSION_MODELS:
            feats: np.ndarray | tuple = (priv_of_fusion, priv_i3d)
        elif model_name in I3D_MODELS:
            feats = priv_i3d
        else:
            feats = priv_of_300

        probs = predict_probs(model, feats, batch_size=batch_size, device=device)
        all_preds.append(_make_predictions_df(run_key, sample_ids, probs, true_labels))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    predictions = pd.concat(all_preds, ignore_index=True)
    metrics     = _supervised_metrics(predictions)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "supervised_metrics.csv", index=False)

    print(f"  Saved -> {output_dir}")
    if not metrics.empty:
        print(metrics[["run", "accuracy", "f1_macro", "mae"]].to_string(index=False))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args   = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    out    = Path(args.output_dir)

    print(f"Device: {device}")
    private_samples = load_private_samples(args.accepted_csv)
    manual_labels   = load_manual_labels(args.manual_labels_csv)
    print(f"Private clips: {len(private_samples)}  manual labels: {sum(v is not None for v in (manual_labels.get(s.clip_id) for s in private_samples))}")

    all_metrics: list[pd.DataFrame] = []

    # --- CMOSE-trained models ---
    if Path(args.cmose_training_root).exists():
        print("\n--- CMOSE-trained models ---")
        m = evaluate_group(
            tag="cmose",
            training_root=args.cmose_training_root,
            labels_json=args.cmose_labels_json,
            openface_dir=args.cmose_openface_dir,
            i3d_dir=args.cmose_i3d_dir,
            private_samples=private_samples,
            private_i3d_dir=args.private_i3d_dir,
            manual_labels=manual_labels,
            target_frames=args.target_frames,
            fusion_frames=args.fusion_frames,
            batch_size=args.batch_size,
            device=device,
            output_dir=out / "private_cmose",
        )
        if not m.empty:
            all_metrics.append(m)
    else:
        print(f"[skip] CMOSE training root not found: {args.cmose_training_root}")

    # --- DaiSEE-trained models ---
    if Path(args.daisee_training_root).exists():
        print("\n--- DaiSEE-trained models ---")
        m = evaluate_group(
            tag="daisee",
            training_root=args.daisee_training_root,
            labels_json=args.daisee_labels_json,
            openface_dir=args.daisee_openface_dir,
            i3d_dir=args.daisee_i3d_dir,
            private_samples=private_samples,
            private_i3d_dir=args.private_i3d_dir,
            manual_labels=manual_labels,
            target_frames=args.target_frames,
            fusion_frames=args.fusion_frames,
            batch_size=args.batch_size,
            device=device,
            output_dir=out / "private_daisee",
        )
        if not m.empty:
            all_metrics.append(m)
    else:
        print(f"[skip] DaiSEE training root not found: {args.daisee_training_root}")

    # --- Side-by-side comparison ---
    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        combined = combined.sort_values(["run"]).reset_index(drop=True)
        comparison_path = out / "private_comparison.csv"
        combined.to_csv(comparison_path, index=False)
        print(f"\n=== Side-by-side comparison saved -> {comparison_path} ===")
        cols = ["run", "accuracy", "macro_accuracy", "f1_macro", "f1_weighted", "mae", "mse",
                "cohen_kappa", "quadratic_weighted_kappa"]
        display_cols = [c for c in cols if c in combined.columns]
        print(combined[display_cols].to_string(index=False))
    else:
        print("No metrics to compare — run training first.")


if __name__ == "__main__":
    main()
