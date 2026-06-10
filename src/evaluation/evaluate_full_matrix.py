"""Full 3 × 3 training-group × test-set evaluation matrix.

Training groups : cmose | daisee | combined
Test sets       : cmose_test | daisee_test | private
Models × losses : 6 × 3 = 18 checkpoints per training group

6 reported metrics per cell:
  accuracy  |  mae  |  cohen_kappa  |  macro_accuracy  |  macro_mae  |  quadratic_weighted_kappa

Output:
  outputs/model_assessment/full_matrix.csv

Training groups with no checkpoints are skipped automatically.
Re-running overwrites the output CSV.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import agreement_metrics_from_confusion, label_distance_metrics_from_confusion
from src.feature_extraction.extract_i3d import load_i3d_dataset_matrices, resolve_i3d_feature_path, load_i3d_matrix
from src.feature_extraction.extract_openface import (
    LABEL_MAP,
    load_cmose_metadata,
    load_openface_matrix,
    resample_frames,
)
from src.models.models import build_model
from src.output_paths import MODEL_ASSESSMENT_DIR, NAIVE_ASSESSMENT_DIR, MANUAL_LABELS_CSV, model_output_name, loss_slug

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fusion (openface_tcn_i3d_fusion) is intentionally excluded from training and assessment.
MODELS = ["openface_mlp", "temporal_cnn", "lstm", "transformer", "i3d_mlp"]
LOSSES = ["cross_entropy", "weighted_cross_entropy", "ordinal"]
OPENFACE_MODELS = {"openface_mlp", "temporal_cnn", "lstm", "transformer"}
I3D_MODELS = {"i3d_mlp"}
FUSION_MODELS = {"openface_tcn_i3d_fusion"}
CLASS_IDS = list(range(4))
I3D_FEAT_DIM = 1024

METRIC_COLS = ["accuracy", "macro_accuracy", "mae", "macro_mae", "cohen_kappa", "quadratic_weighted_kappa"]


@dataclass(frozen=True)
class TrainConfig:
    name: str
    training_root: Path
    labels_json: Path
    openface_dir: Path
    i3d_dir: Path
    target_frames: int = 300   # OpenFace frame count used at training time


@dataclass(frozen=True)
class TestConfig:
    name: str
    labels_json: Path | None
    openface_dir: Path | None
    i3d_dir: Path
    split: str | None                  # for json-split test sets
    accepted_csv: Path | None          # for private
    manual_labels_csv: Path | None     # for private


TRAIN_CONFIGS: list[TrainConfig] = [
    TrainConfig(
        name="cmose",
        training_root=REPO_ROOT / "outputs/training_log/cmose",
        labels_json=REPO_ROOT / "data/CMOSE/final_data_1.json",
        openface_dir=REPO_ROOT / "data/CMOSE/secondFeature",
        i3d_dir=REPO_ROOT / "data/CMOSE/features/i3d",
    ),
    TrainConfig(
        name="daisee",
        training_root=REPO_ROOT / "outputs/training_log/daisee",
        labels_json=REPO_ROOT / "data/DaiSEE/final_data_1.json",
        openface_dir=REPO_ROOT / "data/DaiSEE/features/openface",
        i3d_dir=REPO_ROOT / "data/DaiSEE/features/i3d",
    ),
    TrainConfig(
        name="combined",
        training_root=REPO_ROOT / "outputs/training_log/combined",
        labels_json=REPO_ROOT / "data/combined/final_data_1.json",
        openface_dir=REPO_ROOT / "data/combined/features/openface",
        i3d_dir=REPO_ROOT / "data/combined/features/i3d",
        target_frames=150,   # reduced from 300 to fit combined dataset in RAM
    ),
]

TEST_CONFIGS: list[TestConfig] = [
    TestConfig(
        name="cmose_test",
        labels_json=REPO_ROOT / "data/CMOSE/final_data_1.json",
        openface_dir=REPO_ROOT / "data/CMOSE/secondFeature",
        i3d_dir=REPO_ROOT / "data/CMOSE/features/i3d",
        split="test",
        accepted_csv=None,
        manual_labels_csv=None,
    ),
    TestConfig(
        name="daisee_test",
        labels_json=REPO_ROOT / "data/DaiSEE/final_data_1.json",
        openface_dir=REPO_ROOT / "data/DaiSEE/features/openface",
        i3d_dir=REPO_ROOT / "data/DaiSEE/features/i3d",
        split="test",
        accepted_csv=None,
        manual_labels_csv=None,
    ),
    TestConfig(
        name="private",
        labels_json=None,
        openface_dir=None,
        i3d_dir=REPO_ROOT / "data/private/features/i3d",
        split=None,
        accepted_csv=REPO_ROOT / "data/private/accepted.csv",
        manual_labels_csv=MANUAL_LABELS_CSV,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def _fit_openface_norm(records, *, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
    import logging
    feat_sum = feat_sq = None
    count = 0
    skipped = 0
    for rec in tqdm(records, desc="  Fitting OpenFace norm", unit="sample", leave=False):
        try:
            m = load_openface_matrix(rec.csv_path, target_frames=target_frames).astype(np.float64)
        except Exception:
            skipped += 1
            continue
        if feat_sum is None:
            feat_sum = np.zeros(m.shape[1], dtype=np.float64)
            feat_sq  = np.zeros(m.shape[1], dtype=np.float64)
        feat_sum += m.sum(axis=0)
        feat_sq  += np.square(m).sum(axis=0)
        count    += m.shape[0]
    if skipped:
        logging.getLogger(__name__).warning("  _fit_openface_norm: skipped %d bad samples", skipped)
    mean = feat_sum / count
    std  = np.sqrt(np.maximum(feat_sq / count - np.square(mean), 0.0))
    std[std <= 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _fit_i3d_norm(sample_ids: list[str], *, feature_dir: Path, target_frames: int) -> tuple[np.ndarray, np.ndarray]:
    import logging
    feat_sum = feat_sq = None
    count = 0
    skipped = 0
    for sid in tqdm(sample_ids, desc="  Fitting I3D norm", unit="sample", leave=False):
        try:
            m = load_i3d_matrix(
                resolve_i3d_feature_path(sid, feature_dir), target_frames=target_frames
            ).astype(np.float64)
        except Exception:
            skipped += 1
            continue
        if feat_sum is None:
            feat_sum = np.zeros(m.shape[1], dtype=np.float64)
            feat_sq  = np.zeros(m.shape[1], dtype=np.float64)
        feat_sum += m.sum(axis=0)
        feat_sq  += np.square(m).sum(axis=0)
        count    += m.shape[0]
    if skipped:
        logging.getLogger(__name__).warning("  _fit_i3d_norm: skipped %d missing I3D samples", skipped)
    mean = feat_sum / count
    std  = np.sqrt(np.maximum(feat_sq / count - np.square(mean), 0.0))
    std[std <= 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# Test-set loading
# ---------------------------------------------------------------------------

@dataclass
class TestFeatures:
    sample_ids: list[str]
    true_labels: list[int | None]   # None means unlabeled
    of_300: np.ndarray | None       # (N, 300, 709) normalized
    of_fusion: np.ndarray | None    # (N, fusion_frames, 709) normalized
    i3d: np.ndarray | None          # (N, fusion_frames, 1024) normalized


def _load_test_from_json(
    cfg: TestConfig,
    *,
    of_mean: np.ndarray,
    of_std: np.ndarray,
    i3d_mean: np.ndarray,
    i3d_std: np.ndarray,
    target_frames: int,
    fusion_frames: int,
) -> TestFeatures:
    records = load_cmose_metadata(
        cfg.labels_json, cfg.openface_dir, allowed_splits=(cfg.split,)
    )
    if not records:
        return TestFeatures([], [], None, None, None)

    of_list: list[np.ndarray] = []
    labels: list[int] = []
    valid_records = []
    for rec in tqdm(records, desc=f"  Loading {cfg.name} OpenFace", leave=False):
        try:
            m = load_openface_matrix(rec.csv_path, target_frames=target_frames)
            of_list.append(m)
            labels.append(rec.label_id)
            valid_records.append(rec)
        except Exception:
            pass

    if not of_list:
        return TestFeatures([], [], None, None, None)

    of_raw  = np.stack(of_list, axis=0).astype(np.float32)
    of_300  = _norm(of_raw, of_mean, of_std)
    of_fusion = np.stack(
        [resample_frames(s, target_frames=fusion_frames) for s in of_300], axis=0
    ).astype(np.float32)
    del of_raw; gc.collect()

    sample_ids = [r.sample_id for r in valid_records]
    i3d_raw = load_i3d_dataset_matrices(
        sample_ids, feature_dir=cfg.i3d_dir, target_frames=1,  # I3D is one clip vector; no tiling
        progress_desc=f"  Loading {cfg.name} I3D",
    )
    i3d = _norm(i3d_raw, i3d_mean, i3d_std)
    del i3d_raw; gc.collect()

    return TestFeatures(
        sample_ids=sample_ids,
        true_labels=labels,
        of_300=of_300,
        of_fusion=of_fusion,
        i3d=i3d,
    )


def _load_private_test(
    cfg: TestConfig,
    *,
    of_mean: np.ndarray,
    of_std: np.ndarray,
    i3d_mean: np.ndarray,
    i3d_std: np.ndarray,
    target_frames: int,
    fusion_frames: int,
) -> TestFeatures:
    df = pd.read_csv(cfg.accepted_csv)
    accepted = df[df["is_accepted"].astype(str) == "1"]

    manual: dict[str, int] = {}
    if cfg.manual_labels_csv and cfg.manual_labels_csv.exists():
        mdf = pd.read_csv(cfg.manual_labels_csv)
        if {"clip_id", "manual_label_id"}.issubset(mdf.columns):
            labeled = mdf[mdf["manual_label_id"].notna()].copy()
            notes = labeled["notes"].fillna("").astype(str) if "notes" in labeled.columns else pd.Series([""] * len(labeled))
            labeled = labeled[~notes.str.contains("Delete", case=False, na=False)]
            manual = {str(r.clip_id): int(r.manual_label_id) for r in labeled.itertuples(index=False)}

    of_list: list[np.ndarray] = []
    sample_ids: list[str] = []
    true_labels: list[int | None] = []

    for row in tqdm(accepted.itertuples(index=False), total=len(accepted),
                    desc="  Loading private OpenFace", leave=False):
        csv_path = Path(str(row.openface_csv).replace("\\", "/"))
        if not csv_path.exists():
            continue
        try:
            m = load_openface_matrix(csv_path, target_frames=target_frames)
            of_list.append(m)
            clip_id = str(row.clip_id)
            sample_ids.append(clip_id)
            true_labels.append(manual.get(clip_id))
        except Exception:
            pass

    if not of_list:
        return TestFeatures([], [], None, None, None)

    of_raw    = np.stack(of_list, axis=0).astype(np.float32)
    of_300    = _norm(of_raw, of_mean, of_std)
    of_fusion = np.stack(
        [resample_frames(s, target_frames=fusion_frames) for s in of_300], axis=0
    ).astype(np.float32)
    del of_raw; gc.collect()

    i3d_raw = load_i3d_dataset_matrices(
        sample_ids, feature_dir=cfg.i3d_dir, target_frames=1,  # I3D is one clip vector; no tiling
        progress_desc="  Loading private I3D",
    )
    i3d = _norm(i3d_raw, i3d_mean, i3d_std)
    del i3d_raw; gc.collect()

    return TestFeatures(
        sample_ids=sample_ids,
        true_labels=true_labels,
        of_300=of_300,
        of_fusion=of_fusion,
        i3d=i3d,
    )


# ---------------------------------------------------------------------------
# Inference + metrics
# ---------------------------------------------------------------------------

def _predict_probs(
    model: torch.nn.Module,
    features: np.ndarray | tuple[np.ndarray, np.ndarray],
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


def per_clip_prediction_rows(
    probs: np.ndarray,
    sample_ids: list[str],
    true_labels: list[int | None],
    *,
    base: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one raw prediction row per clip for a single train x test x model cell."""
    pred_ids = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    rows: list[dict[str, Any]] = []
    for i, clip_id in enumerate(sample_ids):
        true = true_labels[i]
        row: dict[str, Any] = dict(base)
        row["clip_id"] = clip_id
        row["true_id"] = int(true) if true is not None else ""
        row["predicted_id"] = int(pred_ids[i])
        row["is_correct"] = bool(pred_ids[i] == true) if true is not None else ""
        row["confidence"] = float(confidences[i])
        for class_id in CLASS_IDS:
            row[f"prob_{class_id}"] = float(probs[i, class_id])
        rows.append(row)
    return rows


def _compute_metrics(probs: np.ndarray, true_labels: list[int | None]) -> dict[str, float] | None:
    """Compute 6 metrics on the labeled subset. Returns None if no labels available."""
    labeled_mask = [i for i, lbl in enumerate(true_labels) if lbl is not None]
    if not labeled_mask:
        return None
    y_true = np.array([true_labels[i] for i in labeled_mask], dtype=int)
    y_pred = probs[labeled_mask].argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    row_sums = cm.sum(axis=1, keepdims=True)
    per_class_acc = np.divide(
        cm.diagonal().astype(float), cm.sum(axis=1).astype(float),
        out=np.zeros(len(CLASS_IDS), dtype=float),
        where=cm.sum(axis=1) > 0,
    )
    from sklearn.metrics import accuracy_score
    return {
        "n_labeled": len(labeled_mask),
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "macro_accuracy":     float(per_class_acc.mean()),
        **label_distance_metrics_from_confusion(cm),
        **agreement_metrics_from_confusion(cm),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target_frames", type=int, default=300)
    p.add_argument("--fusion_frames",  type=int, default=75)
    p.add_argument("--batch_size",     type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output_dir", default=str(NAIVE_ASSESSMENT_DIR))
    p.add_argument(
        "--only_test_set",
        default=None,
        choices=["cmose_test", "daisee_test", "private"],
        help="If set, evaluate only this test set and merge results into the existing CSV.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args   = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    all_rows: list[dict[str, Any]] = []
    per_clip_rows: list[dict[str, Any]] = []
    active_test_configs = (
        [tc for tc in TEST_CONFIGS if tc.name == args.only_test_set]
        if args.only_test_set else TEST_CONFIGS
    )

    for train_cfg in TRAIN_GROUPS_ACTIVE(TRAIN_CONFIGS):
        # Discover checkpoints
        ckpts: list[dict] = []
        for model in MODELS:
            for loss in LOSSES:
                ckpt = train_cfg.training_root / model_output_name(model) / loss_slug(loss) / "best_model.pth"
                if ckpt.exists():
                    ckpts.append({"model": model, "loss": loss, "ckpt": ckpt})
        if not ckpts:
            print(f"\n[skip] {train_cfg.name}: no checkpoints in {train_cfg.training_root}")
            continue
        print(f"\n{'='*60}")
        print(f"Train group: {train_cfg.name}  ({len(ckpts)} checkpoints)")

        # Fit normalizers from training split
        train_records = load_cmose_metadata(
            train_cfg.labels_json, train_cfg.openface_dir, allowed_splits=("train",)
        )
        if not train_records:
            print(f"  [skip] no train records found for {train_cfg.name}")
            continue
        print(f"  Train samples: {len(train_records)}")

        target_frames = train_cfg.target_frames
        of_mean, of_std = _fit_openface_norm(train_records, target_frames=target_frames)
        train_ids = [r.sample_id for r in train_records]
        i3d_mean, i3d_std = _fit_i3d_norm(train_ids, feature_dir=train_cfg.i3d_dir,
                                            target_frames=1)  # I3D is one clip vector; no tiling

        for test_cfg in active_test_configs:
            print(f"\n  Test set: {test_cfg.name}")

            # Load and normalize test features
            if test_cfg.accepted_csv is not None:
                test_feats = _load_private_test(
                    test_cfg,
                    of_mean=of_mean, of_std=of_std,
                    i3d_mean=i3d_mean, i3d_std=i3d_std,
                    target_frames=target_frames,
                    fusion_frames=args.fusion_frames,
                )
            else:
                test_feats = _load_test_from_json(
                    test_cfg,
                    of_mean=of_mean, of_std=of_std,
                    i3d_mean=i3d_mean, i3d_std=i3d_std,
                    target_frames=target_frames,
                    fusion_frames=args.fusion_frames,
                )

            if not test_feats.sample_ids:
                print(f"    [skip] no samples loaded for {test_cfg.name}")
                continue

            n_labeled = sum(1 for lbl in test_feats.true_labels if lbl is not None)
            print(f"    Samples: {len(test_feats.sample_ids)}  labeled: {n_labeled}")

            # Run all checkpoints
            for cfg_item in ckpts:
                model_name = cfg_item["model"]
                loss_name  = cfg_item["loss"]

                i3d_dim = test_feats.i3d.shape[-1] if test_feats.i3d is not None else None
                model, _ = build_model(
                    model_name,
                    input_features=709,
                    i3d_input_features=i3d_dim if model_name in FUSION_MODELS else None,
                    num_classes=4,
                )
                state = torch.load(cfg_item["ckpt"], map_location=device, weights_only=True)
                model.load_state_dict(state)

                if model_name in FUSION_MODELS:
                    feats: Any = (test_feats.of_fusion, test_feats.i3d)
                elif model_name in I3D_MODELS:
                    feats = test_feats.i3d
                else:
                    feats = test_feats.of_300

                probs   = _predict_probs(model, feats, batch_size=args.batch_size, device=device)
                metrics = _compute_metrics(probs, test_feats.true_labels)

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                cell_base = {
                    "train_group": train_cfg.name,
                    "test_set":    test_cfg.name,
                    "model":       model_output_name(model_name),
                    "loss":        loss_slug(loss_name),
                }
                per_clip_rows.extend(
                    per_clip_prediction_rows(
                        probs, test_feats.sample_ids, test_feats.true_labels, base=cell_base
                    )
                )

                row: dict[str, Any] = {
                    **cell_base,
                    "target_frames": target_frames,
                    "n_labeled":     metrics["n_labeled"] if metrics else 0,
                }
                if metrics:
                    for col in METRIC_COLS:
                        row[col] = metrics.get(col, float("nan"))
                else:
                    for col in METRIC_COLS:
                        row[col] = float("nan")
                all_rows.append(row)

            del test_feats; gc.collect()

        del of_mean, of_std, i3d_mean, i3d_std; gc.collect()

    if not all_rows:
        print("\nNo results generated — run training first.")
        return

    results = pd.DataFrame(all_rows)
    out_csv = out / "full_matrix.csv"

    if args.only_test_set and out_csv.exists():
        existing = pd.read_csv(out_csv)
        existing = existing[existing["test_set"] != args.only_test_set]
        results = pd.concat([existing, results], ignore_index=True)

    results.to_csv(out_csv, index=False)

    # Raw per-clip predictions (one row per train x test x model x clip).
    per_clip_df = pd.DataFrame(per_clip_rows)
    per_clip_csv = out / "full_matrix_predictions.csv"
    if args.only_test_set and per_clip_csv.exists():
        existing_pc = pd.read_csv(per_clip_csv)
        existing_pc = existing_pc[existing_pc["test_set"] != args.only_test_set]
        per_clip_df = pd.concat([existing_pc, per_clip_df], ignore_index=True)
    per_clip_df.to_csv(per_clip_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Full matrix saved -> {out_csv}")
    print(f"Shape: {results.shape}")
    print(f"Per-clip predictions saved -> {per_clip_csv}  ({len(per_clip_df)} rows)")

    display_cols = ["train_group", "test_set", "model", "loss"] + METRIC_COLS
    display_cols = [c for c in display_cols if c in results.columns]
    print("\n" + results[display_cols].to_string(index=False))


def TRAIN_GROUPS_ACTIVE(configs: list[TrainConfig]):
    for cfg in configs:
        if cfg.training_root.exists():
            yield cfg
        else:
            print(f"\n[skip] {cfg.name}: training root not found ({cfg.training_root})")


if __name__ == "__main__":
    main()
