"""Evaluate the best SemanticGroupFusionModel checkpoint on all 3 test sets.

Loads the CMOSE train-split normalizer, runs the best hybrid checkpoint
(T_TCN_TCN_TCN_TCN, trained on CMOSE) on cmose_test / daisee_test / private,
and appends rows to outputs/model_assessment/full_matrix.csv using the same
6-metric format.

Usage:
    python scripts/evaluate_hybrid_model.py
    python scripts/evaluate_hybrid_model.py --arch_key T_TCN_TCN_TCN_TCN
    python scripts/evaluate_hybrid_model.py --append   # append to existing CSV (default)
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_full_matrix import (
    TEST_CONFIGS,
    _compute_metrics,
    _fit_openface_norm,
    _load_private_test,
    _load_test_from_json,
    _resolve_device,
    METRIC_COLS,
)
from src.feature_extraction.extract_openface import (
    build_group_column_indices,
    get_openface_feature_columns,
    load_cmose_metadata,
)
from src.models.models import build_semantic_group_model
from src.output_paths import HYBRID_ASSESSMENT_DIR

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CMOSE_LABELS_JSON = REPO_ROOT / "data/CMOSE/final_data_1.json"
CMOSE_OPENFACE_DIR = REPO_ROOT / "data/CMOSE/secondFeature"
HYBRID_TRAINING_ROOT = REPO_ROOT / "outputs/training_log/combined/semantic_group_fusion"
TARGET_FRAMES = 300
FUSION_FRAMES = 75

# Architecture key -> group architectures mapping
ARCH_KEYS = {
    "T_TCN_TCN_TCN_TCN": {"gaze": "transformer", "eye": "tcn", "face": "tcn", "head": "tcn", "au": "tcn"},
    "T_TCN_TCN_TCN_T":   {"gaze": "transformer", "eye": "tcn", "face": "tcn", "head": "tcn", "au": "transformer"},
    "TCN_TCN_T_TCN_TCN": {"gaze": "tcn", "eye": "tcn", "face": "transformer", "head": "tcn", "au": "tcn"},
    "TCN_T_T_TCN_TCN":   {"gaze": "tcn", "eye": "transformer", "face": "transformer", "head": "tcn", "au": "tcn"},
    "TCN_TCN_T_TCN_T":   {"gaze": "tcn", "eye": "tcn", "face": "transformer", "head": "tcn", "au": "transformer"},
}


def _predict(model: torch.nn.Module, features: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(features).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for (x,) in loader:
            logits = model(x.to(device, non_blocking=True))
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch_key", default="T_TCN_TCN_TCN_TCN",
                   help="Arch key to evaluate (subfolder name under hybrid training root).")
    p.add_argument("--train_group", default="cmose",
                   help="Label for train_group column in full_matrix.csv.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output_csv", default=str(HYBRID_ASSESSMENT_DIR / "hybrid_matrix.csv"))
    p.add_argument("--append", action="store_true", default=True,
                   help="Append rows to existing CSV (default). Use --no-append to overwrite.")
    p.add_argument("--no-append", dest="append", action="store_false")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    out_csv = Path(args.output_csv)

    # --- Locate checkpoint ---
    ckpt_dir = HYBRID_TRAINING_ROOT / args.arch_key
    ckpt_path = ckpt_dir / "best_model.pth"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    # Read group architectures from saved metrics.json
    metrics_path = ckpt_dir / "metrics.json"
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text())
        raw_arch = saved.get("config", {}).get("group_architectures")
        if raw_arch:
            import json as _j
            try:
                group_architectures = _j.loads(raw_arch)
            except Exception:
                group_architectures = ARCH_KEYS.get(args.arch_key)
        else:
            group_architectures = ARCH_KEYS.get(args.arch_key)
    else:
        group_architectures = ARCH_KEYS.get(args.arch_key)

    if group_architectures is None:
        print(f"Unknown arch_key '{args.arch_key}'. Add it to ARCH_KEYS or ensure metrics.json exists.")
        return

    print(f"Evaluating hybrid model: {args.arch_key}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Architectures: {group_architectures}")
    print(f"  Device: {device}")

    # --- Fit normalizer from CMOSE train split ---
    print("\nFitting CMOSE train normalizer...")
    train_records = load_cmose_metadata(
        CMOSE_LABELS_JSON, CMOSE_OPENFACE_DIR, allowed_splits=("train",)
    )
    print(f"  Train samples: {len(train_records)}")
    of_mean, of_std = _fit_openface_norm(train_records, target_frames=TARGET_FRAMES)

    # Fit dummy I3D norm (not used by the model, but needed for load helpers)
    i3d_mean = np.zeros(1024, dtype=np.float32)
    i3d_std = np.ones(1024, dtype=np.float32)

    # --- Build group indices ---
    first_csv = train_records[0].csv_path
    feature_columns = get_openface_feature_columns(first_csv)
    group_indices = build_group_column_indices(feature_columns)
    print(f"  Groups: { {k: len(v) for k, v in group_indices.items()} }")

    # --- Build and load model ---
    model, _ = build_semantic_group_model(
        group_indices=group_indices,
        group_architectures=group_architectures,
        num_classes=4,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # --- Evaluate on each test set ---
    new_rows: list[dict] = []

    for test_cfg in TEST_CONFIGS:
        print(f"\nTest set: {test_cfg.name}")
        if test_cfg.accepted_csv is not None:
            test_feats = _load_private_test(
                test_cfg,
                of_mean=of_mean, of_std=of_std,
                i3d_mean=i3d_mean, i3d_std=i3d_std,
                target_frames=TARGET_FRAMES,
                fusion_frames=FUSION_FRAMES,
            )
        else:
            test_feats = _load_test_from_json(
                test_cfg,
                of_mean=of_mean, of_std=of_std,
                i3d_mean=i3d_mean, i3d_std=i3d_std,
                target_frames=TARGET_FRAMES,
                fusion_frames=FUSION_FRAMES,
            )

        if not test_feats.sample_ids:
            print(f"  [skip] no samples loaded")
            continue

        n_labeled = sum(1 for lbl in test_feats.true_labels if lbl is not None)
        print(f"  Samples: {len(test_feats.sample_ids)}  labeled: {n_labeled}")

        probs = _predict(model, test_feats.of_300, batch_size=args.batch_size, device=device)
        metrics = _compute_metrics(probs, test_feats.true_labels)

        row: dict = {
            "train_group":   args.train_group,
            "test_set":      test_cfg.name,
            "model":         "semantic_group_fusion",
            "loss":          "ce",
            "target_frames": TARGET_FRAMES,
            "n_labeled":     metrics["n_labeled"] if metrics else 0,
            "arch_key":      args.arch_key,
        }
        if metrics:
            for col in METRIC_COLS:
                row[col] = metrics.get(col, float("nan"))
            print(f"  acc={metrics['accuracy']:.4f}  mac_acc={metrics['macro_accuracy']:.4f}  "
                  f"mae={metrics['mae']:.4f}  qwk={metrics['quadratic_weighted_kappa']:.4f}")
        else:
            for col in METRIC_COLS:
                row[col] = float("nan")
            print("  [no labeled samples — metrics are NaN]")

        new_rows.append(row)
        del test_feats
        gc.collect()

    if not new_rows:
        print("\nNo results generated.")
        return

    new_df = pd.DataFrame(new_rows)

    # --- Append or overwrite full_matrix.csv ---
    if args.append and out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Remove any previous hybrid rows with same train_group + arch_key
        if "arch_key" in existing.columns:
            mask = (existing["train_group"] == args.train_group) & \
                   (existing["model"] == "semantic_group_fusion") & \
                   (existing["arch_key"] == args.arch_key)
            existing = existing[~mask]
        else:
            mask = (existing["train_group"] == args.train_group) & \
                   (existing["model"] == "semantic_group_fusion")
            existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}  ({len(combined)} total rows)")
    print("\nNew hybrid rows:")
    print(new_df[["train_group", "test_set", "model", "arch_key"] + METRIC_COLS].to_string(index=False))


if __name__ == "__main__":
    main()
