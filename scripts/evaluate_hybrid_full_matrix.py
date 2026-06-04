"""Evaluate all hybrid model checkpoints across all train groups and test sets.

Discovers every arch-key subfolder under:
  outputs/training_log/<group>/openface_temporal_hybrid/
  outputs/training_log/<group>/openface_temporal_i3d_hybrid/

Evaluates each on all 3 test sets and saves to hybrid_matrix.csv.

Usage:
    python scripts/evaluate_hybrid_full_matrix.py
    python scripts/evaluate_hybrid_full_matrix.py --train_groups cmose daisee
    python scripts/evaluate_hybrid_full_matrix.py --model_types openface_temporal_i3d_hybrid
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

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
    TRAIN_CONFIGS,
    TEST_CONFIGS,
    METRIC_COLS,
    _compute_metrics,
    _fit_openface_norm,
    _fit_i3d_norm,
    _load_private_test,
    _load_test_from_json,
    _resolve_device,
    per_clip_prediction_rows,
)
from src.feature_extraction.extract_openface import (
    build_group_column_indices,
    get_openface_feature_columns,
    load_cmose_metadata,
)
from src.models.models import (
    build_openface_temporal_hybrid,
    build_openface_temporal_i3d_hybrid,
)
from src.output_paths import HYBRID_ASSESSMENT_DIR, model_output_name

HYBRID_MODEL_TYPES = ["openface_temporal_hybrid", "openface_temporal_i3d_hybrid"]


def _predict_of(model, of_300, *, batch_size, device):
    dataset = TensorDataset(torch.from_numpy(of_300).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=device.type == "cuda")
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for (x,) in loader:
            logits = model(x.to(device, non_blocking=True))
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _predict_of_i3d(model, of_300, i3d, *, batch_size, device):
    dataset = TensorDataset(torch.from_numpy(of_300).float(), torch.from_numpy(i3d).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        pin_memory=device.type == "cuda")
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for x_of, x_i3d in loader:
            logits = model(x_of.to(device, non_blocking=True),
                           x_i3d.to(device, non_blocking=True))
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _load_arch(ckpt_dir: Path) -> dict[str, str] | None:
    p = ckpt_dir / "metrics.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text()).get("config", {}).get("group_architectures")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_groups", nargs="+", default=["cmose", "daisee", "combined"],
                   choices=["cmose", "daisee", "combined"])
    p.add_argument("--model_types", nargs="+", default=HYBRID_MODEL_TYPES,
                   choices=HYBRID_MODEL_TYPES)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--output_dir", default=str(HYBRID_ASSESSMENT_DIR))
    p.add_argument("--fusion_frames", type=int, default=75)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg_map = {cfg.name: cfg for cfg in TRAIN_CONFIGS}
    all_rows: list[dict[str, Any]] = []
    per_clip_rows: list[dict[str, Any]] = []

    for group_name in args.train_groups:
        train_cfg = train_cfg_map.get(group_name)
        if train_cfg is None:
            print(f"[skip] unknown train group: {group_name}")
            continue

        # Gather checkpoints for all requested model types
        ckpt_dirs_by_type: dict[str, list[Path]] = {}
        for model_type in args.model_types:
            folder = REPO_ROOT / "outputs/training_log" / group_name / model_output_name(model_type)
            if folder.exists():
                dirs = sorted(d for d in folder.iterdir()
                              if d.is_dir() and (d / "best_model.pth").exists())
                if dirs:
                    ckpt_dirs_by_type[model_type] = dirs

        if not ckpt_dirs_by_type:
            print(f"[skip] {group_name}: no hybrid checkpoints found")
            continue

        total_ckpts = sum(len(v) for v in ckpt_dirs_by_type.values())
        print(f"\n{'='*60}")
        print(f"Train group: {group_name}  ({total_ckpts} checkpoints)  frames={train_cfg.target_frames}")

        # Fit OpenFace normalizer
        train_records = load_cmose_metadata(
            train_cfg.labels_json, train_cfg.openface_dir, allowed_splits=("train",)
        )
        if not train_records:
            print(f"  [skip] no train records for {group_name}")
            continue
        print(f"  Train samples: {len(train_records)}")
        of_mean, of_std = _fit_openface_norm(train_records, target_frames=train_cfg.target_frames)

        # Fit I3D normalizer (needed if any I3D model type is active)
        needs_i3d = "openface_temporal_i3d_hybrid" in ckpt_dirs_by_type
        if needs_i3d:
            train_ids = [r.sample_id for r in train_records]
            i3d_mean, i3d_std = _fit_i3d_norm(train_ids, feature_dir=train_cfg.i3d_dir,
                                               target_frames=args.fusion_frames)
        else:
            i3d_mean = np.zeros(1024, dtype=np.float32)
            i3d_std  = np.ones(1024,  dtype=np.float32)

        # Build group indices from first CSV
        feature_columns = get_openface_feature_columns(train_records[0].csv_path)
        group_indices = build_group_column_indices(feature_columns)

        # Load test sets
        test_features = {}
        for test_cfg in TEST_CONFIGS:
            print(f"\n  Loading test set: {test_cfg.name}")
            if test_cfg.accepted_csv is not None:
                tf = _load_private_test(test_cfg, of_mean=of_mean, of_std=of_std,
                                        i3d_mean=i3d_mean, i3d_std=i3d_std,
                                        target_frames=train_cfg.target_frames,
                                        fusion_frames=args.fusion_frames)
            else:
                tf = _load_test_from_json(test_cfg, of_mean=of_mean, of_std=of_std,
                                          i3d_mean=i3d_mean, i3d_std=i3d_std,
                                          target_frames=train_cfg.target_frames,
                                          fusion_frames=args.fusion_frames)
            n_lbl = sum(1 for l in tf.true_labels if l is not None)
            print(f"    Samples: {len(tf.sample_ids)}  labeled: {n_lbl}")
            test_features[test_cfg.name] = tf

        # Evaluate each model type
        for model_type, ckpt_dirs in ckpt_dirs_by_type.items():
            is_i3d = model_type == "openface_temporal_i3d_hybrid"
            for ckpt_dir in tqdm(ckpt_dirs, desc=f"  {group_name}/{model_type}"):
                arch_key = ckpt_dir.name
                group_architectures = _load_arch(ckpt_dir)
                if group_architectures is None:
                    continue

                if is_i3d:
                    model, _ = build_openface_temporal_i3d_hybrid(
                        group_indices=group_indices,
                        group_architectures=group_architectures,
                        num_classes=4,
                    )
                else:
                    model, _ = build_openface_temporal_hybrid(
                        group_indices=group_indices,
                        group_architectures=group_architectures,
                        num_classes=4,
                    )

                state = torch.load(ckpt_dir / "best_model.pth",
                                   map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.to(device).eval()

                for test_cfg in TEST_CONFIGS:
                    tf = test_features[test_cfg.name]
                    if tf.of_300 is None or len(tf.sample_ids) == 0:
                        continue

                    if is_i3d:
                        if tf.i3d is None:
                            continue
                        probs = _predict_of_i3d(model, tf.of_300, tf.i3d,
                                                batch_size=args.batch_size, device=device)
                    else:
                        probs = _predict_of(model, tf.of_300,
                                            batch_size=args.batch_size, device=device)

                    metrics = _compute_metrics(probs, tf.true_labels)
                    cell_base = {
                        "train_group": group_name,
                        "test_set":    test_cfg.name,
                        "model":       model_type,
                        "model_type":  model_type,
                        "loss":        "ce",
                        "arch_key":    arch_key,
                    }
                    per_clip_rows.extend(
                        per_clip_prediction_rows(
                            probs, tf.sample_ids, tf.true_labels, base=cell_base
                        )
                    )
                    row: dict[str, Any] = {
                        **cell_base,
                        "target_frames": train_cfg.target_frames,
                        "n_labeled":     metrics["n_labeled"] if metrics else 0,
                    }
                    for col in METRIC_COLS:
                        row[col] = metrics.get(col, float("nan")) if metrics else float("nan")
                    all_rows.append(row)

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        for tf in test_features.values():
            del tf
        del of_mean, of_std
        gc.collect()

    if not all_rows:
        print("\nNo results generated.")
        return

    new_df = pd.DataFrame(all_rows)
    out_csv = out / "hybrid_matrix.csv"

    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        # Remove rows for groups+model_types we just re-evaluated
        evaluated = set(zip(new_df["train_group"], new_df.get("model_type", new_df["model"])))
        if "model_type" in existing.columns:
            mask = existing.apply(lambda r: (r["train_group"], r["model_type"]) in evaluated, axis=1)
        else:
            mask = existing.apply(lambda r: (r["train_group"], r.get("model", "")) in evaluated, axis=1)
        existing = existing[~mask]
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    df.to_csv(out_csv, index=False)

    # Raw per-clip predictions (one row per train x test x arch x clip).
    per_clip_df = pd.DataFrame(per_clip_rows)
    per_clip_csv = out / "hybrid_matrix_predictions.csv"
    if per_clip_csv.exists() and not per_clip_df.empty:
        existing_pc = pd.read_csv(per_clip_csv)
        evaluated_pc = set(zip(per_clip_df["train_group"], per_clip_df["model_type"]))
        if "model_type" in existing_pc.columns:
            mask_pc = existing_pc.apply(
                lambda r: (r["train_group"], r["model_type"]) in evaluated_pc, axis=1
            )
            existing_pc = existing_pc[~mask_pc]
        per_clip_df = pd.concat([existing_pc, per_clip_df], ignore_index=True)
    per_clip_df.to_csv(per_clip_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Hybrid matrix saved -> {out_csv}  ({len(df)} rows)")
    print(f"Per-clip predictions saved -> {per_clip_csv}  ({len(per_clip_df)} rows)")

    print("\nBest per (train_group, test_set, model_type) by QWK:")
    best = (
        df.sort_values("quadratic_weighted_kappa", ascending=False)
          .groupby(["train_group", "test_set", "model_type"], sort=False)
          .first()
          .reset_index()
    )
    cols = ["train_group", "test_set", "model_type", "arch_key",
            "accuracy", "macro_accuracy", "mae", "quadratic_weighted_kappa"]
    print(best[[c for c in cols if c in best.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
