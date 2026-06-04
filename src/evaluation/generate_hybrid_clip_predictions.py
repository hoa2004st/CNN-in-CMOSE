"""Generate per-clip predictions for the best hybrid checkpoints and merge them
into the wide per-clip prediction CSV consumed by the thesis clip browser.

The naive pipeline (``src.analysis.prediction_generator``) writes
``outputs/model_assessment/naive/predictions_by_clip.csv`` with one
``predicted_label__<run>`` block per naive model/loss run. The hybrid evaluation
(``src/evaluation/evaluate_hybrid_full_matrix.py``) only stores aggregate metrics, so
hybrid models never reach the browser. This script fills that gap:

  1. Read ``hybrid_matrix.csv`` and, for each (train_group, model_type), pick the
     best arch_key by quadratic weighted kappa on the chosen test set.
  2. Re-run inference for those checkpoints on the test set, capturing per-clip
     probabilities (the matrix throws these away).
  3. Build wide ``<column>__<run_key>`` blocks matching the naive schema and
     left-join them onto the naive per-clip CSV.
  4. Write the combined CSV (default:
     ``outputs/model_assessment/comparison/predictions_by_clip_all_groups.csv``).

Run keys are named ``<model_type>/<train_group>`` (e.g.
``openface_temporal_hybrid/cmose``) so the browser's ``_model_group`` buckets
them into the ``OF-Hybrid`` / ``OF+I3D-Hybrid`` groups automatically.

Runs whose feature data is missing locally are skipped with a warning rather
than aborting, so the combined CSV always contains whatever is computable.

Usage:
    python src/evaluation/generate_hybrid_clip_predictions.py
    python src/evaluation/generate_hybrid_clip_predictions.py --test_set cmose_test
    python src/evaluation/generate_hybrid_clip_predictions.py --train_groups cmose
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evaluate_full_matrix import (
    TRAIN_CONFIGS,
    TEST_CONFIGS,
    TestFeatures,
    _fit_i3d_norm,
    _fit_openface_norm,
    _load_private_test,
    _load_test_from_json,
    _predict_probs,
    _resolve_device,
)
from src.feature_extraction.extract_openface import (
    ID_TO_LABEL,
    build_group_column_indices,
    get_openface_feature_columns,
    load_cmose_metadata,
)
from src.models.models import (
    build_openface_temporal_hybrid,
    build_openface_temporal_i3d_hybrid,
)
from src.output_paths import (
    HYBRID_ASSESSMENT_DIR,
    MODEL_COMPARISON_ASSESSMENT_DIR,
    NAIVE_ASSESSMENT_DIR,
    model_output_name,
)

HYBRID_MODEL_TYPES = ["openface_temporal_hybrid", "openface_temporal_i3d_hybrid"]
N_CLASSES = 4
# Value columns produced per run, matching the naive predictions_by_clip schema.
PROB_COLUMNS = [f"prob_{cid}_{ID_TO_LABEL[cid]}" for cid in range(N_CLASSES)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--hybrid_matrix",
        default=str(HYBRID_ASSESSMENT_DIR / "hybrid_matrix.csv"),
        help="Aggregate hybrid metrics CSV used to pick the best arch per group.",
    )
    p.add_argument(
        "--naive_csv",
        default=str(NAIVE_ASSESSMENT_DIR / "predictions_by_clip.csv"),
        help="Wide naive per-clip CSV to merge hybrid columns onto.",
    )
    p.add_argument(
        "--output_csv",
        default=str(MODEL_COMPARISON_ASSESSMENT_DIR / "predictions_by_clip_all_groups.csv"),
        help="Combined per-clip CSV to write.",
    )
    p.add_argument(
        "--test_set",
        default="cmose_test",
        choices=["cmose_test", "daisee_test", "private"],
        help="Test set to predict on. Must match the naive CSV's clips.",
    )
    p.add_argument(
        "--train_groups",
        nargs="+",
        default=["cmose", "daisee", "combined"],
        choices=["cmose", "daisee", "combined"],
    )
    p.add_argument(
        "--model_types",
        nargs="+",
        default=HYBRID_MODEL_TYPES,
        choices=HYBRID_MODEL_TYPES,
    )
    p.add_argument(
        "--selection_metric",
        default="quadratic_weighted_kappa",
        help="Column in hybrid_matrix used to pick the best arch (higher is better).",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--fusion_frames", type=int, default=75)
    return p


def _select_best_archs(
    matrix: pd.DataFrame,
    *,
    test_set: str,
    train_groups: list[str],
    model_types: list[str],
    metric: str,
) -> dict[tuple[str, str], str]:
    """Return {(train_group, model_type): best arch_key} on the given test set."""
    sub = matrix[matrix["test_set"] == test_set].copy()
    if sub.empty:
        return {}
    type_col = "model_type" if "model_type" in sub.columns else "model"
    best: dict[tuple[str, str], str] = {}
    ordered = sub.sort_values(metric, ascending=False)
    for (group, mtype), grp in ordered.groupby(["train_group", type_col], sort=False):
        if group in train_groups and mtype in model_types:
            best[(group, mtype)] = str(grp.iloc[0]["arch_key"])
    return best


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


def _load_test_features(
    test_cfg, *, of_mean, of_std, i3d_mean, i3d_std, target_frames, fusion_frames
) -> TestFeatures:
    if test_cfg.accepted_csv is not None:
        return _load_private_test(
            test_cfg, of_mean=of_mean, of_std=of_std, i3d_mean=i3d_mean,
            i3d_std=i3d_std, target_frames=target_frames, fusion_frames=fusion_frames,
        )
    return _load_test_from_json(
        test_cfg, of_mean=of_mean, of_std=of_std, i3d_mean=i3d_mean,
        i3d_std=i3d_std, target_frames=target_frames, fusion_frames=fusion_frames,
    )


def _wide_block(sample_ids: list[str], probs: np.ndarray, run_key: str) -> pd.DataFrame:
    """Build a per-clip DataFrame (indexed by clip_id) with suffixed columns."""
    predicted_id = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    safe = np.clip(probs, 1e-12, 1.0)
    entropy = -(probs * np.log(safe)).sum(axis=1) / math.log(N_CLASSES)

    data: dict[str, Any] = {
        "predicted_id": predicted_id,
        "predicted_label": [ID_TO_LABEL[int(c)] for c in predicted_id],
        "confidence": confidence,
        "margin": margin,
        "normalized_entropy": entropy,
    }
    for cid, col in enumerate(PROB_COLUMNS):
        data[col] = probs[:, cid]

    frame = pd.DataFrame(data, index=pd.Index(sample_ids, name="clip_id"))
    frame.columns = [f"{col}__{run_key}" for col in frame.columns]
    return frame


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = _resolve_device(args.device)
    print(f"Device: {device}")

    matrix_path = Path(args.hybrid_matrix)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Hybrid matrix not found: {matrix_path}")
    matrix = pd.read_csv(matrix_path)

    best = _select_best_archs(
        matrix,
        test_set=args.test_set,
        train_groups=args.train_groups,
        model_types=args.model_types,
        metric=args.selection_metric,
    )
    if not best:
        print(f"No hybrid rows for test_set={args.test_set} in {matrix_path}.")
        return

    print(f"Selected best archs (by {args.selection_metric}) on {args.test_set}:")
    for (group, mtype), arch in best.items():
        print(f"  {mtype}/{group}: {arch}")

    train_cfg_map = {cfg.name: cfg for cfg in TRAIN_CONFIGS}
    test_cfg = next(tc for tc in TEST_CONFIGS if tc.name == args.test_set)

    wide_blocks: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    # Group selections by train_group so norms/features load once per group.
    groups_in_order = [g for g in args.train_groups if any(k[0] == g for k in best)]
    for group_name in groups_in_order:
        group_runs = {mt: arch for (g, mt), arch in best.items() if g == group_name}
        train_cfg = train_cfg_map[group_name]
        needs_i3d = "openface_temporal_i3d_hybrid" in group_runs

        print(f"\n{'='*60}\nTrain group: {group_name}  frames={train_cfg.target_frames}")
        try:
            train_records = load_cmose_metadata(
                train_cfg.labels_json, train_cfg.openface_dir, allowed_splits=("train",)
            )
            if not train_records:
                raise RuntimeError("no train records")
            of_mean, of_std = _fit_openface_norm(
                train_records, target_frames=train_cfg.target_frames
            )
            feature_columns = get_openface_feature_columns(train_records[0].csv_path)
            group_indices = build_group_column_indices(feature_columns)
        except Exception as exc:  # missing/unreadable OpenFace train features
            for mt in group_runs:
                skipped.append((group_name, mt))
            print(f"  [skip group] cannot fit OpenFace norm for {group_name}: {exc}")
            continue

        if needs_i3d:
            try:
                train_ids = [r.sample_id for r in train_records]
                i3d_mean, i3d_std = _fit_i3d_norm(
                    train_ids, feature_dir=train_cfg.i3d_dir, target_frames=args.fusion_frames
                )
            except Exception as exc:
                skipped.append((group_name, "openface_temporal_i3d_hybrid"))
                group_runs.pop("openface_temporal_i3d_hybrid", None)
                needs_i3d = False
                i3d_mean = np.zeros(1024, dtype=np.float32)
                i3d_std = np.ones(1024, dtype=np.float32)
                print(f"  [skip] cannot fit I3D norm for {group_name}: {exc}")
        else:
            i3d_mean = np.zeros(1024, dtype=np.float32)
            i3d_std = np.ones(1024, dtype=np.float32)

        if not group_runs:
            continue

        try:
            tf = _load_test_features(
                test_cfg, of_mean=of_mean, of_std=of_std, i3d_mean=i3d_mean,
                i3d_std=i3d_std, target_frames=train_cfg.target_frames,
                fusion_frames=args.fusion_frames,
            )
        except Exception as exc:
            for mt in group_runs:
                skipped.append((group_name, mt))
            print(f"  [skip group] cannot load {args.test_set} features: {exc}")
            continue

        if not tf.sample_ids or tf.of_300 is None:
            for mt in group_runs:
                skipped.append((group_name, mt))
            print(f"  [skip group] no test samples for {args.test_set}")
            continue

        for model_type, arch_key in group_runs.items():
            run_key = f"{model_type}/{group_name}"
            ckpt_dir = (
                REPO_ROOT / "outputs/training_log" / group_name
                / model_output_name(model_type) / arch_key
            )
            ckpt = ckpt_dir / "best_model.pth"
            arch = _load_arch(ckpt_dir)
            if not ckpt.exists() or arch is None:
                skipped.append((group_name, model_type))
                print(f"  [skip] missing checkpoint/arch for {run_key}: {ckpt_dir}")
                continue

            is_i3d = model_type == "openface_temporal_i3d_hybrid"
            if is_i3d and tf.i3d is None:
                skipped.append((group_name, model_type))
                print(f"  [skip] {run_key}: no I3D test features")
                continue

            if is_i3d:
                model, _ = build_openface_temporal_i3d_hybrid(
                    group_indices=group_indices, group_architectures=arch, num_classes=N_CLASSES
                )
            else:
                model, _ = build_openface_temporal_hybrid(
                    group_indices=group_indices, group_architectures=arch, num_classes=N_CLASSES
                )
            state = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(state)

            features = (tf.of_300, tf.i3d) if is_i3d else tf.of_300
            probs = _predict_probs(model, features, batch_size=args.batch_size, device=device)
            wide_blocks.append(_wide_block(tf.sample_ids, probs, run_key))
            print(f"  [ok] {run_key}: {len(tf.sample_ids)} clips ({arch_key})")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del tf
        gc.collect()

    # Merge onto the naive per-clip CSV.
    naive_path = Path(args.naive_csv)
    if not naive_path.exists():
        raise FileNotFoundError(f"Naive per-clip CSV not found: {naive_path}")
    base = pd.read_csv(naive_path, encoding="utf-8-sig")

    if wide_blocks:
        combined = base.set_index("clip_id")
        for block in wide_blocks:
            combined = combined.join(block, how="left")
        combined = combined.reset_index()
    else:
        combined = base

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Combined per-clip CSV -> {out_path}  ({len(combined)} clips)")
    print(f"Hybrid runs added: {len(wide_blocks)}")
    if skipped:
        print("Skipped runs (missing data/checkpoints):")
        for group, mtype in skipped:
            print(f"  - {mtype}/{group}")
        print(
            "Restore the relevant feature directories "
            "(e.g. data/CMOSE/features/i3d, data/DaiSEE/features/*, "
            "data/combined/features/openface) and re-run to fill these in."
        )


if __name__ == "__main__":
    main()
