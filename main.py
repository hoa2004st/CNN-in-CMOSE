"""Main entry point for the CMOSE train + unlabeled comparison pipeline."""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch

from src.paper_repro_data import (
    ID_TO_LABEL,
    describe_selection,
    load_cmose_metadata,
    load_dataset_matrices,
    load_i3d_dataset_matrices,
    materialize_i3d_features_from_json,
    resample_frames,
)
from src.paper_repro_model import build_model
from src.paper_repro_preprocess import (
    fit_feature_normalizer,
    normalize_dataset_per_feature,
)
from src.paper_repro_train import (
    evaluate_predictions,
    predict,
    save_json,
    train_model,
)


logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CMOSE_TRAIN_SPLIT = "train"
CMOSE_UNLABELED_SPLIT_KEY = "test"
I3D_ONLY_MODELS = {"i3d_mlp"}
MULTIMODAL_MODELS = {"openface_tcn_i3d_fusion"}
I3D_ENABLED_MODELS = I3D_ONLY_MODELS | MULTIMODAL_MODELS


def _log_chunk_progress(done: int, total: int) -> None:
    logger.info("Normalization progress: %d/%d samples", done, total)


def _log_step(message: str) -> None:
    logger.info("%s", message)


def split_cmose_records_by_usage(
    records: list,
) -> tuple[list, list]:
    train_records = [record for record in records if record.split == CMOSE_TRAIN_SPLIT]
    unlabeled_records = [record for record in records if record.split == CMOSE_UNLABELED_SPLIT_KEY]
    return train_records, unlabeled_records


def _resample_sample_batch(matrices: np.ndarray, *, target_frames: int) -> np.ndarray:
    if matrices.shape[1] == target_frames:
        return matrices.astype(np.float32, copy=False)
    return np.stack(
        [resample_frames(sample, target_frames=target_frames) for sample in matrices],
        axis=0,
    ).astype(np.float32, copy=False)


def _has_materialized_i3d_features(feature_dir: str | Path) -> bool:
    feature_dir = Path(feature_dir)
    if not feature_dir.exists() or not feature_dir.is_dir():
        return False
    return any(feature_dir.glob("*.npy")) or any(feature_dir.glob("*.npz")) or any(feature_dir.glob("*.pt"))


def resolve_output_dir(
    output_dir_arg: str | None,
    *,
    model_name: str,
    loss_name: str,
    focal_gamma: float,
) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return Path("outputs") / model_name


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the narrowed CMOSE train + unlabeled comparison pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--feature_dir",
        default="data/CMOSE/secondFeature/secondFeature",
        help="Directory containing one OpenFace CSV per person clip.",
    )
    parser.add_argument(
        "--labels_json",
        default="data/CMOSE/final_data_1.json",
        help="Path to the CMOSE label JSON aligned with the feature CSVs.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "openface_mlp",
            "temporal_cnn",
            "lstm",
            "transformer",
            "i3d_mlp",
            "openface_tcn_i3d_fusion",
        ],
        default="temporal_cnn",
        help="Model architecture to train under the CMOSE train + unlabeled split.",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=300,
        help="Fixed frame count per OpenFace sample.",
    )
    parser.add_argument(
        "--i3d_feature_dir",
        default="data/CMOSE/i3d",
        help="Directory containing one precomputed I3D feature file per sample id.",
    )
    parser.add_argument(
        "--fusion_frames",
        type=int,
        default=75,
        help="Shared temporal length used by I3D-based models.",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "weighted_cross_entropy", "focal", "ordinal"],
        default="cross_entropy",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA.")
    parser.add_argument("--output_dir")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    logger.info(
        "Runtime device: requested=%s resolved=%s cuda_available=%s cuda_device_count=%d",
        args.device,
        device.type,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    if device.type == "cuda":
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(device))

    output_dir = resolve_output_dir(
        args.output_dir,
        model_name=args.model,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading metadata from %s", args.labels_json)
    records = load_cmose_metadata(
        args.labels_json,
        args.feature_dir,
        allowed_splits=(CMOSE_TRAIN_SPLIT, CMOSE_UNLABELED_SPLIT_KEY),
    )
    selected_records = records
    train_records, unlabeled_records = split_cmose_records_by_usage(selected_records)

    selection_summary = {
        "mode": "cmose_train_unlabeled_split",
        "before_selection": describe_selection(records),
        "after_selection": describe_selection(selected_records),
        "assumptions": {
            "selection_group": None,
            "fixed_frame_count": args.target_frames,
            "normalization": (
                "per-feature z-score using train-set statistics on I3D features"
                if args.model == "i3d_mlp"
                else (
                    "separate per-feature z-score using train-set statistics for OpenFace and I3D"
                    if args.model == "openface_tcn_i3d_fusion"
                    else "per-feature z-score using train-set statistics on OpenFace features"
                )
            ),
            "smote_position": None,
            "dataset_scope": "original CMOSE train samples plus the dataset's unlabeled/evaluation split",
            "train_eval_split_usage": (
                "fit on the train split; use the CMOSE unlabeled/evaluation split "
                "for early stopping and checkpoint selection"
            ),
            "source_split_keys": {
                "train": CMOSE_TRAIN_SPLIT,
                "unlabeled": CMOSE_UNLABELED_SPLIT_KEY,
            },
            "model": args.model,
            "loss": args.loss,
            "openface_frames": args.target_frames,
            "fusion_frames": args.fusion_frames if args.model in I3D_ENABLED_MODELS else None,
        },
    }
    save_json(selection_summary, output_dir / "selection_summary.json")
    logger.info("Saved selection summary to %s", output_dir / "selection_summary.json")

    logger.info("Loading %d selected samples", len(selected_records))
    logger.info(
        "Using CMOSE train + unlabeled split: %d train / %d unlabeled samples (source keys: %s/%s)",
        len(train_records),
        len(unlabeled_records),
        CMOSE_TRAIN_SPLIT,
        CMOSE_UNLABELED_SPLIT_KEY,
    )
    X_train_raw, y_train, train_sample_ids = load_dataset_matrices(
        train_records,
        target_frames=args.target_frames,
        progress_desc="Loading train samples",
    )
    X_unlabeled_raw, y_unlabeled, unlabeled_sample_ids = load_dataset_matrices(
        unlabeled_records,
        target_frames=args.target_frames,
        progress_desc="Loading unlabeled samples",
    )
    raw_input_features = int(X_train_raw.shape[-1])
    input_features = raw_input_features
    i3d_input_features: int | None = None
    X_train_i3d_raw: np.ndarray | None = None
    X_unlabeled_i3d_raw: np.ndarray | None = None
    i3d_materialization_summary: dict[str, int | str] | None = None
    if args.model in I3D_ENABLED_MODELS:
        if not _has_materialized_i3d_features(args.i3d_feature_dir):
            logger.info(
                "No materialized I3D feature directory detected at %s; extracting from %s",
                args.i3d_feature_dir,
                args.labels_json,
            )
            i3d_materialization_summary = materialize_i3d_features_from_json(
                args.labels_json,
                args.i3d_feature_dir,
            )
            logger.info(
                "Materialized %d I3D feature files to %s",
                i3d_materialization_summary["written_files"],
                args.i3d_feature_dir,
            )
        logger.info("Loading aligned I3D features from %s", args.i3d_feature_dir)
        X_train_i3d_raw = load_i3d_dataset_matrices(
            train_sample_ids,
            feature_dir=args.i3d_feature_dir,
            target_frames=args.fusion_frames,
            progress_desc="Loading train I3D features",
        )
        X_unlabeled_i3d_raw = load_i3d_dataset_matrices(
            unlabeled_sample_ids,
            feature_dir=args.i3d_feature_dir,
            target_frames=args.fusion_frames,
            progress_desc="Loading unlabeled I3D features",
        )
        i3d_input_features = int(X_train_i3d_raw.shape[-1])
    model, model_spec = build_model(
        args.model,
        input_features=input_features,
        i3d_input_features=i3d_input_features,
        num_classes=len(ID_TO_LABEL),
    )

    if model_spec.input_kind == "multimodal_sequence":
        if X_train_i3d_raw is None or X_unlabeled_i3d_raw is None:
            raise RuntimeError("I3D features were not loaded for the multimodal model")
        logger.info(
            "Normalizing OpenFace and I3D features with train-fit per-feature z-score for %s",
            args.model,
        )
        openface_mean, openface_std = fit_feature_normalizer(X_train_raw)
        X_train_openface = normalize_dataset_per_feature(
            X_train_raw,
            mean=openface_mean,
            std=openface_std,
            progress_desc="Normalizing train OpenFace samples",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_train_raw
        gc.collect()
        X_unlabeled_openface = normalize_dataset_per_feature(
            X_unlabeled_raw,
            mean=openface_mean,
            std=openface_std,
            progress_desc="Normalizing unlabeled OpenFace samples",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_unlabeled_raw
        gc.collect()
        X_train_openface = _resample_sample_batch(X_train_openface, target_frames=args.fusion_frames)
        X_unlabeled_openface = _resample_sample_batch(
            X_unlabeled_openface,
            target_frames=args.fusion_frames,
        )

        i3d_mean, i3d_std = fit_feature_normalizer(X_train_i3d_raw)
        X_train_i3d = normalize_dataset_per_feature(
            X_train_i3d_raw,
            mean=i3d_mean,
            std=i3d_std,
            progress_desc="Normalizing train I3D features",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_train_i3d_raw
        gc.collect()
        X_unlabeled_i3d = normalize_dataset_per_feature(
            X_unlabeled_i3d_raw,
            mean=i3d_mean,
            std=i3d_std,
            progress_desc="Normalizing unlabeled I3D features",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_unlabeled_i3d_raw
        gc.collect()

        X_train_input = (X_train_openface.astype(np.float32), X_train_i3d.astype(np.float32))
        X_unlabeled_input = (
            X_unlabeled_openface.astype(np.float32),
            X_unlabeled_i3d.astype(np.float32),
        )
        preprocessing_summary = {
            "model": args.model,
            "sample_count": len(train_sample_ids) + len(unlabeled_sample_ids),
            "train_sample_count": len(train_sample_ids),
            "unlabeled_sample_count": len(unlabeled_sample_ids),
            "openface_feature_count": raw_input_features,
            "i3d_feature_count": i3d_input_features,
            "openface_input_frames": args.target_frames,
            "fusion_frames": args.fusion_frames,
            "openface_shape": list(map(int, X_train_input[0].shape[1:])),
            "i3d_shape": list(map(int, X_train_input[1].shape[1:])),
            "normalization": {
                "openface": "per-feature z-score using train-set statistics",
                "i3d": "per-feature z-score using train-set statistics",
            },
            "i3d_materialization": i3d_materialization_summary,
        }
    elif model_spec.input_kind == "i3d_flat_mlp":
        if X_train_i3d_raw is None or X_unlabeled_i3d_raw is None:
            raise RuntimeError("I3D features were not loaded for the I3D MLP model")
        del X_train_raw
        del X_unlabeled_raw
        gc.collect()
        logger.info(
            "Normalizing I3D features with train-fit per-feature z-score for %s",
            args.model,
        )
        i3d_mean, i3d_std = fit_feature_normalizer(X_train_i3d_raw)
        X_train_i3d = normalize_dataset_per_feature(
            X_train_i3d_raw,
            mean=i3d_mean,
            std=i3d_std,
            progress_desc="Normalizing train I3D features",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_train_i3d_raw
        gc.collect()
        X_unlabeled_i3d = normalize_dataset_per_feature(
            X_unlabeled_i3d_raw,
            mean=i3d_mean,
            std=i3d_std,
            progress_desc="Normalizing unlabeled I3D features",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_unlabeled_i3d_raw
        gc.collect()
        X_train_input = X_train_i3d.astype(np.float32)
        X_unlabeled_input = X_unlabeled_i3d.astype(np.float32)
        preprocessing_summary = {
            "model": args.model,
            "sample_count": len(train_sample_ids) + len(unlabeled_sample_ids),
            "train_sample_count": len(train_sample_ids),
            "unlabeled_sample_count": len(unlabeled_sample_ids),
            "i3d_feature_count": i3d_input_features,
            "fusion_frames": args.fusion_frames,
            "i3d_shape": list(map(int, X_train_input.shape[1:])),
            "normalization": "per-feature z-score using train-set statistics on I3D features",
            "i3d_materialization": i3d_materialization_summary,
        }
    else:
        logger.info(
            "Normalizing OpenFace features with train-fit per-feature z-score for %s",
            args.model,
        )
        feature_mean, feature_std = fit_feature_normalizer(X_train_raw)
        X_train_processed = normalize_dataset_per_feature(
            X_train_raw,
            mean=feature_mean,
            std=feature_std,
            progress_desc="Normalizing train samples",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_train_raw
        gc.collect()
        X_unlabeled_processed = normalize_dataset_per_feature(
            X_unlabeled_raw,
            mean=feature_mean,
            std=feature_std,
            progress_desc="Normalizing unlabeled samples",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_unlabeled_raw
        gc.collect()
        X_train_input = X_train_processed.astype(np.float32)
        X_unlabeled_input = X_unlabeled_processed.astype(np.float32)
        del X_train_processed
        del X_unlabeled_processed
        gc.collect()
        preprocessing_summary = {
            "model": args.model,
            "sample_count": len(train_sample_ids) + len(unlabeled_sample_ids),
            "train_sample_count": len(train_sample_ids),
            "unlabeled_sample_count": len(unlabeled_sample_ids),
            "raw_feature_count": raw_input_features,
            "target_frames": args.target_frames,
            "normalization": "per-feature z-score using train-set statistics on OpenFace features",
        }

    save_json(preprocessing_summary, output_dir / "preprocessing_summary.json")
    logger.info("Saved preprocessing summary to %s", output_dir / "preprocessing_summary.json")
    save_json(
        {
            "before_smote": {
                "train": {ID_TO_LABEL[i]: int((y_train == i).sum()) for i in sorted(ID_TO_LABEL)},
                "unlabeled": {
                    ID_TO_LABEL[i]: int((y_unlabeled == i).sum()) for i in sorted(ID_TO_LABEL)
                },
            },
            "after_smote": None,
        },
        output_dir / "smote_summary.json",
    )
    logger.info("Saved class-count summary to %s", output_dir / "smote_summary.json")
    logger.info("Starting model training for %s", args.model)
    history = train_model(
        model,
        X_train_input,
        y_train,
        X_unlabeled_input,
        y_unlabeled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        checkpoint_path=output_dir / "best_model.pth",
        device=device,
        num_workers=args.num_workers,
        use_amp=args.amp,
        progress_callback=_log_step,
    )
    history["selection_split_name"] = "unlabeled"
    history["selection_split_source_key"] = CMOSE_UNLABELED_SPLIT_KEY
    logger.info(
        "Training finished for %s: best_epoch=%d, stopped_early=%s",
        args.model,
        history["best_epoch"],
        history.get("stopped_early", False),
    )

    logger.info("Starting final prediction on the CMOSE unlabeled split used for selection")
    y_pred = predict(
        model,
        X_unlabeled_input,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        use_amp=args.amp,
        progress_callback=_log_step,
    )
    logger.info("Prediction finished; computing metrics")
    metrics = evaluate_predictions(y_unlabeled, y_pred)
    save_json(
        {
            "config": {
                "model": args.model,
                "protocol": "cmose_train_unlabeled_split",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "patience": args.patience,
                "loss": args.loss,
                "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
                "device": device.type,
                "num_workers": args.num_workers,
                "amp": args.amp,
                "seed": args.seed,
                "i3d_feature_dir": (
                    args.i3d_feature_dir if args.model in I3D_ENABLED_MODELS else None
                ),
                "fusion_frames": args.fusion_frames if args.model in I3D_ENABLED_MODELS else None,
                "selection_split": {
                    "train_key": CMOSE_TRAIN_SPLIT,
                    "unlabeled_key": CMOSE_UNLABELED_SPLIT_KEY,
                    "checkpoint_and_early_stopping": "unlabeled",
                },
            },
            "history": history,
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )
    logger.info("Saved metrics to %s", output_dir / "metrics.json")

    print("\n" + "=" * 60)
    print("CMOSE TRAIN + UNLABELED RESULTS")
    print("=" * 60)
    print(f"  Model         : {args.model}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Macro Acc     : {metrics['macro_accuracy']:.4f}")
    print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted) : {metrics['f1_weighted']:.4f}")
    print(f"  Best epoch    : {history['best_epoch']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
