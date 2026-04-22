"""Main entry point for the strict CMOSE comparison pipeline."""

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
)
from src.paper_repro_model import build_model
from src.paper_repro_preprocess import (
    add_channel_dim,
    fit_feature_normalizer,
    normalize_dataset_per_feature,
    preprocess_dataset,
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


def _log_chunk_progress(done: int, total: int) -> None:
    logger.info("Normalization progress: %d/%d samples", done, total)


def _log_step(message: str) -> None:
    logger.info("%s", message)


def _format_loss_suffix(loss_name: str, focal_gamma: float) -> str:
    if loss_name == "cross_entropy":
        return "cross_entropy"
    if loss_name == "weighted_cross_entropy":
        return "weighted_cross_entropy"
    if loss_name == "focal":
        gamma_str = str(focal_gamma).replace(".", "p")
        return f"focal_g{gamma_str}"
    raise ValueError(f"Unknown loss name: {loss_name}")


def resolve_output_dir(
    output_dir_arg: str | None,
    *,
    model_name: str,
    loss_name: str,
    focal_gamma: float,
) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    loss_suffix = _format_loss_suffix(loss_name, focal_gamma)
    return Path("outputs") / f"{model_name}_{loss_suffix}"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the strict CMOSE comparison pipeline on baseline and raw-sequence models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cmose_root",
        default="data/CMOSE",
        help="Root directory of the CMOSE dataset files.",
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
        choices=["paper_cnn", "temporal_cnn", "rectangular_cnn", "lstm", "transformer"],
        default="paper_cnn",
        help="Model architecture to train under the strict CMOSE split.",
    )
    parser.add_argument(
        "--method",
        choices=["pca", "svd"],
        default="svd",
        help="Dimensionality reduction branch used only by paper_cnn.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=300,
        help="Reduced feature dimension per frame from the paper.",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=300,
        help="Fixed frame count per sample before reduction.",
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "weighted_cross_entropy", "focal"],
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
        allowed_splits=("train", "test"),
    )
    selected_records = records
    train_records = [record for record in selected_records if record.split == "train"]
    test_records = [record for record in selected_records if record.split == "test"]

    selection_summary = {
        "mode": "strict_cmose_split",
        "before_selection": describe_selection(records),
        "after_selection": describe_selection(selected_records),
        "assumptions": {
            "selection_group": None,
            "fixed_frame_count": args.target_frames,
            "dimensionality_reduction_fit": (
                "per-sample"
                if args.model == "paper_cnn"
                else None
            ),
            "normalization": (
                "per-sample z-score after per-sample PCA/SVD"
                if args.model == "paper_cnn"
                else "per-feature z-score using train-set statistics"
            ),
            "smote_position": None,
            "dataset_scope": "original labeled CMOSE train/test samples",
            "train_eval_split_usage": "original CMOSE train/test split",
            "model": args.model,
            "loss": args.loss,
        },
    }
    save_json(selection_summary, output_dir / "selection_summary.json")
    logger.info("Saved selection summary to %s", output_dir / "selection_summary.json")

    logger.info("Loading %d selected samples", len(selected_records))
    logger.info(
        "Using original CMOSE split: %d train / %d test samples",
        len(train_records),
        len(test_records),
    )
    X_train_raw, y_train, train_sample_ids = load_dataset_matrices(
        train_records,
        target_frames=args.target_frames,
        progress_desc="Loading train samples",
    )
    X_test_raw, y_test, test_sample_ids = load_dataset_matrices(
        test_records,
        target_frames=args.target_frames,
        progress_desc="Loading test samples",
    )
    input_features = int(X_train_raw.shape[-1])
    model, model_spec = build_model(
        args.model,
        input_size=args.n_components,
        input_features=input_features,
        num_classes=len(ID_TO_LABEL),
    )

    if model_spec.input_kind == "square_matrix":
        logger.info(
            "Applying %s reduction to %d components after split for %s",
            args.method.upper(),
            args.n_components,
            args.model,
        )
        X_train_processed, train_explained = preprocess_dataset(
            X_train_raw,
            method=args.method,
            n_components=args.n_components,
            progress_desc=f"{args.method.upper()} train samples",
        )
        X_test_processed, test_explained = preprocess_dataset(
            X_test_raw,
            method=args.method,
            n_components=args.n_components,
            progress_desc=f"{args.method.upper()} test samples",
        )
        del X_train_raw
        del X_test_raw
        gc.collect()
        X_train_input = add_channel_dim(X_train_processed)
        X_test_input = add_channel_dim(X_test_processed)
        del X_train_processed
        del X_test_processed
        gc.collect()
        preprocessing_summary = {
            "model": args.model,
            "sample_count": len(train_sample_ids) + len(test_sample_ids),
            "train_sample_count": len(train_sample_ids),
            "test_sample_count": len(test_sample_ids),
            "reduction_method": args.method,
            "reduced_feature_count": args.n_components,
            "train_mean_explained_variance": float(train_explained.mean()),
            "train_min_explained_variance": float(train_explained.min()),
            "train_max_explained_variance": float(train_explained.max()),
            "test_mean_explained_variance": float(test_explained.mean()),
            "test_min_explained_variance": float(test_explained.min()),
            "test_max_explained_variance": float(test_explained.max()),
        }
    else:
        logger.info(
            "Normalizing raw frame-feature sequences with train-fit per-feature z-score for %s",
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
        X_test_processed = normalize_dataset_per_feature(
            X_test_raw,
            mean=feature_mean,
            std=feature_std,
            progress_desc="Normalizing test samples",
            chunk_size=32,
            progress_callback=_log_chunk_progress,
        )
        del X_test_raw
        gc.collect()
        if model_spec.input_kind == "frame_feature_map":
            X_train_input = add_channel_dim(X_train_processed)
            X_test_input = add_channel_dim(X_test_processed)
        else:
            X_train_input = X_train_processed.astype(np.float32)
            X_test_input = X_test_processed.astype(np.float32)
        del X_train_processed
        del X_test_processed
        gc.collect()
        preprocessing_summary = {
            "model": args.model,
            "sample_count": len(train_sample_ids) + len(test_sample_ids),
            "train_sample_count": len(train_sample_ids),
            "test_sample_count": len(test_sample_ids),
            "reduction_method": None,
            "reduced_feature_count": None,
            "raw_feature_count": input_features,
            "target_frames": args.target_frames,
            "normalization": "per-feature z-score using train-set statistics",
        }

    save_json(preprocessing_summary, output_dir / "preprocessing_summary.json")
    logger.info("Saved preprocessing summary to %s", output_dir / "preprocessing_summary.json")
    save_json(
        {
            "before_smote": {
                "train": {ID_TO_LABEL[i]: int((y_train == i).sum()) for i in sorted(ID_TO_LABEL)},
                "test": {ID_TO_LABEL[i]: int((y_test == i).sum()) for i in sorted(ID_TO_LABEL)},
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
        X_test_input,
        y_test,
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
    logger.info(
        "Training finished for %s: best_epoch=%d, stopped_early=%s",
        args.model,
        history["best_epoch"],
        history.get("stopped_early", False),
    )

    logger.info("Starting final prediction on the held-out test split")
    y_pred = predict(
        model,
        X_test_input,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        use_amp=args.amp,
        progress_callback=_log_step,
    )
    logger.info("Prediction finished; computing metrics")
    metrics = evaluate_predictions(y_test, y_pred)
    save_json(
        {
            "config": {
                "model": args.model,
                "method": args.method if args.model == "paper_cnn" else None,
                "protocol": "strict_cmose_split",
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
            },
            "history": history,
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )
    logger.info("Saved metrics to %s", output_dir / "metrics.json")

    print("\n" + "=" * 60)
    print("STRICT CMOSE REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"  Model         : {args.model}")
    print(f"  Method        : {args.method.upper() if args.model == 'paper_cnn' else 'N/A'}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted) : {metrics['f1_weighted']:.4f}")
    print(f"  Best epoch    : {history['best_epoch']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
