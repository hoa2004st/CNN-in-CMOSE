"""Main entry point for the paper-faithful CMOSE reproduction pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from src.paper_repro_data import (
    ID_TO_LABEL,
    describe_selection,
    load_cmose_metadata,
    load_dataset_matrices,
    select_paper_style_subset,
)
from src.paper_repro_model import PaperEngagementCNN
from src.paper_repro_preprocess import (
    apply_smote,
    flatten_matrices,
    preprocess_dataset,
    reshape_flattened_samples,
)
from src.paper_repro_train import (
    evaluate_predictions,
    predict,
    save_json,
    split_after_smote,
    train_model,
)


logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's PCA/SVD + CNN pipeline on CMOSE.",
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
        "--method",
        choices=["pca", "svd"],
        default="svd",
        help="Dimensionality reduction branch to run.",
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
    parser.add_argument(
        "--include_unlabel",
        action="store_true",
        help="Also include entries marked as 'unlabel' in final_data_1.json.",
    )
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_splits = ["train", "test"]
    if args.include_unlabel:
        allowed_splits.append("unlabel")

    logger.info("Loading metadata from %s", args.labels_json)
    records = load_cmose_metadata(
        args.labels_json,
        args.feature_dir,
        allowed_splits=allowed_splits,
    )
    selected_records = select_paper_style_subset(records)

    selection_summary = {
        "before_selection": describe_selection(records),
        "after_selection": describe_selection(selected_records),
        "assumptions": {
            "selection_group": "base_video_id",
            "fixed_frame_count": args.target_frames,
            "dimensionality_reduction_fit": "per-sample",
            "normalization": "per-sample min-max to [0,1]",
            "smote_position": "before final 80/20 split",
            "train_eval_split_usage": (
                "single held-out 20% split is reused for early stopping and final evaluation"
            ),
        },
    }
    save_json(selection_summary, output_dir / "selection_summary.json")

    logger.info("Loading %d selected samples", len(selected_records))
    matrices, labels, sample_ids = load_dataset_matrices(
        selected_records,
        target_frames=args.target_frames,
    )

    logger.info("Applying %s reduction to %d components", args.method.upper(), args.n_components)
    processed, explained = preprocess_dataset(
        matrices,
        method=args.method,
        n_components=args.n_components,
    )
    save_json(
        {
            "sample_count": len(sample_ids),
            "mean_explained_variance": float(explained.mean()),
            "min_explained_variance": float(explained.min()),
            "max_explained_variance": float(explained.max()),
        },
        output_dir / "preprocessing_summary.json",
    )

    X_flat = flatten_matrices(processed)
    X_balanced, y_balanced = apply_smote(X_flat, labels, random_state=args.seed)
    save_json(
        {
            "before_smote": {ID_TO_LABEL[i]: int((labels == i).sum()) for i in sorted(ID_TO_LABEL)},
            "after_smote": {
                ID_TO_LABEL[i]: int((y_balanced == i).sum()) for i in sorted(ID_TO_LABEL)
            },
        },
        output_dir / "smote_summary.json",
    )

    X_train, X_test, y_train, y_test = split_after_smote(
        X_balanced,
        y_balanced,
        random_state=args.seed,
    )
    X_train_cnn = reshape_flattened_samples(X_train, side=args.n_components)
    X_test_cnn = reshape_flattened_samples(X_test, side=args.n_components)

    model = PaperEngagementCNN(input_size=args.n_components)
    history = train_model(
        model,
        X_train_cnn,
        y_train,
        X_test_cnn,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=output_dir / "best_model.pth",
    )

    y_pred = predict(model, X_test_cnn, batch_size=args.batch_size)
    metrics = evaluate_predictions(y_test, y_pred)
    save_json(
        {
            "config": {
                "method": args.method,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            },
            "history": history,
            "metrics": metrics,
        },
        output_dir / "metrics.json",
    )

    print("\n" + "=" * 60)
    print("PAPER-STYLE REPRODUCTION RESULTS")
    print("=" * 60)
    print(f"  Method        : {args.method.upper()}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted) : {metrics['f1_weighted']:.4f}")
    print(f"  Best epoch    : {history['best_epoch']}")
    print("\nClassification Report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
