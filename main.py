"""Main entry point for the CNN-in-CMOSE pipeline.

Usage
-----
    python main.py --cmose_root /path/to/CMOSE

Run ``python main.py --help`` for all options.

Pipeline steps
--------------
1. Load train / val / test OpenFace features from the CMOSE dataset.
2. Fit a StandardScaler + PCA on the training split.
3. Transform all splits and reshape into (n, 1, H, W) tensors.
4. Train a CNN with class-weighted cross-entropy.
5. Evaluate on the test split and print metrics.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

from src.data_loader import load_split, DEFAULT_FEATURE_COLS
from src.preprocess import Preprocessor
from src.model import EngagementCNN
from src.train import train as train_model
from src.evaluate import evaluate, predict


logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the SVD/PCA + CNN pipeline on the CMOSE dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--cmose_root",
        required=True,
        help="Root directory of the CMOSE dataset (must contain labels/ and openface/).",
    )
    p.add_argument(
        "--n_components",
        type=int,
        default=64,
        help="Number of PCA components (must factorise into a rectangle for CNN input).",
    )
    p.add_argument(
        "--use_svd",
        action="store_true",
        help="Use TruncatedSVD instead of PCA.",
    )
    p.add_argument(
        "--aggregation",
        choices=["mean", "std", "mean_std", "max"],
        default="mean_std",
        help="How to aggregate per-frame features into a clip-level vector.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class-weighted loss (not recommended for imbalanced data).",
    )
    p.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save checkpoints, preprocessor, and evaluation artefacts.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    logger.info("Loading CMOSE dataset from %s …", args.cmose_root)
    X_train, y_train, _ = load_split(
        args.cmose_root, "train",
        feature_cols=DEFAULT_FEATURE_COLS,
        aggregation=args.aggregation,
    )
    X_val, y_val, _ = load_split(
        args.cmose_root, "val",
        feature_cols=DEFAULT_FEATURE_COLS,
        aggregation=args.aggregation,
    )
    X_test, y_test, _ = load_split(
        args.cmose_root, "test",
        feature_cols=DEFAULT_FEATURE_COLS,
        aggregation=args.aggregation,
    )

    # ── 2. Preprocess (scale + PCA) ─────────────────────────────────────────
    preprocessor = Preprocessor(
        n_components=args.n_components,
        use_svd=args.use_svd,
        random_state=args.seed,
    )
    X_train_r = preprocessor.fit_transform(X_train)
    X_val_r = preprocessor.transform(X_val)
    X_test_r = preprocessor.transform(X_test)
    preprocessor.save(output_dir / "preprocessor.pkl")

    # ── 3. Reshape for CNN ──────────────────────────────────────────────────
    grid_h, grid_w = _best_grid(args.n_components)
    logger.info("CNN input grid: %d × %d", grid_h, grid_w)

    X_train_cnn = Preprocessor.reshape_for_cnn(X_train_r, grid_h, grid_w)
    X_val_cnn = Preprocessor.reshape_for_cnn(X_val_r, grid_h, grid_w)
    X_test_cnn = Preprocessor.reshape_for_cnn(X_test_r, grid_h, grid_w)

    # ── 4. Train ────────────────────────────────────────────────────────────
    model = EngagementCNN(grid_h=grid_h, grid_w=grid_w)
    logger.info("Model: %s", model)

    history = train_model(
        model,
        X_train_cnn, y_train,
        X_val_cnn, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        patience=args.patience,
        checkpoint_path=output_dir / "best_model.pth",
    )

    # Load the best checkpoint for evaluation
    best_ckpt = output_dir / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location="cpu"))
        logger.info("Loaded best checkpoint from epoch %d.", history["best_epoch"])

    # ── 5. Evaluate ─────────────────────────────────────────────────────────
    logger.info("Evaluating on test set …")
    y_pred = predict(model, X_test_cnn)
    metrics = evaluate(y_true=y_test, y_pred=y_pred, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted) : {metrics['f1_weighted']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _best_grid(n: int) -> tuple[int, int]:
    """Return the most square-like (h, w) factorisation with h <= w."""
    h = int(math.isqrt(n))
    while h >= 1:
        if n % h == 0:
            return h, n // h
        h -= 1
    return 1, n


if __name__ == "__main__":
    main()
