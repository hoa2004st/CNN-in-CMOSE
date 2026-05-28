"""Run the full model/loss comparison in one Python process."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_parser, run_experiment
from src.output_paths import TRAINING_LOG_DIR, loss_slug, model_output_name


MODELS = [
    "openface_mlp",
    "temporal_cnn",
    "lstm",
    "transformer",
    "i3d_mlp",
    "openface_tcn_i3d_fusion",
]

LOSSES = [
    "cross_entropy",
    "weighted_cross_entropy",
    "ordinal",
]


def build_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all kept models across the configured loss sweep in one process.",
    )
    parser.add_argument("--run_root", default=str(TRAINING_LOG_DIR))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_dir", default="data/CMOSE/features/openface")
    parser.add_argument("--labels_json", default="data/CMOSE/final_data_1.json")
    parser.add_argument("--i3d_feature_dir", default="data/CMOSE/features/i3d")
    parser.add_argument("--target_frames", type=int, default=300)
    parser.add_argument("--fusion_frames", type=int, default=75)
    parser.add_argument(
        "--naive_losses",
        nargs="+",
        default=LOSSES,
        choices=LOSSES,
        help="Losses used for all comparison models.",
    )
    return parser


def main() -> None:
    args = build_runner_parser().parse_args()
    run_root = Path(args.run_root)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    main_parser = build_parser()

    print(f"Run root: {run_root}")
    print(f"Logs: {log_dir}")

    for model in MODELS:
        model_root = run_root / model_output_name(model)
        model_root.mkdir(parents=True, exist_ok=True)

        losses = list(args.naive_losses)
        for loss in losses:
            loss_dir = model_root / loss_slug(loss)
            loss_dir.mkdir(parents=True, exist_ok=True)
            if (loss_dir / "metrics.json").exists():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Skipping {model}/{loss} (metrics.json already exists)")
                continue
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Starting {model} with loss={loss}")
            print(f"[{timestamp}] Output: {loss_dir}")

            base_args = [
                "--model",
                model,
                "--output_dir",
                str(loss_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--patience",
                str(args.patience),
                "--device",
                args.device,
                "--num_workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--feature_dir",
                args.feature_dir,
                "--labels_json",
                args.labels_json,
                "--i3d_feature_dir",
                args.i3d_feature_dir,
                "--target_frames",
                str(args.target_frames),
                "--fusion_frames",
                str(args.fusion_frames),
                "--loss",
                loss,
                *(["--amp"] if args.amp else []),
            ]
            experiment_args = main_parser.parse_args(base_args)
            run_experiment(experiment_args)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Finished {model} with loss={loss}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Generating training-log visualizations for {run_root}")
    from scripts.visualize_models_outputs import main as visualize_main

    import sys

    previous_argv = sys.argv[:]
    try:
        sys.argv = ["visualize_models_outputs.py", "--outputs_dir", str(run_root)]
        visualize_main()
    finally:
        sys.argv = previous_argv

    print("All comparison runs completed.")


if __name__ == "__main__":
    main()
