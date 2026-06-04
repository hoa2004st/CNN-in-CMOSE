"""Run hybrid ablations on CMOSE, DaiSEE and Combined sequentially (single GPU).

For every selected dataset and every selected hybrid model type, runs the full
2^5 per-group architecture sweep (see run_hybrid_ablation.py).

Usage:
    python src/training/run_all_hybrid_ablations.py --amp
    python src/training/run_all_hybrid_ablations.py --amp --dry_run
    python src/training/run_all_hybrid_ablations.py --amp --datasets cmose --model_types openface_temporal_i3d_hybrid
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.run_hybrid_ablation import build_runner_parser
import src.training.run_hybrid_ablation as _ablation_module


DATASET_CONFIGS = [
    {
        "dataset":         "cmose",
        "feature_dir":     "data/CMOSE/features/openface",
        "labels_json":     "data/CMOSE/final_data_1.json",
        "i3d_feature_dir": "data/CMOSE/features/i3d",
        "target_frames":   300,
    },
    {
        "dataset":         "daisee",
        "feature_dir":     "data/DaiSEE/features/openface",
        "labels_json":     "data/DaiSEE/final_data_1.json",
        "i3d_feature_dir": "data/DaiSEE/features/i3d",
        "target_frames":   300,
    },
    {
        "dataset":         "combined",
        "feature_dir":     "data/combined/features/openface",
        "labels_json":     "data/combined/final_data_1.json",
        "i3d_feature_dir": "data/combined/features/i3d",
        "target_frames":   150,
    },
]

MODEL_TYPES = ["openface_temporal_hybrid", "openface_temporal_i3d_hybrid"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run_root", default="outputs/training_log")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aux_weight", type=float, default=0.2)
    p.add_argument("--loss", default="cross_entropy")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--datasets", nargs="+", default=["cmose", "daisee", "combined"],
                   choices=["cmose", "daisee", "combined"],
                   help="Which datasets to run (default: all three).")
    p.add_argument("--model_types", nargs="+", default=MODEL_TYPES,
                   choices=MODEL_TYPES,
                   help="Which hybrid model types to run (default: both).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    runner_parser = build_runner_parser()

    active = [cfg for cfg in DATASET_CONFIGS if cfg["dataset"] in args.datasets]

    for cfg in active:
        for model_type in args.model_types:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{'='*60}")
            print(f"[{ts}] Starting ablation: dataset={cfg['dataset']}  "
                  f"model={model_type}  frames={cfg['target_frames']}")
            print(f"{'='*60}")

            argv = [
                "--run_root",        args.run_root,
                "--dataset",         cfg["dataset"],
                "--model",           model_type,
                "--feature_dir",     cfg["feature_dir"],
                "--labels_json",     cfg["labels_json"],
                "--i3d_feature_dir", cfg["i3d_feature_dir"],
                "--target_frames",   str(cfg["target_frames"]),
                "--epochs",          str(args.epochs),
                "--batch_size",      str(args.batch_size),
                "--lr",              str(args.lr),
                "--patience",        str(args.patience),
                "--device",          args.device,
                "--num_workers",     str(args.num_workers),
                "--seed",            str(args.seed),
                "--aux_weight",      str(args.aux_weight),
                "--loss",            args.loss,
                *(["--amp"]      if args.amp      else []),
                *(["--dry_run"]  if args.dry_run  else []),
            ]
            runner_parser.parse_args(argv)   # validate
            prev_argv = sys.argv[:]
            try:
                sys.argv = [_ablation_module.__file__] + argv
                _ablation_module.main()
            finally:
                sys.argv = prev_argv

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Finished ablation: {cfg['dataset']}  model={model_type}")

    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
