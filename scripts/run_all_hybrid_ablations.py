"""Run hybrid ablations on DaiSEE then Combined datasets sequentially (single GPU).

Usage:
    python scripts/run_all_hybrid_ablations.py --amp
    python scripts/run_all_hybrid_ablations.py --amp --dry_run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_hybrid_ablation import build_runner_parser
import scripts.run_hybrid_ablation as _ablation_module


DATASET_CONFIGS = [
    {
        "dataset":       "daisee",
        "feature_dir":   "data/DaiSEE/features/openface",
        "labels_json":   "data/DaiSEE/final_data_1.json",
        "target_frames": 300,
    },
    {
        "dataset":       "combined",
        "feature_dir":   "data/combined/features/openface",
        "labels_json":   "data/combined/final_data_1.json",
        "target_frames": 150,
    },
]


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
    p.add_argument("--datasets", nargs="+", default=["daisee", "combined"],
                   choices=["daisee", "combined"],
                   help="Which datasets to run (default: both).")
    return p


def main() -> None:
    args = build_parser().parse_args()
    runner_parser = build_runner_parser()

    active = [cfg for cfg in DATASET_CONFIGS if cfg["dataset"] in args.datasets]

    for cfg in active:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*60}")
        print(f"[{ts}] Starting ablation: dataset={cfg['dataset']}  frames={cfg['target_frames']}")
        print(f"{'='*60}")

        argv = [
            "--run_root",       args.run_root,
            "--dataset",        cfg["dataset"],
            "--feature_dir",    cfg["feature_dir"],
            "--labels_json",    cfg["labels_json"],
            "--target_frames",  str(cfg["target_frames"]),
            "--epochs",         str(args.epochs),
            "--batch_size",     str(args.batch_size),
            "--lr",             str(args.lr),
            "--patience",       str(args.patience),
            "--device",         args.device,
            "--num_workers",    str(args.num_workers),
            "--seed",           str(args.seed),
            "--aux_weight",     str(args.aux_weight),
            "--loss",           args.loss,
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
        print(f"[{ts}] Finished ablation: {cfg['dataset']}")

    print("\nAll ablations complete.")


if __name__ == "__main__":
    main()
