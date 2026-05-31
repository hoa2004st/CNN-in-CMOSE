"""Exhaustive 2^N ablation over per-group architecture assignments for hybrid models.

Each of the N OpenFace semantic groups (gaze, eye, face, head, au) is independently
assigned either "transformer" or "tcn".  With N=5 this produces 2^5 = 32 runs.

Supports both model types:
  --model openface_temporal_hybrid       (OpenFace-only, default)
  --model openface_temporal_i3d_hybrid   (OpenFace groups + I3D stream)

Usage:
    python scripts/run_hybrid_ablation.py --dataset cmose --amp
    python scripts/run_hybrid_ablation.py --dataset combined --model openface_temporal_i3d_hybrid --amp
    python scripts/run_hybrid_ablation.py --dataset combined --dry_run
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_parser, run_experiment
from src.feature_extraction.extract_openface import OPENFACE_GROUP_ORDER
from src.output_paths import model_output_name


ARCHS = ("transformer", "tcn")
ARCH_SHORT = {"transformer": "T", "tcn": "TCN"}


def arch_key(assignment: dict[str, str]) -> str:
    return "_".join(ARCH_SHORT[assignment[g]] for g in OPENFACE_GROUP_ORDER)


def build_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_root", default="outputs/training_log")
    parser.add_argument("--dataset", default="cmose",
                        help="Dataset label (cmose | daisee | combined). Sets output subfolder.")
    parser.add_argument(
        "--model",
        default="openface_temporal_hybrid",
        choices=["openface_temporal_hybrid", "openface_temporal_i3d_hybrid"],
        help="Which hybrid model to train.",
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aux_weight", type=float, default=0.2)
    parser.add_argument("--loss", default="cross_entropy",
                        choices=["cross_entropy", "weighted_cross_entropy", "focal", "ordinal"])
    parser.add_argument("--feature_dir", default="data/CMOSE/features/openface")
    parser.add_argument("--labels_json", default="data/CMOSE/final_data_1.json")
    parser.add_argument("--i3d_feature_dir", default="data/CMOSE/features/i3d")
    parser.add_argument("--target_frames", type=int, default=300,
                        help="OpenFace frame count. Use 150 for combined dataset.")
    parser.add_argument("--fusion_frames", type=int, default=75)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_runner_parser().parse_args()
    model_folder = model_output_name(args.model)
    run_root = Path(args.run_root) / args.dataset / model_folder
    run_root.mkdir(parents=True, exist_ok=True)

    main_parser = build_parser()
    all_assignments = [
        dict(zip(OPENFACE_GROUP_ORDER, combo))
        for combo in itertools.product(ARCHS, repeat=len(OPENFACE_GROUP_ORDER))
    ]
    total = len(all_assignments)

    print(f"Model    : {args.model}  ({model_folder}/)")
    print(f"Dataset  : {args.dataset}")
    print(f"Run root : {run_root}")
    print(f"Groups   : {OPENFACE_GROUP_ORDER}")
    print(f"Variants : {total} (2^{len(OPENFACE_GROUP_ORDER)})")
    print(f"Loss     : {args.loss}  Frames : {args.target_frames}")
    print()

    completed = skipped = 0
    for idx, assignment in enumerate(all_assignments, start=1):
        key = arch_key(assignment)
        out_dir = run_root / key
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "metrics.json").exists():
            skipped += 1
            print(f"[{idx:02d}/{total}] SKIP {key}")
            continue

        arch_json = json.dumps(assignment)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{idx:02d}/{total}] {ts}  START  {key}  {assignment}")

        if args.dry_run:
            print(f"          [dry_run] -> {out_dir}")
            continue

        base_args = [
            "--model", args.model,
            "--group_architectures", arch_json,
            "--aux_weight", str(args.aux_weight),
            "--output_dir", str(out_dir),
            "--loss", args.loss,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--device", args.device,
            "--num_workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--feature_dir", args.feature_dir,
            "--labels_json", args.labels_json,
            "--i3d_feature_dir", args.i3d_feature_dir,
            "--target_frames", str(args.target_frames),
            "--fusion_frames", str(args.fusion_frames),
            "--dataset", args.dataset,
            *(["--amp"] if args.amp else []),
        ]
        run_experiment(main_parser.parse_args(base_args))

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{idx:02d}/{total}] {ts}  DONE   {key}")
        completed += 1

    print(f"\nAblation complete: {completed} trained, {skipped} skipped, "
          f"{total - completed - skipped} dry-run.")


if __name__ == "__main__":
    main()
