"""Convert MMAction clip-level I3D outputs into CMOSE-compatible .npy files.

MMAction's ``clip_feature_extraction.py`` writes one pickle per input video,
mirroring the input path under the output root (e.g.
``<raw_root>/.../<clip>.avi.pkl``). CMOSE expects one 1024-dim ``.npy`` per
sample id under ``data/DaiSEE/features/i3d/<clipstem>_person0.npy``.

This mirrors the converter used for the private dataset: a 2048-dim ResNet
feature is collapsed to 1024 by averaging adjacent pairs.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".webm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Root dir MMAction wrote the per-clip .pkl features into.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/DaiSEE/features/i3d"),
    )
    parser.add_argument("--person-suffix", default="_person0")
    return parser.parse_args()


def sample_id_from_pkl(pkl_path: Path, person_suffix: str) -> str:
    """`<clip>.avi.pkl` -> `<clip>_person0`."""
    name = pkl_path.name[: -len(".pkl")] if pkl_path.name.endswith(".pkl") else pkl_path.name
    stem = Path(name)
    if stem.suffix.lower() in VIDEO_SUFFIXES:
        name = stem.stem
    return f"{name}{person_suffix}"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped: list[str] = []
    for pkl_path in sorted(args.raw_root.rglob("*.pkl")):
        with pkl_path.open("rb") as f:
            obj = pickle.load(f)
        vec = np.asarray(obj, dtype=np.float32).reshape(-1)
        if vec.size == 2048:
            vec = vec.reshape(1024, 2).mean(axis=1).astype(np.float32)
        elif vec.size != 1024:
            skipped.append(f"{pkl_path} (dim={vec.size})")
            continue
        sample_id = sample_id_from_pkl(pkl_path, args.person_suffix)
        np.save(args.out_dir / f"{sample_id}.npy", vec)
        written += 1

    print(f"Wrote {written} I3D .npy files -> {args.out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} files with unexpected dims:")
        for item in skipped[:20]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
