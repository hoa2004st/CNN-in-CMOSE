"""Build a combined CMOSE + DaiSEE dataset for training.

Creates:
  data/combined/features/openface/  -- NTFS hardlinks, IDs prefixed cmose__/daisee__
  data/combined/features/i3d/       -- NTFS hardlinks, IDs prefixed cmose__/daisee__
  data/combined/final_data_1.json   -- merged labels JSON (all splits included)

All three splits (train / unlabel / test) are included so the combined
training run can use its own internal test split, and so the comprehensive
evaluator can load combined test samples too.

Re-running is safe: existing files are skipped.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_extraction.extract_openface import load_cmose_metadata

SOURCES: dict[str, dict] = {
    "cmose": {
        "labels_json": REPO_ROOT / "data/CMOSE/final_data_1.json",
        "openface_dir": REPO_ROOT / "data/CMOSE/secondFeature",
        "i3d_dir":      REPO_ROOT / "data/CMOSE/features/i3d",
    },
    "daisee": {
        "labels_json": REPO_ROOT / "data/DaiSEE/final_data_1.json",
        "openface_dir": REPO_ROOT / "data/DaiSEE/features/openface",
        "i3d_dir":      REPO_ROOT / "data/DaiSEE/features/i3d",
    },
}

OUT_DIR     = REPO_ROOT / "data/combined"
OUT_OF_DIR  = OUT_DIR / "features/openface"
OUT_I3D_DIR = OUT_DIR / "features/i3d"
OUT_LABELS  = OUT_DIR / "final_data_1.json"


def _link(src: Path, dst: Path) -> None:
    """Create dst as a hardlink to src. Falls back to copy on cross-device / permission errors."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def main() -> None:
    OUT_OF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_I3D_DIR.mkdir(parents=True, exist_ok=True)

    combined: dict[str, dict] = {}

    for prefix, cfg in SOURCES.items():
        records = load_cmose_metadata(
            cfg["labels_json"],
            cfg["openface_dir"],
            allowed_splits=("train", "unlabel", "test"),
        )
        i3d_missing = 0
        for rec in tqdm(records, desc=f"Linking {prefix}", unit="sample"):
            new_id = f"{prefix}__{rec.sample_id}"

            _link(rec.csv_path, OUT_OF_DIR / f"{new_id}.csv")

            i3d_src: Path | None = None
            for suf in (".npy", ".npz", ".pt"):
                cand = cfg["i3d_dir"] / f"{rec.sample_id}{suf}"
                if cand.exists():
                    i3d_src = cand
                    break
            if i3d_src is not None:
                _link(i3d_src, OUT_I3D_DIR / f"{new_id}{i3d_src.suffix}")
            else:
                i3d_missing += 1

            combined[new_id] = {
                "label": rec.label_name,
                "split": rec.split,
                "source": prefix,
                "source_id": rec.sample_id,
            }

        splits_in_source = Counter(v["split"] for k, v in combined.items() if v["source"] == prefix)
        print(f"  {prefix}: {len(records)} samples | splits: {dict(splits_in_source)} | I3D missing: {i3d_missing}")

    OUT_LABELS.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_splits = Counter(v["split"] for v in combined.values())
    all_sources = Counter(v["source"] for v in combined.values())
    print(f"\nTotal: {len(combined)} samples")
    print(f"  Splits:  {dict(all_splits)}")
    print(f"  Sources: {dict(all_sources)}")
    print(f"  JSON  -> {OUT_LABELS}")
    print(f"  OF    -> {OUT_OF_DIR}")
    print(f"  I3D   -> {OUT_I3D_DIR}")


if __name__ == "__main__":
    main()
