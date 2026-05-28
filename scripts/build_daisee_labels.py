"""Build CMOSE-compatible labels for the DAiSEE engagement dataset.

DAiSEE ships one 10-second clip per row with four affect labels
(Boredom, Engagement, Confusion, Frustration), each in {0,1,2,3}.
This pipeline only consumes the *engagement* label, mapped onto the same
4-class scheme the CMOSE loader expects:

    DAiSEE engagement 0 -> "Highly Disengage" (id 0)
    DAiSEE engagement 1 -> "Disengage"        (id 1)
    DAiSEE engagement 2 -> "Engage"           (id 2)
    DAiSEE engagement 3 -> "Highly Engage"    (id 3)

Split mapping (DAiSEE -> pipeline source-split keys used by main.py):

    Train      -> "train"    (fit)
    Validation -> "unlabel"  (early stopping / checkpoint selection)
    Test       -> "test"     (final reporting)

The CMOSE OpenFace loader splits each sample id on ``_person``; DAiSEE clips
have a single subject, so every sample id is suffixed with ``_person0`` to keep
``load_cmose_metadata`` working without any code change.

Outputs (schema identical to data/CMOSE):
    <out_json>  : { "<clipstem>_person0": {"split": ..., "label": ..., ...}, ... }
    <out_csv>   : clip_id,label,split   (label as int 0..3)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ENGAGEMENT_ID_TO_NAME = {
    0: "Highly Disengage",
    1: "Disengage",
    2: "Engage",
    3: "Highly Engage",
}

SPLIT_FROM_FILENAME = {
    "train": "train",
    "validation": "unlabel",
    "valid": "unlabel",
    "val": "unlabel",
    "test": "test",
}

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder of the extracted DAiSEE Kaggle dataset.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/DaiSEE/final_data_1.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/DaiSEE/labels.csv"),
    )
    parser.add_argument(
        "--person-suffix",
        default="",
        help="Suffix appended to each clip id (empty = match data/private convention).",
    )
    return parser.parse_args()


def clip_stem(raw_clip_id: str) -> str:
    """Strip a trailing video extension from a DAiSEE ClipID cell."""
    name = raw_clip_id.strip()
    p = Path(name)
    if p.suffix.lower() in VIDEO_SUFFIXES:
        return p.stem
    return name


def find_label_csvs(dataset_root: Path) -> dict[str, Path]:
    """Locate per-split label CSVs by filename, case-insensitively."""
    found: dict[str, Path] = {}
    for csv_path in dataset_root.rglob("*.csv"):
        lower = csv_path.name.lower()
        if "label" not in lower:
            continue
        for token, split in SPLIT_FROM_FILENAME.items():
            if token in lower and split not in found:
                found[split] = csv_path
    return found


def normalize_header(fieldnames: list[str]) -> dict[str, str]:
    """Map stripped/lowercased header -> original header (handles trailing spaces)."""
    return {name.strip().lower(): name for name in fieldnames}


def read_split_rows(csv_path: Path, split: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty label file: {csv_path}")
        header = normalize_header(reader.fieldnames)
        if "clipid" not in header:
            raise ValueError(f"No ClipID column in {csv_path} (got {reader.fieldnames})")
        if "engagement" not in header:
            raise ValueError(f"No Engagement column in {csv_path} (got {reader.fieldnames})")
        clip_key, eng_key = header["clipid"], header["engagement"]
        for record in reader:
            raw_clip = record.get(clip_key)
            raw_eng = record.get(eng_key)
            if raw_clip is None or raw_eng is None or not str(raw_clip).strip():
                continue
            engagement = int(str(raw_eng).strip())
            if engagement not in ENGAGEMENT_ID_TO_NAME:
                raise ValueError(f"Unexpected engagement value {engagement!r} in {csv_path}")
            rows.append((clip_stem(str(raw_clip)), engagement))
    return rows


def main() -> None:
    args = parse_args()
    label_csvs = find_label_csvs(args.dataset_root)
    missing = {"train", "unlabel", "test"} - set(label_csvs)
    if missing:
        raise SystemExit(
            f"Could not find label CSVs for splits {sorted(missing)} under "
            f"{args.dataset_root}. Found: { {k: str(v) for k, v in label_csvs.items()} }"
        )

    final_data: dict[str, dict] = {}
    csv_rows: list[tuple[str, int, str]] = []
    per_split_counts: dict[str, int] = {}

    for split, csv_path in sorted(label_csvs.items()):
        rows = read_split_rows(csv_path, split)
        per_split_counts[split] = len(rows)
        for stem, engagement in rows:
            sample_id = f"{stem}{args.person_suffix}"
            label_name = ENGAGEMENT_ID_TO_NAME[engagement]
            final_data[sample_id] = {
                "clip_id": sample_id,
                "base_clip": stem,
                "split": split,
                "label": label_name,
                "label_id": engagement,
            }
            csv_rows.append((sample_id, engagement, split))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(final_data, indent=2, sort_keys=True), encoding="utf-8"
    )

    csv_rows.sort(key=lambda item: item[0])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "label", "split"])
        writer.writerows(csv_rows)

    print(f"Wrote {len(final_data)} samples -> {args.out_json}")
    print(f"Wrote {len(csv_rows)} rows    -> {args.out_csv}")
    for split in ("train", "unlabel", "test"):
        print(f"  {split:8s}: {per_split_counts.get(split, 0)} clips")


if __name__ == "__main__":
    main()
