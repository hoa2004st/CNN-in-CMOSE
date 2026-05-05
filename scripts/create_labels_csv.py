"""Create data/CMOSE/labels.csv from data/CMOSE/final_data_1.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABEL_TO_ID = {
    "Highly Disengage": 0,
    "Disengage": 1,
    "Engage": 2,
    "Highly Engage": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate labels.csv with columns: clip_id,label,split."
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        default=Path("data/CMOSE/final_data_1.json"),
        help="Path to CMOSE final_data_1.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/CMOSE/labels.csv"),
        help="Output labels.csv path",
    )
    return parser.parse_args()


def label_to_id(raw_label: object) -> int:
    if isinstance(raw_label, int):
        if raw_label in (0, 1, 2, 3):
            return raw_label
        raise ValueError(f"Invalid numeric label: {raw_label}")
    if not isinstance(raw_label, str):
        raise ValueError(f"Unsupported label type: {type(raw_label).__name__}")
    if raw_label not in LABEL_TO_ID:
        raise ValueError(f"Unknown label name: {raw_label!r}")
    return LABEL_TO_ID[raw_label]


def main() -> None:
    args = parse_args()
    raw = json.loads(args.source_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected top-level JSON object keyed by clip_id")

    rows: list[tuple[str, int, str]] = []
    for clip_id, meta in raw.items():
        if not isinstance(meta, dict):
            continue
        split = meta.get("split")
        label = meta.get("label")
        if not isinstance(split, str):
            continue
        label_id = label_to_id(label)
        rows.append((str(clip_id), label_id, split))

    rows.sort(key=lambda item: item[0])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "label", "split"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
