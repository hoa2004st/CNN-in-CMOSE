"""Generate dataset-analysis charts under outputs/dataset_analysis/."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.output_paths import (
    DATASET_CMOSE_DIR,
    DATASET_COMPARISON_DIR,
    DATASET_PRIVATE_DIR,
    DOMAIN_DIFFERENCE_DIR,
    MANUAL_LABELS_CSV,
)
from src.visualization.style import CLASS_LABELS, class_color_list

LABEL_TO_ID = {label: index for index, label in enumerate(CLASS_LABELS)}
ID_TO_LABEL = {index: label for index, label in enumerate(CLASS_LABELS)}
STACKED_CLASS_LABELS_BOTTOM_TO_TOP = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]
BAR_WIDTH = 0.52


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _label_id(raw_label: object) -> int | None:
    if isinstance(raw_label, int) and raw_label in ID_TO_LABEL:
        return raw_label
    if isinstance(raw_label, str):
        if raw_label.strip().isdigit():
            value = int(raw_label)
            return value if value in ID_TO_LABEL else None
        return LABEL_TO_ID.get(raw_label)
    return None


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _count_rows(counts: Counter[int], *, domain: str) -> list[dict[str, object]]:
    total = sum(counts.values())
    return [
        {
            "domain": domain,
            "class_id": class_id,
            "class_label": ID_TO_LABEL[class_id],
            "count": int(counts.get(class_id, 0)),
            "proportion": float(counts.get(class_id, 0) / total) if total else 0.0,
        }
        for class_id in ID_TO_LABEL
    ]


def _plot_class_distribution(rows: list[dict[str, object]], title: str, path: Path) -> None:
    labels = [str(row["class_label"]) for row in rows]
    counts = [int(row["count"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160)
    ax.bar(labels, counts, color=class_color_list(labels), width=BAR_WIDTH)
    ax.set_title(title)
    ax.set_ylabel("Clips")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_split_distribution(split_counts: Counter[str], path: Path) -> None:
    splits = ["train", "unlabel", "test"]
    labels = [split for split in splits if split in split_counts]
    labels.extend(split for split in sorted(split_counts) if split not in labels)
    counts = [int(split_counts[split]) for split in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.bar(labels, counts, color=["#0072B2", "#E69F00", "#009E73", "#8C564B"][: len(labels)], width=BAR_WIDTH)
    ax.set_title("CMOSE Data Split")
    ax.set_ylabel("Clips")
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_stacked_distribution(cmose_rows: list[dict[str, object]], private_rows: list[dict[str, object]], path: Path) -> None:
    datasets = ["CMOSE", "Private"]
    rows_by_domain = {
        "CMOSE": {int(row["class_id"]): float(row["proportion"]) * 100 for row in cmose_rows},
        "Private": {int(row["class_id"]): float(row["proportion"]) * 100 for row in private_rows},
    }
    x = np.linspace(0.0, 1.0, 300)
    smooth_x = x * x * (3.0 - 2.0 * x)
    bottoms = np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(2.8, 8.8), dpi=160)
    for label in STACKED_CLASS_LABELS_BOTTOM_TO_TOP:
        class_id = LABEL_TO_ID[label]
        cmose_value = rows_by_domain["CMOSE"].get(class_id, 0.0)
        private_value = rows_by_domain["Private"].get(class_id, 0.0)
        values = cmose_value + (private_value - cmose_value) * smooth_x
        tops = bottoms + values
        ax.fill_between(x, bottoms, tops, label=label, color=class_color_list([label])[0], linewidth=0)
        ax.plot(x, tops, color="white", linewidth=0.8, alpha=0.65)
        bottoms = tops

    ax.set_title("Class Distribution Comparison")
    ax.set_ylabel("Percentage")
    ax.set_xticks([0.0, 1.0], datasets)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 100)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_cmose_counts(labels_json: Path) -> tuple[Counter[int], Counter[str]]:
    raw = json.loads(labels_json.read_text(encoding="utf-8"))
    class_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    for meta in raw.values():
        if not isinstance(meta, dict):
            continue
        class_id = _label_id(meta.get("label"))
        split = meta.get("split")
        if class_id is not None:
            class_counts[class_id] += 1
        if isinstance(split, str):
            split_counts[split] += 1
    return class_counts, split_counts


def load_private_counts(manual_labels_csv: Path) -> Counter[int]:
    class_counts: Counter[int] = Counter()
    if not manual_labels_csv.exists():
        return class_counts
    with manual_labels_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            notes = str(row.get("notes", ""))
            if "delete" in notes.lower():
                continue
            class_id = _label_id(row.get("manual_label_id") or row.get("manual_label"))
            if class_id is not None:
                class_counts[class_id] += 1
    return class_counts


def accepted_count(accepted_csv: Path) -> int:
    if not accepted_csv.exists():
        return 0
    with accepted_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for row in csv.DictReader(handle) if str(row.get("is_accepted", "")).strip() == "1")


def run(args: argparse.Namespace) -> None:
    labels_json = _resolve(args.labels_json)
    manual_labels_csv = _resolve(args.manual_labels_csv)
    accepted_csv = _resolve(args.private_accepted_csv)

    cmose_dir = _resolve(args.cmose_output_dir)
    private_dir = _resolve(args.private_output_dir)
    comparison_dir = _resolve(args.comparison_output_dir)
    domain_dir = _resolve(args.domain_difference_dir)
    for directory in (cmose_dir, private_dir, comparison_dir, domain_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cmose_counts, split_counts = load_cmose_counts(labels_json)
    private_counts = load_private_counts(manual_labels_csv)

    cmose_rows = _count_rows(cmose_counts, domain="cmose")
    private_rows = _count_rows(private_counts, domain="private")

    _write_csv(cmose_dir / "class_distribution.csv", cmose_rows, ["domain", "class_id", "class_label", "count", "proportion"])
    _write_csv(private_dir / "class_distribution.csv", private_rows, ["domain", "class_id", "class_label", "count", "proportion"])
    _write_csv(
        comparison_dir / "class_distribution_comparison.csv",
        cmose_rows + private_rows,
        ["domain", "class_id", "class_label", "count", "proportion"],
    )
    split_rows = [{"split": split, "count": int(count)} for split, count in sorted(split_counts.items())]
    _write_csv(cmose_dir / "data_split.csv", split_rows, ["split", "count"])

    delta_rows = []
    private_total = sum(private_counts.values())
    cmose_total = sum(cmose_counts.values())
    for class_id, label in ID_TO_LABEL.items():
        cmose_prop = cmose_counts.get(class_id, 0) / cmose_total if cmose_total else 0.0
        private_prop = private_counts.get(class_id, 0) / private_total if private_total else 0.0
        delta_rows.append(
            {
                "class_id": class_id,
                "class_label": label,
                "private_minus_cmose_proportion": private_prop - cmose_prop,
            }
        )
    _write_csv(domain_dir / "class_distribution_delta.csv", delta_rows, ["class_id", "class_label", "private_minus_cmose_proportion"])

    _plot_split_distribution(split_counts, cmose_dir / "data_split_diagram.png")
    _plot_class_distribution(cmose_rows, "CMOSE Class Distribution", cmose_dir / "class_distribution_barchart.png")
    _plot_class_distribution(private_rows, "Private Class Distribution", private_dir / "class_distribution_barchart.png")
    _plot_stacked_distribution(cmose_rows, private_rows, comparison_dir / "class_distribution_stacked_barchart.png")

    (cmose_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": "cmose",
                "source": str(labels_json),
                "num_samples": int(sum(cmose_counts.values())),
                "splits": dict(split_counts),
                "class_distribution": cmose_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (private_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": "private",
                "accepted_manifest": str(accepted_csv),
                "manual_labels_csv": str(manual_labels_csv),
                "accepted_samples": accepted_count(accepted_csv),
                "labeled_samples": int(sum(private_counts.values())),
                "class_distribution": private_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (domain_dir / "domain_difference_summary.json").write_text(
        json.dumps(
            {
                "basis": "manual private labels vs CMOSE label distribution",
                "class_distribution_delta": delta_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved CMOSE dataset analysis to {cmose_dir}")
    print(f"Saved private dataset analysis to {private_dir}")
    print(f"Saved comparison dataset analysis to {comparison_dir}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-json", default="data/CMOSE/final_data_1.json")
    parser.add_argument("--private-accepted-csv", default="data/private/accepted.csv")
    parser.add_argument("--manual-labels-csv", default=str(MANUAL_LABELS_CSV))
    parser.add_argument("--cmose-output-dir", default=str(DATASET_CMOSE_DIR))
    parser.add_argument("--private-output-dir", default=str(DATASET_PRIVATE_DIR))
    parser.add_argument("--comparison-output-dir", default=str(DATASET_COMPARISON_DIR))
    parser.add_argument("--domain-difference-dir", default=str(DOMAIN_DIFFERENCE_DIR))
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
