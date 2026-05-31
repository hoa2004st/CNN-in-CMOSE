"""Class distribution analysis for the private dataset.

Reads data/private/private_manual_labels.csv and produces:
  - outputs/dataset_analysis/private/class_distribution.csv
  - outputs/dataset_analysis/private/class_distribution_barchart.png
  - outputs/dataset_analysis/private/dataset_summary.json

Split distribution is skipped because the private dataset is test-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import CLASS_COLORS, CLASS_LABELS
from src.output_paths import MANUAL_LABELS_CSV

OUT_DIR = REPO_ROOT / "outputs/dataset_analysis/private"
CLASS_ORDER = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]
LABEL_ID_MAP = {"Highly Disengage": 0, "Disengage": 1, "Engage": 2, "Highly Engage": 3}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_csv = REPO_ROOT / MANUAL_LABELS_CSV
    if not labels_csv.exists():
        labels_csv = REPO_ROOT / "data/private/private_manual_labels.csv"
    if not labels_csv.exists():
        print(f"Labels file not found: {labels_csv}")
        return

    df = pd.read_csv(labels_csv)
    # Filter out rows with notes containing "Delete"
    if "notes" in df.columns:
        df = df[~df["notes"].fillna("").str.contains("Delete", case=False, na=False)]

    total = len(df)
    print(f"Private dataset: {total} labeled clips")

    # --- Class distribution ---
    counts = df["manual_label"].value_counts()
    dist_rows = []
    for cls in CLASS_ORDER:
        n = int(counts.get(cls, 0))
        dist_rows.append({
            "class": cls,
            "label_id": LABEL_ID_MAP[cls],
            "count": n,
            "proportion": round(n / total, 6) if total else 0.0,
        })

    dist_df = pd.DataFrame(dist_rows)
    dist_csv = OUT_DIR / "class_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)
    print(f"  Saved class distribution -> {dist_csv}")
    for row in dist_rows:
        print(f"    {row['class']:<20}  {row['count']:>4}  ({row['proportion']*100:.1f}%)")

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(CLASS_ORDER))
    bar_colors = [CLASS_COLORS.get(cls, "#888888") for cls in CLASS_ORDER]
    cnts = [r["count"] for r in dist_rows]
    bars = ax.bar(xs, cnts, color=bar_colors, width=0.6, edgecolor="white", linewidth=0.8)

    # Value labels above bars
    for bar, cnt in zip(bars, cnts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cnts) * 0.02,
            str(cnt),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_ORDER], fontsize=10)
    ax.set_ylabel("Number of clips", fontsize=11)
    ax.set_title(f"Private Dataset — Class Distribution (n={total})", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(cnts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    chart_path = OUT_DIR / "class_distribution_barchart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar chart -> {chart_path}")

    # --- Summary JSON ---
    summary = {
        "dataset": "private",
        "total_clips": total,
        "usage": "test_only",
        "class_distribution": {
            r["class"]: {"count": r["count"], "proportion": r["proportion"]}
            for r in dist_rows
        },
        "label_source": str(labels_csv),
    }
    summary_path = OUT_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Saved summary JSON -> {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
