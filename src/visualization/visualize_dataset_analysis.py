"""Generate dataset-analysis charts and tables under outputs/dataset_analysis/."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import (
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_LABEL_SHORT,
    blend_with_white,
)

OUT_DIR = REPO_ROOT / "outputs" / "dataset_analysis"

DATASETS: dict[str, Path] = {
    "CMOSE": REPO_ROOT / "data/CMOSE/final_data_1.json",
    "DAiSEE": REPO_ROOT / "data/DaiSEE/final_data_1.json",
    "Combined": REPO_ROOT / "data/combined/final_data_1.json",
}

# Private set has no final_data json; its distribution is precomputed here.
PRIVATE_SUMMARY = OUT_DIR / "private" / "dataset_summary.json"

CLASS_ORDER = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]
# Legend ordered top-to-bottom to match the visual stacking of the bars
# (top segment first): Highly Engage > Engage > Disengage > Highly Disengage.
LEGEND_ORDER = list(reversed(CLASS_ORDER))
SPLIT_ORDER = ["train", "unlabel", "test"]
SPLIT_DISPLAY = {"train": "Train", "unlabel": "Val", "test": "Test"}

# Colorblind-friendly per dataset
DATASET_COLORS = {"CMOSE": "#0072B2", "DAiSEE": "#D55E00", "Combined": "#009E73"}


def load_entries(json_path: Path) -> list[dict]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return list(raw.values())


def load_private_dist() -> dict[str, dict] | None:
    """Read the precomputed private-set class distribution, if available."""
    if not PRIVATE_SUMMARY.exists():
        return None
    data = json.loads(PRIVATE_SUMMARY.read_text(encoding="utf-8"))
    cd = data.get("class_distribution", {})
    if not all(cls in cd for cls in CLASS_ORDER):
        return None
    return {
        cls: {"count": int(cd[cls]["count"]), "proportion": float(cd[cls]["proportion"])}
        for cls in CLASS_ORDER
    }


def class_dist(entries: list[dict]) -> dict[str, dict]:
    counts: Counter[str] = Counter(e["label"] for e in entries)
    total = sum(counts.values())
    return {
        cls: {"count": int(counts.get(cls, 0)), "proportion": counts.get(cls, 0) / total if total else 0.0}
        for cls in CLASS_ORDER
    }


def split_class_dist(entries: list[dict]) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for split in SPLIT_ORDER:
        split_entries = [e for e in entries if e.get("split") == split]
        if split_entries:
            result[split] = class_dist(split_entries)
    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_stacked_bar_datasets(
    all_dists: dict[str, dict[str, dict]],
    path: Path,
    *,
    use_proportion: bool = True,
) -> None:
    """Stacked bar chart: one bar per dataset, stacked by class."""
    dataset_names = list(all_dists.keys())
    x = np.arange(len(dataset_names))
    bar_width = 0.48

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    bottoms = np.zeros(len(dataset_names))

    for cls in CLASS_ORDER:
        values = np.array([
            all_dists[ds][cls]["proportion"] * 100 if use_proportion else all_dists[ds][cls]["count"]
            for ds in dataset_names
        ])
        bars = ax.bar(x, values, bar_width, bottom=bottoms, label=cls, color=CLASS_COLORS[cls])
        for i, (bar, val) in enumerate(zip(bars, values)):
            mid = bottoms[i] + val / 2
            if use_proportion and val >= 4.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
            elif not use_proportion and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid,
                    f"{int(val):,}",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
        bottoms += values

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=11)
    ax.set_ylabel("Proportion (%)" if use_proportion else "Clip count", fontsize=10)
    ax.set_title("Class Distribution by Dataset", fontsize=12)
    if use_proportion:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[cls]) for cls in LEGEND_ORDER]
    ax.legend(handles, LEGEND_ORDER, loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0.0, fontsize=8, title="Class", title_fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(REPO_ROOT)}")


def plot_stacked_bar_splits(
    dataset_name: str,
    split_dists: dict[str, dict[str, dict]],
    all_entries: list[dict],
    path: Path,
    *,
    use_proportion: bool = True,
) -> None:
    """Stacked bar chart: one bar per split for a single dataset."""
    splits = [s for s in SPLIT_ORDER if s in split_dists]
    split_totals = Counter(e.get("split") for e in all_entries)
    x = np.arange(len(splits))
    bar_width = 0.50

    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=160)
    bottoms = np.zeros(len(splits))

    for cls in CLASS_ORDER:
        values = np.array([
            split_dists[s][cls]["proportion"] * 100 if use_proportion else split_dists[s][cls]["count"]
            for s in splits
        ])
        bars = ax.bar(x, values, bar_width, bottom=bottoms, label=cls, color=CLASS_COLORS[cls])
        for i, (bar, val) in enumerate(zip(bars, values)):
            mid = bottoms[i] + val / 2
            if use_proportion and val >= 4.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid,
                    f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
            elif not use_proportion and val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid,
                    f"{int(val):,}",
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                )
        bottoms += values

    tick_labels = [
        f"{SPLIT_DISPLAY.get(s, s)}\n(n={split_totals[s]:,})" for s in splits
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Proportion (%)" if use_proportion else "Clip count", fontsize=10)
    ax.set_title(f"{dataset_name} — Class Distribution by Split", fontsize=11)
    if use_proportion:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[cls]) for cls in LEGEND_ORDER]
    ax.legend(handles, LEGEND_ORDER, loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0.0, fontsize=8, title="Class", title_fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(REPO_ROOT)}")


def plot_all_splits_combined(
    dataset_split_dists: dict[str, dict[str, dict[str, dict]]],
    dataset_entries: dict[str, list[dict]],
    path: Path,
    *,
    use_proportion: bool = True,
) -> None:
    """One figure with subplots: one column per dataset, stacked by class per split."""
    dataset_names = list(dataset_split_dists.keys())
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(1, n_datasets, figsize=(4.5 * n_datasets, 5.2), dpi=160, sharey=True)
    if n_datasets == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        split_dists = dataset_split_dists[ds_name]
        entries = dataset_entries[ds_name]
        split_totals = Counter(e.get("split") for e in entries)
        splits = [s for s in SPLIT_ORDER if s in split_dists]
        x = np.arange(len(splits))
        bar_width = 0.50
        bottoms = np.zeros(len(splits))

        for cls in CLASS_ORDER:
            values = np.array([
                split_dists[s][cls]["proportion"] * 100 if use_proportion else split_dists[s][cls]["count"]
                for s in splits
            ])
            bars = ax.bar(x, values, bar_width, bottom=bottoms, label=cls, color=CLASS_COLORS[cls])
            for i, (bar, val) in enumerate(zip(bars, values)):
                mid = bottoms[i] + val / 2
                if use_proportion and val >= 4.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        mid,
                        f"{val:.1f}%",
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )
            bottoms += values

        tick_labels = [
            f"{SPLIT_DISPLAY.get(s, s)}\n(n={split_totals[s]:,})" for s in splits
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(ds_name, fontsize=11, fontweight="bold")
        if use_proportion:
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Proportion (%)" if use_proportion else "Clip count", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[cls]) for cls in LEGEND_ORDER]
    fig.legend(handles, LEGEND_ORDER, loc="lower center", ncol=4, fontsize=9,
               title="Engagement class", title_fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Class Distribution by Split", fontsize=13, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(REPO_ROOT)}")


def plot_class_distribution_table(
    all_overall_dists: dict[str, dict[str, dict]],
    path: Path,
    *,
    dataset_names: tuple[str, ...] = ("CMOSE", "DAiSEE"),
) -> None:
    """Render the overall class distribution (share) as a table image."""
    datasets = [ds for ds in dataset_names if ds in all_overall_dists]

    col_labels = ["Engagement class"]
    for ds in datasets:
        col_labels += [ds]

    cell_text: list[list[str]] = []
    cell_colors: list[list[str]] = []
    # Rows top-to-bottom mirror the bar-chart legend: Highly Engage -> Highly Disengage.
    for cls in LEGEND_ORDER:
        row = [f"{cls} ({CLASS_LABEL_SHORT[cls]})"]
        crow = [blend_with_white(CLASS_COLORS[cls], 0.55)]
        for ds in datasets:
            d = all_overall_dists[ds][cls]
            row += [f"{d['proportion'] * 100:.1f}%"]
            crow += ["#FFFFFF"]
        cell_text.append(row)
        cell_colors.append(crow)

    n_cols = len(col_labels)
    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(1.9 + 1.45 * n_cols, 0.9 + 0.45 * n_rows), dpi=200)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)

    for j in range(n_cols):
        header = table[0, j]
        header.set_facecolor("#0072B2")
        header.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        table[i, 0].set_text_props(fontweight="bold")

    ax.set_title(
        f"Overall Class Distribution: {' vs '.join(datasets)}",
        fontsize=12, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(REPO_ROOT)}")


def main() -> None:
    all_entries: dict[str, list[dict]] = {}
    all_overall_dists: dict[str, dict[str, dict]] = {}
    all_split_dists: dict[str, dict[str, dict[str, dict]]] = {}

    for ds_name, json_path in DATASETS.items():
        if not json_path.exists():
            print(f"  [skip] {ds_name}: {json_path} not found")
            continue
        entries = load_entries(json_path)
        all_entries[ds_name] = entries
        all_overall_dists[ds_name] = class_dist(entries)
        all_split_dists[ds_name] = split_class_dist(entries)
        print(f"  Loaded {ds_name}: {len(entries)} samples")

    # --- CSV: overall class distribution per dataset ---
    overall_rows = []
    for ds_name, dist in all_overall_dists.items():
        for cls in CLASS_ORDER:
            overall_rows.append({
                "dataset": ds_name,
                "class_label": cls,
                "count": dist[cls]["count"],
                "proportion": round(dist[cls]["proportion"], 6),
            })
    write_csv(
        OUT_DIR / "class_distribution_overall.csv",
        overall_rows,
        ["dataset", "class_label", "count", "proportion"],
    )
    print(f"  Saved: outputs/dataset_analysis/class_distribution_overall.csv")

    # --- CSV: per-split class distribution per dataset ---
    split_rows = []
    for ds_name, split_dists in all_split_dists.items():
        entries = all_entries[ds_name]
        split_totals = Counter(e.get("split") for e in entries)
        for split in SPLIT_ORDER:
            if split not in split_dists:
                continue
            dist = split_dists[split]
            for cls in CLASS_ORDER:
                split_rows.append({
                    "dataset": ds_name,
                    "split": split,
                    "split_total": split_totals[split],
                    "class_label": cls,
                    "count": dist[cls]["count"],
                    "proportion": round(dist[cls]["proportion"], 6),
                })
    write_csv(
        OUT_DIR / "class_distribution_by_split.csv",
        split_rows,
        ["dataset", "split", "split_total", "class_label", "count", "proportion"],
    )
    print(f"  Saved: outputs/dataset_analysis/class_distribution_by_split.csv")

    # --- Table: overall class distribution, CMOSE vs DAiSEE vs Private ---
    table_dists = dict(all_overall_dists)
    private_dist = load_private_dist()
    if private_dist is not None:
        table_dists["Private"] = private_dist
    plot_class_distribution_table(
        table_dists,
        OUT_DIR / "class_distribution_table.png",
        dataset_names=("CMOSE", "DAiSEE", "Private"),
    )

    # --- Chart: stacked bar by dataset (proportion) ---
    plot_stacked_bar_datasets(
        all_overall_dists,
        OUT_DIR / "class_distribution_stacked_bar.png",
        use_proportion=True,
    )

    # --- Chart: stacked bar by dataset (count) ---
    plot_stacked_bar_datasets(
        all_overall_dists,
        OUT_DIR / "class_distribution_stacked_bar_counts.png",
        use_proportion=False,
    )

    # --- Chart: per-split stacked bar, all datasets in one figure ---
    plot_all_splits_combined(
        all_split_dists,
        all_entries,
        OUT_DIR / "class_distribution_by_split_stacked_bar.png",
        use_proportion=True,
    )

    # --- Charts: per-dataset split breakdown (individual files) ---
    for ds_name in all_split_dists:
        plot_stacked_bar_splits(
            ds_name,
            all_split_dists[ds_name],
            all_entries[ds_name],
            OUT_DIR / ds_name.lower() / "class_distribution_by_split_stacked_bar.png",
            use_proportion=True,
        )

    print("\nDataset analysis complete.")


if __name__ == "__main__":
    main()
