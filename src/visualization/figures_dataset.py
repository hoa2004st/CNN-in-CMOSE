"""Dataset composition figures: class distribution overall and per split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import CLASS_LABELS, class_color

_DATASET_ORDER = ["CMOSE", "DaiSEE", "Combined"]
_SPLIT_ORDER = ["train", "unlabel", "test"]
_SPLIT_DISPLAY = {"train": "Train", "unlabel": "Val", "test": "Test"}


def _stacked(ax, frame: pd.DataFrame, group_col: str, group_order, value_col: str) -> None:
    groups = [g for g in group_order if g in set(frame[group_col])]
    bottoms = np.zeros(len(groups))
    for label in CLASS_LABELS:
        heights = [
            float(frame[(frame[group_col] == g) & (frame["class_label"] == label)][value_col].sum())
            for g in groups
        ]
        ax.bar(range(len(groups)), heights, bottom=bottoms, label=label, color=class_color(label))
        bottoms += np.asarray(heights)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_SPLIT_DISPLAY.get(g, g) for g in groups])


def fig_overall(directory: Path | None = None) -> Path:
    frame = ag.load_class_distribution_overall()
    fig, ax = new_fig(figsize=(6.2, 4.2))
    _stacked(ax, frame, "dataset", _DATASET_ORDER, "proportion")
    ax.set_ylabel("Class proportion")
    ax.set_ylim(0, 1)
    ax.set_title("Engagement class distribution by dataset")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save(fig, "dataset_class_distribution_overall", directory=directory)


def fig_by_split(directory: Path | None = None) -> Path:
    frame = ag.load_class_distribution_by_split()
    datasets = [d for d in _DATASET_ORDER if d in set(frame["dataset"])]
    fig, axes = new_fig(1, len(datasets), figsize=(3.0 * len(datasets), 3.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        _stacked(ax, frame[frame["dataset"] == dataset], "split", _SPLIT_ORDER, "proportion")
        ax.set_title(dataset)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Class proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Class", bbox_to_anchor=(1.0, 0.9), loc="upper left")
    fig.suptitle("Class distribution per train / val / test split", y=1.02, fontweight="bold")
    return save(fig, "dataset_class_distribution_by_split", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_overall(directory), fig_by_split(directory)]
