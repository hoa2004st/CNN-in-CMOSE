"""Dataset composition figures: class distribution overall."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import CLASS_LABELS, CLASS_LABEL_SHORT, class_color

_DATASET_ORDER = ["CMOSE", "DAiSEE", "Combined"]
_SPLIT_DISPLAY = {"train": "Train", "unlabel": "Val", "test": "Test"}


def _stacked(ax, frame: pd.DataFrame, group_col: str, group_order, value_col: str) -> None:
    groups = [g for g in group_order if g in set(frame[group_col])]
    bottoms = np.zeros(len(groups))
    for label in CLASS_LABELS:
        heights = [
            float(frame[(frame[group_col] == g) & (frame["class_label"] == label)][value_col].sum())
            for g in groups
        ]
        ax.bar(range(len(groups)), heights, bottom=bottoms,
               label=CLASS_LABEL_SHORT.get(label, label), color=class_color(label))
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


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_overall(directory)]
