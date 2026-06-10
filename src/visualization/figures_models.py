"""Base-model comparison figures (in-domain CMOSE -> CMOSE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import (
    MODEL_ORDER,
    display_model_name,
    loss_display_name,
    run_color,
)

_LOSS_ORDER = ["ce", "weighted_ce", "ordinal"]


def _grouped_bar(ax, frame, metric: str) -> None:
    models = [m for m in MODEL_ORDER if m in set(frame["model"])]
    width = 0.26
    for j, loss in enumerate(_LOSS_ORDER):
        offsets = np.arange(len(models)) + (j - 1) * width
        values, colors = [], []
        for model in models:
            sel = frame[(frame["model"] == model) & (frame["loss"] == loss)]
            values.append(float(sel[metric].iloc[0]) if len(sel) else np.nan)
            colors.append(run_color(model, loss))
        bars = ax.bar(offsets, values, width=width, color=colors,
                      label=loss_display_name(loss), edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=20, ha="right")
    title = ag.METRIC_DISPLAY.get(metric, metric)
    if metric in ag.LOWER_BETTER_METRICS:
        title += " (lower is better)"
    ax.set_title(title)


def fig_indomain_base(directory: Path | None = None) -> Path:
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")]
    fig, axes = new_fig(1, 3, figsize=(15.5, 4.6))
    _grouped_bar(axes[0], frame, "quadratic_weighted_kappa")
    _grouped_bar(axes[1], frame, "macro_accuracy")
    _grouped_bar(axes[2], frame, "macro_mae")
    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Macro-MAE (class-index units)")
    axes[2].legend(title="Loss", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Base models x losses, in-domain (CMOSE -> CMOSE)", y=1.02, fontweight="bold")
    return save(fig, "base_models_indomain", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_indomain_base(directory)]
