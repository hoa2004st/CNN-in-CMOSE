"""Training-dynamics and loss-tradeoff figures."""

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
_LOSS_MARKER = {"ce": "o", "weighted_ce": "s", "ordinal": "^"}

# QWK (the single primary/selection metric, shown first and emphasised) against the two
# balanced secondary metrics it trades off with; the loss study uses no others.
_TRADEOFF_METRICS = [
    "quadratic_weighted_kappa",
    "macro_accuracy",
    "macro_mae",
]


def fig_loss_metric_tradeoff(directory: Path | None = None) -> Path:
    """Per-metric lines across the three losses: shows what each loss trades.

    One panel per metric (QWK plus the two balanced secondary metrics); one line per base
    model across CE -> Weighted CE -> Ordinal. QWK -- the selection metric -- is generally
    highest under CE, while macro-accuracy rises and macro-MAE drops (improves) as the loss
    is rebalanced, making the Pareto nature of the loss choice explicit.
    """
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")]
    models = [m for m in MODEL_ORDER if m in set(frame["model"])]
    x = np.arange(len(_LOSS_ORDER))
    fig, axes = new_fig(1, len(_TRADEOFF_METRICS), figsize=(3.7 * len(_TRADEOFF_METRICS), 4.0))
    for ax, metric in zip(axes, _TRADEOFF_METRICS):
        for model in models:
            ys = []
            for loss in _LOSS_ORDER:
                sel = frame[(frame["model"] == model) & (frame["loss"] == loss)]
                ys.append(float(sel[metric].iloc[0]) if len(sel) else np.nan)
            ax.plot(x, ys, "-", color=run_color(model), lw=1.6, zorder=2)
            for xi, loss in enumerate(_LOSS_ORDER):
                ax.scatter(xi, ys[xi], s=42, marker=_LOSS_MARKER[loss],
                           color=run_color(model, loss), edgecolor="black",
                           linewidth=0.4, zorder=3,
                           label=display_model_name(model) if loss == "ce" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([loss_display_name(l) for l in _LOSS_ORDER], rotation=15, ha="right")
        title = ag.METRIC_DISPLAY.get(metric, metric)
        ax.set_title(title)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Baseline", loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Loss trade-offs: QWK versus the balanced secondary metrics across losses (in-domain CMOSE)",
                 y=1.04, fontweight="bold")
    return save(fig, "loss_metric_tradeoff", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_loss_metric_tradeoff(directory)]
