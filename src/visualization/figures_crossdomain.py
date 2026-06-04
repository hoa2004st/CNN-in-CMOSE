"""Cross-domain generalization heatmaps (3x3 train x test, best model per cell)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save


def _heatmap(ax, pivot, title: str, *, vmin, vmax, fmt="{:.2f}", show_y=True) -> None:
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([ag.TEST_SET_DISPLAY.get(c, c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    if show_y:
        ax.set_yticklabels([ag.TRAIN_GROUP_DISPLAY.get(r, r) for r in pivot.index])
        ax.set_ylabel("Trained on")
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Tested on")
    ax.set_title(title)
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    color="black", fontsize=9,
                    fontweight="bold" if i == j or (pivot.index[i] == "combined") else "normal")
    return im


def fig_crossdomain(frame, name: str, suptitle: str, directory: Path | None = None) -> Path:
    fig, axes = new_fig(1, 2, figsize=(11, 4.4), layout="constrained")
    for k, (ax, metric, vmin, vmax) in enumerate((
        (axes[0], "accuracy", 0.0, 0.8),
        (axes[1], "quadratic_weighted_kappa", -0.05, 0.6),
    )):
        pivot = ag.cell_matrix(frame, metric)
        im = _heatmap(ax, pivot, ag.METRIC_DISPLAY[metric], vmin=vmin, vmax=vmax, show_y=(k == 0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontweight="bold")
    return save(fig, name, directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    base = ag.load_matrix()
    hybrid = ag.load_hybrid_matrix()
    return [
        fig_crossdomain(base, "crossdomain_base",
                        "Cross-domain generalization — best base model per cell", directory),
        fig_crossdomain(hybrid, "crossdomain_hybrid",
                        "Cross-domain generalization — best hybrid config per cell", directory),
    ]
