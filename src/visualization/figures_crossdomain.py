"""Cross-domain generalization figures: 3x3 heatmaps plus the in-domain-vs-transfer scatter.

Beyond the best-per-cell heatmaps, this module asks whether in-domain strength predicts
generalization at all (per-config scatter, hybrid population overlaid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from src.analysis import aggregate as ag
from src.visualization.figbase import autosize_cell_text, mark_cell_texts, new_fig, save
from src.visualization.figures_models import ALL_METRICS_PANEL_ORDER
from src.visualization.style import (
    MODEL_ORDER,
    display_model_name,
    model_color,
)

_METRIC = "quadratic_weighted_kappa"

# Absolute-value colour scale per metric for the 3x3 cross-domain heatmaps, in the same
# metric order as the all-metrics overview grid (Figure base_models_all_metrics). Each entry
# is (vmin, vmax, cmap); the two lower-is-better ordinal-distance metrics use a reversed
# colour map so green always marks the better value.
_HEATMAP_SCALE = {
    "accuracy": (0.2, 0.8, "RdYlGn"),
    "mae": (0.2, 1.1, "RdYlGn_r"),
    "cohen_kappa": (-0.05, 0.55, "RdYlGn"),
    "macro_accuracy": (0.2, 0.7, "RdYlGn"),
    "macro_mae": (0.4, 1.3, "RdYlGn_r"),
    "quadratic_weighted_kappa": (-0.05, 0.6, "RdYlGn"),
}


def _heatmap(ax, pivot, title: str, *, vmin, vmax, cmap="RdYlGn", fmt="{:.3f}",
             show_y=True, show_x=True) -> None:
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([ag.TEST_SET_DISPLAY.get(c, c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    if show_y:
        ax.set_yticklabels([ag.TRAIN_GROUP_DISPLAY.get(r, r) for r in pivot.index])
        ax.set_ylabel("Trained on")
    else:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel("Tested on")
    ax.set_title(title)
    ax.grid(False)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            texts.append(ax.text(j, i, fmt.format(val), ha="center", va="center",
                                 color="black", fontsize=12))
    mark_cell_texts(ax, texts)
    return im


def fig_crossdomain(frame, name: str, directory: Path | None = None) -> Path:
    # All six metrics as 3x3 train x test heatmaps, in the same panel order as the
    # all-metrics overview grid (Figure base_models_all_metrics): the micro block
    # (accuracy, MAE, Cohen kappa) on the top row and the macro / order-aware block
    # (macro-accuracy, macro-MAE, QWK) below it, QWK last. Every cell shows the ONE
    # QWK-selected best config's value on that metric (qwk_selected_cell_matrix), so all six
    # panels profile the single model that would be deployed there. The two lower-is-better
    # metrics (MAE, macro-MAE) use a reversed colour map so green always marks the better
    # value. 2x3 keeps each matrix legible.
    fig, axes = new_fig(2, 3, figsize=(15.0, 8.6), layout="constrained")
    for k, metric in enumerate(ALL_METRICS_PANEL_ORDER):
        ax = axes[k // 3][k % 3]
        vmin, vmax, cmap = _HEATMAP_SCALE[metric]
        pivot = ag.qwk_selected_cell_matrix(frame, metric)
        title = ag.METRIC_DISPLAY[metric]
        im = _heatmap(ax, pivot, title, vmin=vmin, vmax=vmax, cmap=cmap,
                      show_y=(k % 3 == 0), show_x=(k >= 3))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    autosize_cell_text(fig)
    return save(fig, name, directory=directory)


def _delta_heatmap(ax, delta, *, cmap, show_y=True, show_x=True, fmt="{:+.3f}"):
    """Draw one diverging hybrid-minus-base panel, scale centred on zero."""
    data = delta.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data)))
    im = ax.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(delta.columns)))
    ax.set_xticklabels([ag.TEST_SET_DISPLAY.get(c, c) for c in delta.columns])
    ax.set_yticks(range(len(delta.index)))
    if show_y:
        ax.set_yticklabels([ag.TRAIN_GROUP_DISPLAY.get(r, r) for r in delta.index])
        ax.set_ylabel("Trained on")
    else:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel("Tested on")
    ax.grid(False)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            texts.append(ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                                 color="black", fontsize=10))
    mark_cell_texts(ax, texts)
    return im


def fig_crossdomain_delta(directory: Path | None = None) -> Path:
    """Best-hybrid minus best-base in every cell of the 3x3 train x test matrix, all six metrics.

    The cell-by-cell subtraction of the two cross-dataset heatmaps (QWK-selected best hybrid
    per cell minus QWK-selected best base per cell), drawn for all six metrics in the same 2x3
    panel order as the all-metrics overview grid (Figure base_models_all_metrics). Both sides
    select one config per cell by QWK and report its profile, so this is exactly the proposed
    matrix (Figure crossdomain_hybrid) minus the baseline matrix (Figure crossdomain_base).
    Each panel's diverging scale is centred on zero and signed so green is always the
    hybrid-favouring direction (hence ``RdYlGn_r`` for the lower-is-better MAE and macro-MAE,
    where a negative delta is the win). The in-domain CMOSE diagonal cell is the headline
    in-domain gain and the Private column is the private-set gain by training source.
    """
    base_frame = ag.load_matrix()
    hybrid_frame = ag.load_hybrid_matrix()
    fig, axes = new_fig(2, 3, figsize=(15.0, 8.6), layout="constrained")
    for k, metric in enumerate(ALL_METRICS_PANEL_ORDER):
        ax = axes[k // 3][k % 3]
        cmap = "RdYlGn_r" if metric in ag.LOWER_BETTER_METRICS else "RdYlGn"
        delta = (ag.qwk_selected_cell_matrix(hybrid_frame, metric)
                 - ag.qwk_selected_cell_matrix(base_frame, metric))
        im = _delta_heatmap(ax, delta, cmap=cmap, show_y=(k % 3 == 0), show_x=(k >= 3))
        ax.set_title(ag.METRIC_DISPLAY[metric])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="$\\Delta$ (proposed $-$ baseline)")
    # Larger than the other heatmaps' 0.55, since the signed 6-char deltas read small.
    autosize_cell_text(fig, frac=0.75)
    return save(fig, "crossdomain_delta", directory=directory)


def _scatter_points_indomain_transfer(frame, keys: list[str]):
    """One point per config: in-domain QWK (trained and tested on CMOSE) vs transfer QWK
    (trained on Combined, tested on the held-out private set).

    Each config contributes a single (x, y) pair built from two of its own runs -- its
    pure CMOSE in-domain cell and its Combined->Private deployment cell -- so the scatter
    is one point per configuration, not a pool of several training populations (the earlier
    version pooled cmose- and daisee-trained models, two distinct populations, into one
    correlation).
    """
    indomain = frame[(frame["train_group"] == "cmose") &
                     (frame["test_set"] == "cmose_test")].set_index(keys)[_METRIC]
    transfer = frame[(frame["train_group"] == "combined") &
                     (frame["test_set"] == "private")].set_index(keys)[_METRIC]
    joined = indomain.to_frame("in").join(transfer.to_frame("out"), how="inner").reset_index()
    return joined.dropna(subset=["in", "out"])


def fig_indomain_vs_generalization(include_hybrid: bool = True,
                                   directory: Path | None = None) -> Path:
    """Scatter: does a config's CMOSE in-domain QWK predict its private-set transfer?

    x is the pure in-domain cell (trained and tested on CMOSE); y is the deployment cell
    (trained on Combined, tested on the held-out private set). Each config is one point that
    pairs its own two runs, so the scatter is a single population (the earlier version
    pooled cmose- and daisee-trained models). Spearman rho quantifies how much of the
    in-domain ranking survives the shift to the private set; the hybrid overlay shows
    whether the relationship is architecture-specific.
    """
    base_points = _scatter_points_indomain_transfer(ag.load_matrix(), ["model", "loss"])
    fig, ax = new_fig(figsize=(7.6, 5.6))

    hybrid_points = None
    if include_hybrid:
        hybrid_points = _scatter_points_indomain_transfer(
            ag.load_hybrid_matrix(), ["model_type", "arch_key", "loss"])
        ax.scatter(hybrid_points["in"], hybrid_points["out"], s=12, marker="o",
                   color="#BBBBBB", alpha=0.45, linewidth=0,
                   label=f"Proposed-model configs (n={len(hybrid_points)})")
    ax.scatter(base_points["in"], base_points["out"], s=46, marker="o",
               c=[model_color(m) for m in base_points["model"]], edgecolor="black",
               linewidth=0.5, label=f"Baseline configs (n={len(base_points)})")

    rho_base = spearmanr(base_points["in"], base_points["out"]).statistic
    note = f"Spearman $\\rho$ (baseline, n={len(base_points)}) = {rho_base:.2f}"
    if hybrid_points is not None:
        rho_hybrid = spearmanr(hybrid_points["in"], hybrid_points["out"]).statistic
        note += f"\nSpearman $\\rho$ (proposed model, n={len(hybrid_points)}) = {rho_hybrid:.2f}"
    ax.annotate(note, xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("In-domain QWK (trained and tested on CMOSE)")
    ax.set_ylabel("Transfer QWK (trained on Combined, tested on Private)")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker="s", linestyle="", color=model_color(m),
                       label=display_model_name(m)) for m in MODEL_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    name = "indomain_vs_generalization_hybrid" if include_hybrid else "indomain_vs_generalization_base"
    return save(fig, name, directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    base = ag.load_matrix()
    hybrid = ag.load_hybrid_matrix()
    return [
        fig_crossdomain(base, "crossdomain_base", directory),
        fig_crossdomain(hybrid, "crossdomain_hybrid", directory),
        fig_crossdomain_delta(directory),
        fig_indomain_vs_generalization(True, directory),
    ]
