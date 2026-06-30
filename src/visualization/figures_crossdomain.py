"""Cross-domain generalization figures: 3x3 heatmaps plus the in-domain-vs-transfer scatter.

Beyond the best-per-cell heatmaps, this module asks whether in-domain strength predicts
generalization at all (per-config scatter, hybrid population overlaid).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import (
    MODEL_ORDER,
    display_model_name,
    model_color,
)

_METRIC = "quadratic_weighted_kappa"


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
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    color="black", fontsize=12)
    return im


def fig_crossdomain(frame, name: str, suptitle: str, directory: Path | None = None) -> Path:
    # QWK (the single primary/selection metric, first/emphasised) and the two balanced
    # secondary metrics, plus raw accuracy, the deceptive foil:
    # its high off-diagonal cells against near-zero QWK are the disproof-of-accuracy evidence.
    # Macro-MAE on a reversed colour scale, since lower is better. 2x2 so each matrix stays large.
    fig, axes = new_fig(2, 2, figsize=(10.6, 8.4), layout="constrained")
    for k, (ax, metric, vmin, vmax, cmap) in enumerate((
        (axes[0][0], "quadratic_weighted_kappa", -0.05, 0.6, "RdYlGn"),
        (axes[0][1], "macro_accuracy", 0.2, 0.7, "RdYlGn"),
        (axes[1][0], "macro_mae", 0.4, 1.3, "RdYlGn_r"),
        (axes[1][1], "accuracy", 0.3, 0.8, "RdYlGn"),
    )):
        pivot = ag.cell_matrix(frame, metric)
        title = ag.METRIC_DISPLAY[metric]
        im = _heatmap(ax, pivot, title, vmin=vmin, vmax=vmax, cmap=cmap,
                      show_y=(k % 2 == 0), show_x=(k >= 2))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(suptitle, fontweight="bold")
    return save(fig, name, directory=directory)


def fig_crossdomain_delta(directory: Path | None = None) -> Path:
    """Best-hybrid minus best-base QWK in every cell of the 3x3 train x test matrix.

    The cell-by-cell subtraction of the two cross-dataset QWK heatmaps (best hybrid per
    cell minus best base per cell). One panel unifies the two comparisons that were
    previously separate figures: the in-domain CMOSE diagonal cell is the headline
    in-domain gain, and the Private column is the private-set gain by training source.
    Every cell is positive --- the hybrid wins the whole matrix --- and the gain is
    largest on the decisive Private column.
    """
    base = ag.cell_matrix(ag.load_matrix(), _METRIC)
    hybrid = ag.cell_matrix(ag.load_hybrid_matrix(), _METRIC)
    delta = hybrid - base
    data = delta.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data)))

    fig, ax = new_fig(figsize=(6.6, 5.2))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(delta.columns)))
    ax.set_xticklabels([ag.TEST_SET_DISPLAY.get(c, c) for c in delta.columns])
    ax.set_yticks(range(len(delta.index)))
    ax.set_yticklabels([ag.TRAIN_GROUP_DISPLAY.get(r, r) for r in delta.index])
    ax.set_xlabel("Tested on")
    ax.set_ylabel("Trained on")
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center",
                    color="black", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta$QWK (hybrid $-$ base)")
    fig.suptitle("Hybrid advantage per cell: best hybrid $-$ best base (QWK)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, "crossdomain_delta", directory=directory)


def _delta_heatmap(ax, delta, *, cmap, show_y=True, show_x=True,
                   fmt="{:+.3f}", highlight_fn=None):
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
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            highlight = highlight_fn(i, j) if highlight_fn else False
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                    color="black", fontsize=10,
                    fontweight="bold" if highlight else "normal")
    return im


def fig_crossdomain_delta_full(directory: Path | None = None) -> Path:
    """Four-metric version of the per-cell hybrid-minus-base figure.

    The same cell-by-cell (best hybrid - best base) subtraction as
    ``fig_crossdomain_delta``, drawn for all four cross-domain metrics in a 2x2 grid
    (mirroring ``fig_crossdomain``'s layout) so the QWK headline can be read against
    accuracy, macro-accuracy and the lower-is-better macro-MAE. Each panel's diverging
    scale is centred on zero and signed so green is always the hybrid-favouring
    direction (hence ``RdYlGn_r`` for macro-MAE, where a negative delta is the win).
    The Private column and the in-domain CMOSE diagonal cell are bolded in every panel.
    """
    base_frame = ag.load_matrix()
    hybrid_frame = ag.load_hybrid_matrix()
    fig, axes = new_fig(2, 2, figsize=(10.6, 8.4), layout="constrained")
    for k, (ax, metric, cmap) in enumerate((
        (axes[0][0], "quadratic_weighted_kappa", "RdYlGn"),
        (axes[0][1], "macro_accuracy", "RdYlGn"),
        (axes[1][0], "macro_mae", "RdYlGn_r"),
        (axes[1][1], "accuracy", "RdYlGn"),
    )):
        delta = ag.cell_matrix(hybrid_frame, metric) - ag.cell_matrix(base_frame, metric)
        private_col = list(delta.columns).index("private")

        def highlight_fn(i, j, d=delta, pc=private_col):
            return (j == pc) or (d.index[i] == "cmose" and d.columns[j] == "cmose_test")

        im = _delta_heatmap(ax, delta, cmap=cmap, show_y=(k % 2 == 0),
                            show_x=(k >= 2), highlight_fn=highlight_fn)
        title = ag.METRIC_DISPLAY[metric]
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="$\\Delta$ (hybrid $-$ base)")
    fig.suptitle("Hybrid advantage per cell: best hybrid $-$ best base",
                 fontweight="bold")
    return save(fig, "crossdomain_delta_full", directory=directory)


def _architecture_ranking(frame) -> pd.DataFrame:
    """Per-architecture in-domain QWK vs mean QWK over the unseen-target cells.

    In-domain is the development cell (trained and tested on CMOSE), best over loss.
    Unseen is the mean over the five unseen-target cells (best over loss within each cell),
    i.e. the genuine generalization regime ``ag.UNSEEN_TARGET_CELLS``. Returned indexed by
    model key in ``MODEL_ORDER``, with columns ``in_domain`` and ``unseen``.
    """
    indomain = (frame[(frame["train_group"] == "cmose") & (frame["test_set"] == "cmose_test")]
                .groupby("model")[_METRIC].max())
    unseen_cells = frame[ag.unseen_target_mask(frame)]
    best_per_cell = (unseen_cells
                     .groupby(["model", "train_group", "test_set"])[_METRIC].max()
                     .reset_index())
    unseen = best_per_cell.groupby("model")[_METRIC].mean()
    table = pd.DataFrame({"in_domain": indomain, "unseen": unseen})
    return table.reindex([m for m in MODEL_ORDER if m in table.index])


def fig_architecture_ranking_shift(directory: Path | None = None) -> Path:
    """Slopegraph: the architecture QWK ranking is reshuffled out of domain.

    Left column is in-domain QWK (trained and tested on CMOSE); right column is the mean
    QWK over the five unseen-target cells. One coloured line per architecture; the crossings
    are the reshuffle. The I3D MLP is the headline: second in-domain, it drops to fourth on
    unseen targets, while the OpenFace TCN and Transformer hold the top. Ranks (1 = best) are
    annotated at each end so the order change is legible despite the out-of-domain collapse.
    """
    table = _architecture_ranking(ag.load_matrix())
    x_in, x_out = 0.0, 1.0
    in_rank = table["in_domain"].rank(ascending=False, method="first").astype(int)
    out_rank = table["unseen"].rank(ascending=False, method="first").astype(int)
    y_top = max(table["in_domain"]) * 1.08

    def _declutter(values: pd.Series, gap: float) -> dict:
        """Nudge label y-positions apart (top-down) so close lines' labels don't overlap;
        the markers stay at the true value, only the text moves."""
        placed = {}
        prev = None
        for model, y in values.sort_values(ascending=False).items():
            y = min(y, prev - gap) if prev is not None else y
            placed[model] = y
            prev = y
        return placed

    gap = 0.028 * y_top
    in_label_y = _declutter(table["in_domain"], gap)
    out_label_y = _declutter(table["unseen"], gap)

    fig, ax = new_fig(figsize=(7.4, 5.6))
    for model, row in table.iterrows():
        color = model_color(model)
        ax.plot([x_in, x_out], [row["in_domain"], row["unseen"]], "-o",
                color=color, linewidth=2.2, markersize=7, zorder=3)
        ax.annotate(f"{display_model_name(model)}  ({in_rank[model]})",
                    xy=(x_in, in_label_y[model]), xytext=(-10, 0),
                    textcoords="offset points", ha="right",
                    va="center", color=color, fontsize=9, fontweight="bold")
        ax.annotate(f"({out_rank[model]})  {display_model_name(model)}",
                    xy=(x_out, out_label_y[model]), xytext=(10, 0),
                    textcoords="offset points", ha="left",
                    va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_xticks([x_in, x_out])
    ax.set_xticklabels(["In-domain\n(CMOSE)", "Unseen targets\n(mean of 5 cells)"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, max(table["in_domain"]) * 1.08)
    ax.set_ylabel("QWK")
    ax.set_title("The architecture ranking is reshuffled out of domain")
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    fig.tight_layout()
    return save(fig, "architecture_ranking_shift", directory=directory)


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
                   label=f"Hybrid configs (n={len(hybrid_points)})")
    ax.scatter(base_points["in"], base_points["out"], s=46, marker="o",
               c=[model_color(m) for m in base_points["model"]], edgecolor="black",
               linewidth=0.5, label=f"Base configs (n={len(base_points)})")

    rho_base = spearmanr(base_points["in"], base_points["out"]).statistic
    note = f"Spearman $\\rho$ (base, n={len(base_points)}) = {rho_base:.2f}"
    if hybrid_points is not None:
        rho_hybrid = spearmanr(hybrid_points["in"], hybrid_points["out"]).statistic
        note += f"\nSpearman $\\rho$ (hybrid, n={len(hybrid_points)}) = {rho_hybrid:.2f}"
    ax.annotate(note, xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("In-domain QWK (trained and tested on CMOSE)")
    ax.set_ylabel("Transfer QWK (trained on Combined, tested on Private)")
    ax.set_title("In-domain CMOSE QWK versus private-set transfer")
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
        fig_crossdomain(base, "crossdomain_base",
                        "Cross-domain generalization — best base model per cell", directory),
        fig_crossdomain(hybrid, "crossdomain_hybrid",
                        "Cross-domain generalization — best hybrid config per cell", directory),
        fig_architecture_ranking_shift(directory),
        fig_crossdomain_delta(directory),
        fig_crossdomain_delta_full(directory),
        fig_indomain_vs_generalization(True, directory),
    ]
