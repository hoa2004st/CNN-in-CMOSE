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


def _heatmap(ax, pivot, title: str, *, vmin, vmax, cmap="RdYlGn", fmt="{:.2f}",
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
                    color="black", fontsize=9,
                    fontweight="bold" if i == j or (pivot.index[i] == "combined") else "normal")
    return im


def fig_crossdomain(frame, name: str, suptitle: str, directory: Path | None = None) -> Path:
    # The three primary metrics (QWK first/emphasised) plus raw accuracy, the deceptive foil:
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
        if metric in ag.LOWER_BETTER_METRICS:
            title += " (lower is better)"
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
    # Bold the two cells the old per-figure comparisons isolated: the in-domain CMOSE
    # diagonal (the headline in-domain gain) and the whole Private column (the unseen,
    # self-collected target).
    private_col = list(delta.columns).index("private")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            highlight = (j == private_col) or (delta.index[i] == "cmose"
                                               and delta.columns[j] == "cmose_test")
            ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center",
                    color="black", fontsize=11,
                    fontweight="bold" if highlight else "normal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$\\Delta$QWK (hybrid $-$ base)")
    fig.suptitle("Hybrid advantage per cell: best hybrid $-$ best base (QWK)",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, "crossdomain_delta", directory=directory)


def _scatter_points(frame, keys: list[str]):
    """(x, y, rows) per trained model: in-domain QWK vs mean QWK on its unseen cells.

    Only cmose- and daisee-trained models have a true in-domain cell; each contributes
    one point with y averaged over that source's unseen-target cells.
    """
    points = []
    for source, in_test in [("cmose", "cmose_test"), ("daisee", "daisee_test")]:
        sub = frame[frame["train_group"] == source]
        indomain = sub[sub["test_set"] == in_test].set_index(keys)[_METRIC]
        unseen_tests = [t for g, t in ag.UNSEEN_TARGET_CELLS if g == source]
        unseen = (sub[sub["test_set"].isin(unseen_tests)]
                  .groupby(keys)[_METRIC].mean())
        joined = indomain.to_frame("in").join(unseen.to_frame("out"), how="inner").reset_index()
        joined["source"] = source
        points.append(joined)
    return pd.concat(points, ignore_index=True)


def fig_indomain_vs_generalization(include_hybrid: bool = False,
                                   directory: Path | None = None) -> Path:
    """Scatter: does a model's in-domain QWK predict its QWK on unseen targets?

    One point per trained model (config x train source); Spearman rho quantifies how much
    of the in-domain ranking survives the shift. The hybrid variant overlays the ablation
    population to show whether the (lack of) relationship is architecture-specific.
    """
    base_points = _scatter_points(ag.load_matrix(), ["model", "loss"])
    source_marker = {"cmose": "o", "daisee": "^"}
    fig, ax = new_fig(figsize=(7.6, 5.6))

    hybrid_points = None
    if include_hybrid:
        hybrid_points = _scatter_points(ag.load_hybrid_matrix(), ["model_type", "arch_key", "loss"])
        for source, marker in source_marker.items():
            sub = hybrid_points[hybrid_points["source"] == source]
            ax.scatter(sub["in"], sub["out"], s=12, marker=marker, color="#BBBBBB",
                       alpha=0.45, linewidth=0,
                       label=f"Hybrid configs ({ag.TRAIN_GROUP_DISPLAY[source]}-trained)")
    for source, marker in source_marker.items():
        sub = base_points[base_points["source"] == source]
        ax.scatter(sub["in"], sub["out"], s=46, marker=marker,
                   c=[model_color(m) for m in sub["model"]], edgecolor="black",
                   linewidth=0.5, label=f"Base configs ({ag.TRAIN_GROUP_DISPLAY[source]}-trained)")

    rho_base = spearmanr(base_points["in"], base_points["out"]).statistic
    note = f"Spearman $\\rho$ (base, n={len(base_points)}) = {rho_base:.2f}"
    if hybrid_points is not None:
        rho_hybrid = spearmanr(hybrid_points["in"], hybrid_points["out"]).statistic
        note += f"\nSpearman $\\rho$ (hybrid, n={len(hybrid_points)}) = {rho_hybrid:.2f}"
    ax.annotate(note, xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=9)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("In-domain QWK (train corpus = test corpus)")
    ax.set_ylabel("Mean QWK on unseen-target cells")
    ax.set_title("Does in-domain strength predict generalization?")
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
        fig_crossdomain_delta(directory),
        fig_indomain_vs_generalization(True, directory),
    ]
