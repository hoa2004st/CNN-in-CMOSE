"""Prediction-agreement and metric-reliability figures.

Two views the score tables cannot give: a pairwise Cohen-kappa heatmap over every in-domain
base config (do models give the same answers?), and a Kendall-tau heatmap showing how far the
six metrics agree on the model ranking (which metric to trust).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.analysis import agreement as agr
from src.visualization.figbase import new_fig, save
from src.visualization.style import COMPARISON_LOSS_SLUGS


def _annotated_heatmap(ax, matrix, *, vmin, vmax, cmap, fmt="{:.2f}", fontsize=7,
                       diverging=False):
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isnan(value):
                continue
            # White text only on dark cells: far from the center for a diverging map,
            # near the top for a sequential one.
            if diverging:
                dark = abs(value - (vmin + vmax) / 2) / ((vmax - vmin) / 2) > 0.6
            else:
                dark = (value - vmin) / (vmax - vmin) > 0.6
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=fontsize,
                    color="white" if dark else "black")
    ax.grid(False)
    return im


def fig_agreement_base(directory: Path | None = None) -> Path:
    """Pairwise Cohen-kappa between all in-domain base configs (model-major blocks).

    Block structure answers where disagreement comes from: the 3x3 diagonal blocks are the
    same architecture under different losses, off-diagonal blocks are different
    architectures. kappa is chance-corrected, so imbalance does not inflate it.
    """
    wide, _ = agr.indomain_prediction_table()
    kappa = agr.pairwise_kappa(wide)
    labels = [agr.config_label(m, l) for m, l in kappa.columns]

    fig, ax = new_fig(figsize=(9.6, 8.2))
    im = _annotated_heatmap(ax, kappa, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    n_loss = len(COMPARISON_LOSS_SLUGS)
    for cut in range(n_loss, len(labels), n_loss):  # separate the per-model blocks
        ax.axhline(cut - 0.5, color="white", lw=1.6)
        ax.axvline(cut - 0.5, color="white", lw=1.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Pairwise prediction agreement (Cohen $\\kappa$) — in-domain CMOSE")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen $\\kappa$")
    return save(fig, "agreement_base_models", directory=directory)


def _fig_metric_correlation(frame, name: str, subtitle: str,
                            directory: Path | None = None) -> Path:
    corr = agr.metric_rank_correlation(frame)
    labels = [ag.METRIC_DISPLAY[m] + ("*" if m in ag.LOWER_BETTER_METRICS else "")
              for m in corr.columns]
    fig, ax = new_fig(figsize=(6.4, 5.2))
    im = _annotated_heatmap(ax, corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", fontsize=9,
                            diverging=True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"Metric rank agreement (Kendall $\\tau$) — {subtitle}\n"
                 "(*sign-flipped: lower is better)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Kendall $\\tau$")
    return save(fig, name, directory=directory)


def fig_metric_correlation_base(directory: Path | None = None) -> Path:
    """Do the six metrics rank the 15 base configs the same way (in-domain CMOSE)?"""
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    return _fig_metric_correlation(cell, "metric_correlation_base",
                                   "15 base configs, in-domain CMOSE", directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [
        fig_agreement_base(directory),
        fig_metric_correlation_base(directory),
    ]
