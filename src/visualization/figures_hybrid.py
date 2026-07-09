"""Semantic-group hybrid ablation figure — the thesis centerpiece.

The figure uses the in-domain cell (train CMOSE, test CMOSE) of the hybrid matrix so the
architecture signal is not confounded by the cross-domain collapse. The reference line in
every panel is the SINGLE QWK-selected baseline (same cell, naive matrix) read on that
panel's metric -- one coherent model across all panels, consistent with QWK being the only
selection metric -- not the per-metric best of all baseline configs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.figures_models import ALL_METRICS_PANEL_ORDER
from src.visualization.style import display_model_name

_METRIC = ag.SELECTION_METRIC
_VARIANT_COLOR = {"I3D stream disabled": "#56B4E9", "I3D stream enabled": "#D55E00"}


def _indomain_hybrid():
    h = ag.load_hybrid_matrix()
    return h[(h["train_group"] == "cmose") & (h["test_set"] == "cmose_test")].copy()


def _qwk_best_base_row():
    """The single in-domain CMOSE baseline selected by QWK (the only selection metric)."""
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    return cell.loc[cell[ag.SELECTION_METRIC].idxmax()]


def _ablation_panel(ax, frame, metric: str, variants, base_row) -> None:
    data = [frame[frame["variant"] == v][metric].to_numpy() for v in variants]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True, showfliers=False,
                    medianprops=dict(color="black"),
                    meanprops=dict(marker="D", markersize=5,
                                   markerfacecolor="white", markeredgecolor="black"))
    for mean in bp["means"]:
        mean.set_zorder(4)
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(_VARIANT_COLOR[v])
        patch.set_alpha(0.55)
    for i, (v, vals) in enumerate(zip(variants, data), start=1):
        jitter = np.random.default_rng(0).normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=_VARIANT_COLOR[v],
                   edgecolor="black", linewidth=0.3, zorder=3, alpha=0.8)
    base = float(base_row[metric])
    label = ag.METRIC_DISPLAY[metric]
    ax.axhline(base, ls="--", color="gray", lw=1.2,
               label=f"QWK-selected baseline ({label}={base:.3f})")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(variants)
    ax.set_title(label)
    ax.legend(loc="best", fontsize=7)


def fig_ablation_all_metrics(directory: Path | None = None) -> Path:
    """All six metrics across all configs, split by +/- I3D (the Ch.5 all-metrics overview).

    One panel per metric in a 2x3 grid. The dashed reference line in every panel is the same
    model -- the QWK-selected baseline -- read on that panel's metric, so the bar is one
    coherent model rather than a per-metric champion. QWK and the macro metrics cleanly
    separate the I3D-fused family above that baseline; accuracy is high but flat and barely
    moves off it, so it cannot tell the architectures apart --- the same anti-accuracy point
    made for the baselines, now at the scale of the 243-config ablation.
    """
    frame = _indomain_hybrid()
    base_row = _qwk_best_base_row()
    variants = ["I3D stream disabled", "I3D stream enabled"]
    n_configs = max((len(frame[frame["variant"] == v]) for v in variants), default=0)
    fig, axes = new_fig(2, 3, figsize=(13.5, 8.6))
    for ax, metric in zip(axes.ravel(), ALL_METRICS_PANEL_ORDER):
        _ablation_panel(ax, frame, metric, variants, base_row)
    for row in axes:
        row[0].set_ylabel("Score (in-domain CMOSE)")
    fig.suptitle("Proposed hybrid model on all six metrics (in-domain CMOSE)",
                 y=1.01, fontweight="bold")
    fig.tight_layout()
    return save(fig, "hybrid_ablation_all_metrics", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_ablation_all_metrics(directory)]
