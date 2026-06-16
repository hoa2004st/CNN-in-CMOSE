"""Semantic-group hybrid ablation figure — the thesis centerpiece.

The figure uses the in-domain cell (train CMOSE, test CMOSE) of the hybrid matrix so the
architecture signal is not confounded by the cross-domain collapse. A reference line for the
best base model (same cell, naive matrix) anchors the comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.figures_models import ALL_METRICS_PANEL_ORDER

_METRIC = "quadratic_weighted_kappa"
_VARIANT_COLOR = {"Hybrid (OpenFace only)": "#56B4E9", "Hybrid + I3D": "#D55E00"}


def _indomain_hybrid():
    h = ag.load_hybrid_matrix()
    return h[(h["train_group"] == "cmose") & (h["test_set"] == "cmose_test")].copy()


def _best_base_indomain(metric: str = _METRIC) -> float:
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    return float(cell[metric].min() if metric in ag.LOWER_BETTER_METRICS else cell[metric].max())


def _ablation_panel(ax, frame, metric: str, variants) -> None:
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
    base = _best_base_indomain(metric)
    label = ag.METRIC_DISPLAY[metric]
    ax.axhline(base, ls="--", color="gray", lw=1.2, label=f"Best base model ({label}={base:.3f})")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(variants)
    suffix = " (lower is better)" if metric in ag.LOWER_BETTER_METRICS else ""
    ax.set_title(label + suffix)
    ax.legend(loc="best", fontsize=7)


def fig_ablation_all_metrics(directory: Path | None = None) -> Path:
    """All six metrics across all configs, split by +/- I3D (the Ch.5 all-metrics overview).

    One panel per metric in a 2x3 grid, each against the best base model on that metric.
    QWK and the macro metrics cleanly separate the I3D-fused family above the baseline;
    accuracy is high but flat and barely moves off the baseline, so it cannot tell the
    architectures apart --- the same anti-accuracy point made for the baselines, now at the
    scale of the 243-config ablation.
    """
    frame = _indomain_hybrid()
    variants = ["Hybrid (OpenFace only)", "Hybrid + I3D"]
    n_configs = max((len(frame[frame["variant"] == v]) for v in variants), default=0)
    fig, axes = new_fig(2, 3, figsize=(13.5, 8.6))
    for ax, metric in zip(axes.ravel(), ALL_METRICS_PANEL_ORDER):
        _ablation_panel(ax, frame, metric, variants)
    for row in axes:
        row[0].set_ylabel("Score (in-domain CMOSE)")
    fig.suptitle(f"Hybrid ablation across {n_configs} group-architecture configs "
                 f"on all six metrics (in-domain CMOSE)", y=1.01, fontweight="bold")
    fig.tight_layout()
    return save(fig, "hybrid_ablation_all_metrics", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_ablation_all_metrics(directory)]
