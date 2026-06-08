"""Semantic-group hybrid ablation figures — the thesis centerpiece.

All three figures use the in-domain cell (train CMOSE, test CMOSE) of the hybrid matrix so
the architecture signal is not confounded by the cross-domain collapse. A reference line for
the best base model (same cell, naive matrix) anchors the comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import display_model_name

_METRIC = "quadratic_weighted_kappa"
_VARIANT_COLOR = {"Hybrid (OpenFace only)": "#56B4E9", "Hybrid + I3D": "#D55E00"}
_ARCH_COLOR = {"TCN": "#0072B2", "T": "#CC79A7", "LSTM": "#E69F00"}


def _indomain_hybrid():
    h = ag.load_hybrid_matrix()
    return h[(h["train_group"] == "cmose") & (h["test_set"] == "cmose_test")].copy()


def _best_base_indomain() -> float:
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    return float(cell[_METRIC].max())


def fig_ablation_distribution(directory: Path | None = None) -> Path:
    """QWK spread across all group-architecture configs, split by +/- I3D."""
    frame = _indomain_hybrid()
    variants = ["Hybrid (OpenFace only)", "Hybrid + I3D"]
    data = [frame[frame["variant"] == v][_METRIC].to_numpy() for v in variants]
    n_configs = max((len(d) for d in data), default=0)
    fig, ax = new_fig(figsize=(6.4, 4.6))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True,
                    medianprops=dict(color="black"))
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(_VARIANT_COLOR[v])
        patch.set_alpha(0.55)
    for i, (v, vals) in enumerate(zip(variants, data), start=1):
        jitter = np.random.default_rng(0).normal(0, 0.05, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=_VARIANT_COLOR[v],
                   edgecolor="black", linewidth=0.3, zorder=3, alpha=0.8)
    base = _best_base_indomain()
    ax.axhline(base, ls="--", color="gray", lw=1.2, label=f"Best base model (QWK={base:.3f})")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(variants)
    ax.set_ylabel("QWK (in-domain CMOSE)")
    ax.set_title(f"Hybrid ablation: QWK across {n_configs} group-architecture configs")
    ax.legend(loc="lower right")
    return save(fig, "hybrid_ablation_distribution", directory=directory)


def fig_group_marginal(directory: Path | None = None) -> Path:
    """For each semantic group, mean QWK when its encoder is TCN, Transformer, or LSTM."""
    frame = _indomain_hybrid()
    groups = ag.OPENFACE_GROUP_ORDER
    tokens = ag.ARCH_TOKENS
    width = 0.8 / len(tokens)
    fig, ax = new_fig(figsize=(8.4, 4.6))
    for j, token in enumerate(tokens):
        means, errs = [], []
        for g in groups:
            vals = frame[frame[f"arch_{g}"] == token][_METRIC]
            means.append(float(vals.mean()))
            errs.append(float(vals.std()))
        offsets = np.arange(len(groups)) + (j - (len(tokens) - 1) / 2) * width
        bars = ax.bar(offsets, means, width=width, yerr=errs, capsize=3,
                      color=_ARCH_COLOR[token], alpha=0.85, edgecolor="white",
                      label=ag.ARCH_TOKEN_DISPLAY[token])
        ax.bar_label(bars, fmt="%.3f", fontsize=6.0, padding=1)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([ag.GROUP_DISPLAY[g] for g in groups], rotation=15, ha="right")
    ax.set_ylabel("Mean QWK (in-domain CMOSE)")
    ax.set_ylim(bottom=max(0.0, frame[_METRIC].min() - 0.05))
    ax.set_title("Per-group marginal effect of encoder choice (TCN vs Transformer vs LSTM)")
    ax.legend(title="Encoder for this group")
    return save(fig, "hybrid_group_marginal", directory=directory)


def fig_best_comparison(directory: Path | None = None) -> Path:
    """Best hybrid (+/- I3D) vs the best base model, in-domain, on QWK and macro-accuracy."""
    h = _indomain_hybrid()
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    best_base = cell.loc[cell[_METRIC].idxmax()]
    best_of = h[~h["has_i3d"]].loc[h[~h["has_i3d"]][_METRIC].idxmax()]
    best_i3d = h[h["has_i3d"]].loc[h[h["has_i3d"]][_METRIC].idxmax()]

    entries = [
        (f"Best base\n({display_model_name(best_base['model'])}/{best_base['loss']})", best_base, "#999999"),
        (f"Best hybrid\nOF-only ({best_of['arch_key']})", best_of, _VARIANT_COLOR["Hybrid (OpenFace only)"]),
        (f"Best hybrid\n+I3D ({best_i3d['arch_key']})", best_i3d, _VARIANT_COLOR["Hybrid + I3D"]),
    ]
    metrics = ["quadratic_weighted_kappa", "macro_accuracy", "accuracy"]
    fig, ax = new_fig(figsize=(7.6, 4.6))
    width = 0.25
    for j, metric in enumerate(metrics):
        offsets = np.arange(len(entries)) + (j - 1) * width
        vals = [float(e[1][metric]) for e in entries]
        bars = ax.bar(offsets, vals, width=width, label=ag.METRIC_DISPLAY[metric],
                      edgecolor="white", linewidth=0.4,
                      color=["#4D4D4D", "#0072B2", "#009E73"][j])
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)
    ax.set_xticks(np.arange(len(entries)))
    ax.set_xticklabels([e[0] for e in entries], fontsize=8)
    ax.set_ylabel("Score (in-domain CMOSE)")
    ax.set_title("Best hybrid vs best base model")
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    return save(fig, "hybrid_best_comparison", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [
        fig_ablation_distribution(directory),
        fig_group_marginal(directory),
        fig_best_comparison(directory),
    ]
