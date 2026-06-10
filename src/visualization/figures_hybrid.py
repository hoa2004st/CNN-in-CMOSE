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


def _best_base_indomain(metric: str = _METRIC) -> float:
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    return float(cell[metric].min() if metric in ag.LOWER_BETTER_METRICS else cell[metric].max())


def _ablation_panel(ax, frame, metric: str, variants) -> None:
    data = [frame[frame["variant"] == v][metric].to_numpy() for v in variants]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True,
                    medianprops=dict(color="black"))
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
    suffix = " — lower is better" if metric in ag.LOWER_BETTER_METRICS else ""
    ax.set_ylabel(f"{label} (in-domain CMOSE){suffix}")
    ax.set_title(label)
    ax.legend(loc="upper left", fontsize=8)


def fig_ablation_distribution(directory: Path | None = None) -> Path:
    """QWK and macro-MAE spread across all group-architecture configs, split by +/- I3D."""
    frame = _indomain_hybrid()
    variants = ["Hybrid (OpenFace only)", "Hybrid + I3D"]
    n_configs = max((len(frame[frame["variant"] == v]) for v in variants), default=0)
    fig, axes = new_fig(1, 2, figsize=(11, 4.6))
    _ablation_panel(axes[0], frame, _METRIC, variants)
    _ablation_panel(axes[1], frame, "macro_mae", variants)
    fig.suptitle(f"Hybrid ablation across {n_configs} group-architecture configs "
                 f"(in-domain CMOSE)", y=1.02, fontweight="bold")
    return save(fig, "hybrid_ablation_distribution", directory=directory)


def fig_group_marginal(directory: Path | None = None) -> Path:
    """For each semantic group, the QWK distribution when its encoder is TCN, Transformer, or LSTM.

    A box plot (not a bar chart) because the configurations form a distribution: the marginal
    spread per encoder is the point, and a box plot reads honestly without a zero baseline.
    """
    frame = _indomain_hybrid()
    groups = ag.OPENFACE_GROUP_ORDER
    tokens = ag.ARCH_TOKENS
    width = 0.8 / len(tokens)
    fig, ax = new_fig(figsize=(8.4, 4.6))
    for j, token in enumerate(tokens):
        data = [frame[frame[f"arch_{g}"] == token][_METRIC].to_numpy() for g in groups]
        offsets = np.arange(len(groups)) + (j - (len(tokens) - 1) / 2) * width
        bp = ax.boxplot(data, positions=offsets, widths=width * 0.9, patch_artist=True,
                        showmeans=True, manage_ticks=False,
                        medianprops=dict(color="black", linewidth=1.0),
                        meanprops=dict(marker="D", markersize=3.5,
                                       markerfacecolor="white", markeredgecolor="black"),
                        flierprops=dict(marker="o", markersize=2.5, alpha=0.5,
                                        markerfacecolor=_ARCH_COLOR[token],
                                        markeredgecolor="none"))
        for patch in bp["boxes"]:
            patch.set_facecolor(_ARCH_COLOR[token])
            patch.set_alpha(0.55)
        for whisker in bp["whiskers"] + bp["caps"]:
            whisker.set_color(_ARCH_COLOR[token])
        bp["boxes"][0].set_label(ag.ARCH_TOKEN_DISPLAY[token])
    base = _best_base_indomain()
    ax.axhline(base, ls="--", color="gray", lw=1.2, label=f"Best base model (QWK={base:.3f})")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([ag.GROUP_DISPLAY[g] for g in groups], rotation=15, ha="right")
    ax.set_ylabel("QWK (in-domain CMOSE)")
    ax.set_title("Per-group marginal effect of encoder choice (TCN vs Transformer vs LSTM)")
    ax.legend(title="Encoder for this group", loc="upper left", bbox_to_anchor=(1.02, 1))
    return save(fig, "hybrid_group_marginal", directory=directory)


def fig_best_comparison(directory: Path | None = None) -> Path:
    """Best hybrid (+/- I3D) vs the best base model, in-domain, on the three primary metrics.

    The two higher-is-better metrics (QWK, macro-accuracy) share the left axis; macro-MAE
    (lower is better) gets its own panel so the differing scale and direction read honestly.
    """
    h = _indomain_hybrid()
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    best_base = cell.loc[cell[_METRIC].idxmax()]
    best_of = h[~h["has_i3d"]].loc[h[~h["has_i3d"]][_METRIC].idxmax()]
    best_i3d = h[h["has_i3d"]].loc[h[h["has_i3d"]][_METRIC].idxmax()]

    entries = [
        (f"Best base\n{display_model_name(best_base['model'])}/{best_base['loss']}", best_base, "#999999"),
        ("Best hybrid\n(OF-only)", best_of, _VARIANT_COLOR["Hybrid (OpenFace only)"]),
        ("Best hybrid\n(+I3D)", best_i3d, _VARIANT_COLOR["Hybrid + I3D"]),
    ]
    fig, (axL, axR) = new_fig(1, 2, figsize=(11, 4.6))
    width = 0.38
    for j, metric in enumerate(["quadratic_weighted_kappa", "macro_accuracy"]):
        offsets = np.arange(len(entries)) + (j - 0.5) * width
        vals = [float(e[1][metric]) for e in entries]
        bars = axL.bar(offsets, vals, width=width, label=ag.METRIC_DISPLAY[metric],
                       edgecolor="white", linewidth=0.4, color=["#0072B2", "#009E73"][j])
        axL.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    axL.set_xticks(np.arange(len(entries)))
    axL.set_xticklabels([e[0] for e in entries], fontsize=8)
    axL.set_ylabel("Score (in-domain CMOSE)")
    axL.set_title("QWK and Macro-Accuracy (higher is better)")
    axL.set_ylim(0, max(float(e[1]["macro_accuracy"]) for e in entries) * 1.45)
    axL.legend(title="Metric", loc="upper left", ncol=2, fontsize=8)

    vals = [float(e[1]["macro_mae"]) for e in entries]
    bars = axR.bar(np.arange(len(entries)), vals, width=0.55,
                   color=[e[2] for e in entries], edgecolor="black", linewidth=0.4)
    axR.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    axR.set_xticks(np.arange(len(entries)))
    axR.set_xticklabels([e[0] for e in entries], fontsize=8)
    axR.set_ylabel("Macro-MAE (class-index units)")
    axR.set_title("Macro-MAE (lower is better)")
    fig.suptitle("Best hybrid vs best base model (in-domain CMOSE)", y=1.02, fontweight="bold")
    return save(fig, "hybrid_best_comparison", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [
        fig_ablation_distribution(directory),
        fig_group_marginal(directory),
        fig_best_comparison(directory),
    ]
