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
from src.visualization.figures_models import ALL_METRICS_PANEL_ORDER
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


def _fig_group_marginal(frame, *, name: str, title: str, ylabel: str,
                        base_value: float, base_label: str,
                        directory: Path | None = None) -> Path:
    """Shared body for the per-group marginal box plots (in-domain and unseen variants)."""
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
    ax.axhline(base_value, ls="--", color="gray", lw=1.2, label=base_label)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([ag.GROUP_DISPLAY[g] for g in groups], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Encoder for this group", loc="upper left", bbox_to_anchor=(1.02, 1))
    return save(fig, name, directory=directory)


def fig_group_marginal(directory: Path | None = None) -> Path:
    """For each semantic group, the QWK distribution when its encoder is TCN, Transformer, or LSTM.

    A box plot (not a bar chart) because the configurations form a distribution: the marginal
    spread per encoder is the point, and a box plot reads honestly without a zero baseline.
    """
    base = _best_base_indomain()
    return _fig_group_marginal(
        _indomain_hybrid(),
        name="hybrid_group_marginal",
        title="Per-group marginal effect of encoder choice (TCN vs Transformer vs LSTM)",
        ylabel="QWK (in-domain CMOSE)",
        base_value=base, base_label=f"Best base model (QWK={base:.3f})",
        directory=directory)


def fig_group_marginal_unseen(directory: Path | None = None) -> Path:
    """The same per-group marginal, pooled over the unseen-target cells.

    Answers whether the in-domain encoder preferences (head pose -> TCN, everything else
    encoder-agnostic) are corpus-specific or survive domain shift.
    """
    h = ag.load_hybrid_matrix()
    frame = h[ag.unseen_target_mask(h)]
    m = ag.load_matrix()
    base = float(m[ag.unseen_target_mask(m)]
                 .groupby(["train_group", "test_set"])[_METRIC].max().mean())
    return _fig_group_marginal(
        frame,
        name="hybrid_group_marginal_unseen",
        title="Per-group marginal effect of encoder choice — unseen-target cells",
        ylabel="QWK (pooled unseen-target cells)",
        base_value=base, base_label=f"Best base model, mean over unseen cells (QWK={base:.3f})",
        directory=directory)


# (train_group, test_set) buckets for the paired I3D ablation: cells where the test corpus
# was seen during training, the two cross-corpus cells, and the (always unseen) private set.
_SEEN_CELLS = [("cmose", "cmose_test"), ("daisee", "daisee_test"),
               ("combined", "cmose_test"), ("combined", "daisee_test")]
_CROSS_CELLS = [("cmose", "daisee_test"), ("daisee", "cmose_test")]


def fig_i3d_paired_delta(directory: Path | None = None) -> Path:
    """Paired effect of fusing I3D: delta QWK between the same arch_key with vs without I3D.

    Pairing each OpenFace-only config with its exact +I3D twin removes the configuration as
    a confounder, so the distribution of deltas IS the I3D effect — shown separately for
    seen-target cells, the cross-corpus cells, and the private set.
    """
    h = ag.load_hybrid_matrix()
    pivot = h.pivot_table(index=["train_group", "test_set", "arch_key", "loss"],
                          columns="has_i3d", values=_METRIC)
    delta = (pivot[True] - pivot[False]).dropna().rename("delta").reset_index()
    cell = list(zip(delta["train_group"], delta["test_set"]))
    delta["regime"] = ["Seen target" if c in _SEEN_CELLS
                       else "Cross-corpus" if c in _CROSS_CELLS
                       else "Private set" for c in cell]
    regimes = ["Seen target", "Cross-corpus", "Private set"]
    data = [delta[delta["regime"] == r]["delta"].to_numpy() for r in regimes]

    fig, ax = new_fig(figsize=(8.0, 4.8))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showmeans=True, showfliers=False,
                    medianprops=dict(color="black"),
                    meanprops=dict(marker="D", markersize=5,
                                   markerfacecolor="white", markeredgecolor="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor(_VARIANT_COLOR["Hybrid + I3D"])
        patch.set_alpha(0.55)
    for i, values in enumerate(data, start=1):
        jitter = np.random.default_rng(0).normal(0, 0.06, size=len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=8, alpha=0.35,
                   color=_VARIANT_COLOR["Hybrid + I3D"], linewidth=0, zorder=3)
    ax.axhline(0, ls="--", color="gray", lw=1.2)
    top = max(np.max(values) for values in data)
    bottom = min(np.min(values) for values in data)
    span = top - bottom
    ax.set_ylim(bottom - 0.04 * span, top + 0.22 * span)  # headroom for the annotations
    for i, values in enumerate(data, start=1):
        share_pos = (values > 0).mean() * 100
        ax.annotate(f"mean {values.mean():+.3f}\n{share_pos:.0f}% > 0",
                    xy=(i, top + 0.20 * span), ha="center", va="top", fontsize=8)
    ax.set_xticks(range(1, len(regimes) + 1))
    ax.set_xticklabels([f"{r}\n(n={len(d)} pairs)" for r, d in zip(regimes, data)])
    ax.set_ylabel("$\\Delta$QWK  (with I3D $-$ without I3D, same config)")
    ax.set_title("Paired effect of I3D fusion across evaluation regimes")
    return save(fig, "i3d_paired_delta", directory=directory)


def fig_best_comparison(directory: Path | None = None) -> Path:
    """Best hybrid (+/- I3D) vs the best base model, in-domain, on the three primary metrics.

    All three primary metrics in a single panel (QWK first/emphasised). On this scale macro-MAE
    sits comfortably alongside the two higher-is-better metrics, so one panel reads cleanly.
    """
    h = _indomain_hybrid()
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    best_base = cell.loc[cell[_METRIC].idxmax()]
    best_of = h[~h["has_i3d"]].loc[h[~h["has_i3d"]][_METRIC].idxmax()]
    best_i3d = h[h["has_i3d"]].loc[h[h["has_i3d"]][_METRIC].idxmax()]

    entries = [
        (f"Best base\n{display_model_name(best_base['model'])}/{best_base['loss']}", best_base),
        ("Best hybrid\n(OF-only)", best_of),
        ("Best hybrid\n(+I3D)", best_i3d),
    ]
    metrics = ["quadratic_weighted_kappa", "macro_accuracy", "macro_mae"]
    metric_color = {"quadratic_weighted_kappa": "#0072B2", "macro_accuracy": "#009E73",
                    "macro_mae": "#D55E00"}
    fig, ax = new_fig(figsize=(8.6, 4.8))
    width = 0.26
    x = np.arange(len(entries))
    for j, metric in enumerate(metrics):
        label = ag.METRIC_DISPLAY[metric] + (" (lower is better)"
                                             if metric in ag.LOWER_BETTER_METRICS else "")
        bars = ax.bar(x + (j - 1) * width, [float(e[1][metric]) for e in entries], width,
                      color=metric_color[metric], edgecolor="white", linewidth=0.4, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in entries], fontsize=8)
    ax.set_ylabel("Score  /  Macro-MAE (class-index units)")
    ax.set_ylim(0, max(float(e[1][m_]) for e in entries for m_ in metrics) * 1.25)
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    fig.suptitle("Best hybrid vs best base model (in-domain CMOSE)", y=1.02, fontweight="bold")
    return save(fig, "hybrid_best_comparison", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [
        fig_ablation_all_metrics(directory),
        fig_group_marginal(directory),
        fig_group_marginal_unseen(directory),
        fig_i3d_paired_delta(directory),
        fig_best_comparison(directory),
    ]
