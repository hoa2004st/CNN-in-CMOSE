"""Base-model comparison figures (in-domain CMOSE -> CMOSE)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import (
    MODEL_ORDER,
    display_model_name,
    loss_display_name,
    run_color,
)

_LOSS_ORDER = ["ce", "weighted_ce", "ordinal"]


def _grouped_bar(ax, frame, metric: str) -> None:
    models = [m for m in MODEL_ORDER if m in set(frame["model"])]
    width = 0.26
    for j, loss in enumerate(_LOSS_ORDER):
        offsets = np.arange(len(models)) + (j - 1) * width
        values, colors = [], []
        for model in models:
            sel = frame[(frame["model"] == model) & (frame["loss"] == loss)]
            values.append(float(sel[metric].iloc[0]) if len(sel) else np.nan)
            colors.append(run_color(model, loss))
        bars = ax.bar(offsets, values, width=width, color=colors,
                      label=loss_display_name(loss), edgecolor="white", linewidth=0.4)
        ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=1)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([display_model_name(m) for m in models], rotation=20, ha="right")
    title = ag.METRIC_DISPLAY.get(metric, metric)
    if metric in ag.LOWER_BETTER_METRICS:
        title += " (lower is better)"
    ax.set_title(title)


def fig_indomain_base(directory: Path | None = None) -> Path:
    """Accuracy vs QWK, base models x losses (the anti-accuracy disproof figure).

    The opening section uses exactly these two metrics: accuracy (which the majority class
    inflates) against QWK (chance-corrected, order-aware). The I3D MLP tops accuracy while the
    OpenFace TCN tops QWK, so accuracy and QWK disagree on the winner --- the reason the thesis
    abandons accuracy as the yardstick.
    """
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")]
    fig, axes = new_fig(1, 2, figsize=(11.5, 4.6))
    _grouped_bar(axes[0], frame, "accuracy")
    _grouped_bar(axes[1], frame, "quadratic_weighted_kappa")
    axes[0].set_ylabel("Score")
    axes[1].legend(title="Loss", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Accuracy vs QWK: base models x losses, in-domain (CMOSE -> CMOSE)",
                 y=1.02, fontweight="bold")
    return save(fig, "base_models_indomain", directory=directory)


_OPENFACE_MODELS = ["openface_mlp", "temporal_cnn", "lstm", "transformer"]
# The three primary metrics, QWK first (emphasised). Used everywhere after the opening section.
_PRIMARY_METRICS = ["quadratic_weighted_kappa", "macro_accuracy", "macro_mae"]


def fig_openface_vs_i3d(directory: Path | None = None) -> Path:
    """Best OpenFace encoder vs the I3D MLP, in-domain CMOSE under cross-entropy.

    All three primary metrics in one panel. The per-frame OpenFace descriptor with a temporal
    encoder wins every order-aware metric, led by QWK; the contrast with the global I3D clip
    vector --- complementary rather than redundant --- motivates fusing the two streams.
    """
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")
                   & (matrix["loss"] == "ce")]
    of = frame[frame["model"].isin(_OPENFACE_MODELS)]
    best_of = of.loc[of["quadratic_weighted_kappa"].idxmax()]
    i3d = frame[frame["model"] == "i3d_mlp"].iloc[0]
    series = {
        f'OpenFace ({display_model_name(best_of["model"])})': best_of,
        display_model_name("i3d_mlp"): i3d,
    }
    fig, ax = new_fig(figsize=(8.2, 4.6))
    x = np.arange(len(_PRIMARY_METRICS))
    width = 0.38
    for j, (label, row) in enumerate(series.items()):
        color = run_color(row["model"], "ce")
        bars = ax.bar(x + (j - 0.5) * width, [float(row[m]) for m in _PRIMARY_METRICS],
                      width=width, color=color, edgecolor="white", linewidth=0.5, label=label)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels([ag.METRIC_DISPLAY.get(m, m)
                        + ("\n(lower is better)" if m in ag.LOWER_BETTER_METRICS else "")
                        for m in _PRIMARY_METRICS])
    ax.set_ylabel("Score  /  Macro-MAE (class-index units)")
    ax.set_ylim(0, max(float(r[m]) for r in series.values() for m in _PRIMARY_METRICS) * 1.35)
    ax.legend(title="Feature family", loc="upper center", ncol=2)
    fig.suptitle("OpenFace vs I3D features, in-domain CMOSE (cross-entropy)",
                 y=1.02, fontweight="bold")
    return save(fig, "openface_vs_i3d", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_indomain_base(directory), fig_openface_vs_i3d(directory)]
