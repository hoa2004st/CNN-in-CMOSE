"""Training-dynamics and loss-tradeoff figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import (
    CURVE_COLORS,
    MODEL_ORDER,
    display_model_name,
    loss_display_name,
    run_color,
)

_LOSS_ORDER = ["ce", "weighted_ce", "ordinal"]
_LOSS_MARKER = {"ce": "o", "weighted_ce": "s", "ordinal": "^"}


def fig_loss_curves(directory: Path | None = None) -> Path:
    """Representative train/val loss curves: CMOSE base models, CE loss."""
    hist = ag.load_training_histories()
    models = [m for m in MODEL_ORDER if m in set(hist["model"])]
    fig, axes = new_fig(1, len(models), figsize=(2.6 * len(models), 3.4), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sel = hist[(hist["dataset"] == "cmose") & (hist["model"] == model) & (hist["loss"] == "ce")]
        if not len(sel):
            ax.set_visible(False)
            continue
        row = sel.iloc[0]
        epochs = np.arange(1, len(row["train_losses"]) + 1)
        ax.plot(epochs, row["train_losses"], color=CURVE_COLORS["train_loss"], label="Train")
        ax.plot(epochs, row["eval_losses"], color=CURVE_COLORS["eval_loss"], label="Val")
        if row["best_epoch"] is not None:
            ax.axvline(row["best_epoch"] + 1, color=CURVE_COLORS["best_epoch"],
                       ls="--", lw=0.9, alpha=0.7)
        ax.set_title(display_model_name(model), fontsize=9)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (CE)")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
            break
    fig.suptitle("Training vs validation loss (CMOSE, cross-entropy); dashed = best epoch",
                 y=1.03, fontweight="bold")
    return save(fig, "loss_curves_cmose_ce", directory=directory)


def fig_loss_tradeoff(directory: Path | None = None) -> Path:
    """Accuracy vs macro-accuracy, in-domain, colored/marked by loss."""
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")]
    fig, ax = new_fig(figsize=(6.0, 5.2))
    for loss in _LOSS_ORDER:
        sub = frame[frame["loss"] == loss]
        ax.scatter(sub["accuracy"], sub["macro_accuracy"],
                   s=70, marker=_LOSS_MARKER[loss],
                   color=[run_color(m, loss) for m in sub["model"]],
                   edgecolor="black", linewidth=0.5, label=loss_display_name(loss), zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(display_model_name(r["model"]), (r["accuracy"], r["macro_accuracy"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.7)
    lo = float(min(frame["accuracy"].min(), frame["macro_accuracy"].min())) - 0.03
    ax.plot([lo, 0.8], [lo, 0.8], ls=":", color="gray", lw=1, label="Acc = Macro-Acc")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-accuracy")
    ax.set_title("Accuracy vs macro-accuracy tradeoff by loss\n(in-domain CMOSE -> CMOSE)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    return save(fig, "loss_accuracy_tradeoff", directory=directory)


_TRADEOFF_METRICS = [
    "accuracy",
    "macro_accuracy",
    "macro_mae",
    "quadratic_weighted_kappa",
]


def fig_loss_metric_tradeoff(directory: Path | None = None) -> Path:
    """Per-metric lines across the three losses: shows what each loss trades.

    One panel per primary metric; one line per base model across CE -> Weighted CE ->
    Ordinal. Accuracy falls, macro-accuracy rises, and macro-MAE drops (improves) as the
    loss is rebalanced, making the Pareto nature of the loss choice explicit.
    """
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")]
    models = [m for m in MODEL_ORDER if m in set(frame["model"])]
    x = np.arange(len(_LOSS_ORDER))
    fig, axes = new_fig(1, len(_TRADEOFF_METRICS), figsize=(3.7 * len(_TRADEOFF_METRICS), 4.0))
    for ax, metric in zip(axes, _TRADEOFF_METRICS):
        for model in models:
            ys = []
            for loss in _LOSS_ORDER:
                sel = frame[(frame["model"] == model) & (frame["loss"] == loss)]
                ys.append(float(sel[metric].iloc[0]) if len(sel) else np.nan)
            ax.plot(x, ys, "-", color=run_color(model), lw=1.6, zorder=2)
            for xi, loss in enumerate(_LOSS_ORDER):
                ax.scatter(xi, ys[xi], s=42, marker=_LOSS_MARKER[loss],
                           color=run_color(model, loss), edgecolor="black",
                           linewidth=0.4, zorder=3,
                           label=display_model_name(model) if loss == "ce" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([loss_display_name(l) for l in _LOSS_ORDER], rotation=15, ha="right")
        title = ag.METRIC_DISPLAY.get(metric, metric)
        if metric in ag.LOWER_BETTER_METRICS:
            title += " (lower is better)"
        ax.set_title(title)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Base model", loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("What each loss trades: primary metrics across losses (in-domain CMOSE)",
                 y=1.04, fontweight="bold")
    return save(fig, "loss_metric_tradeoff", directory=directory)


def fig_loss_pareto_mae(directory: Path | None = None) -> Path:
    """Accuracy vs macro-MAE Pareto by loss --- the ordinal-distance face of the trade-off.

    Companion to ``loss_accuracy_tradeoff`` (accuracy vs macro-accuracy): here the y-axis is
    the balanced ordinal error, inverted so that better (lower macro-MAE) is up, so Pareto-best
    points sit upper-right exactly as in the macro-accuracy view.
    """
    matrix = ag.load_matrix()
    frame = matrix[(matrix["train_group"] == "cmose") & (matrix["test_set"] == "cmose_test")]
    fig, ax = new_fig(figsize=(6.0, 5.2))
    for loss in _LOSS_ORDER:
        sub = frame[frame["loss"] == loss]
        ax.scatter(sub["accuracy"], sub["macro_mae"],
                   s=70, marker=_LOSS_MARKER[loss],
                   color=[run_color(m, loss) for m in sub["model"]],
                   edgecolor="black", linewidth=0.5, label=loss_display_name(loss), zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(display_model_name(r["model"]), (r["accuracy"], r["macro_mae"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.7)
    ax.invert_yaxis()  # lower macro-MAE is better -> better points sit higher
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Macro-MAE (lower is better)")
    ax.set_title("Accuracy vs macro-MAE tradeoff by loss\n(in-domain CMOSE -> CMOSE)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    return save(fig, "loss_pareto_mae", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [
        fig_loss_curves(directory),
        fig_loss_tradeoff(directory),
        fig_loss_metric_tradeoff(directory),
        fig_loss_pareto_mae(directory),
    ]
