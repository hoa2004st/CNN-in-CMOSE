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
    axes[0].legend(loc="upper right")
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
    ax.legend(loc="lower right")
    return save(fig, "loss_accuracy_tradeoff", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_loss_curves(directory), fig_loss_tradeoff(directory)]
