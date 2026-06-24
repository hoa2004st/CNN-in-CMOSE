"""Private-set figures — the real-world, self-collected, test-only generalization story.

The private set is labeled by hand and never used for training, so for it there is *no*
in-domain option: the only lever is the choice of training source. These figures compare
the best combined-trained base model against the best combined-trained hybrid on the
private set: a side-by-side confusion matrix and a per-class F1 bar chart that localises
where the hybrid's generalization gain lands.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import (
    CLASS_LABEL_SHORT,
    CLASS_LABELS,
    FAMILY_COLORS,
    display_model_name,
)

_METRIC = "quadratic_weighted_kappa"
_SHORT = [CLASS_LABEL_SHORT[c] for c in CLASS_LABELS]


# --------------------------------------------------------------------------------------
# Best combined-trained model selection on the private set (shared by both figures)
# --------------------------------------------------------------------------------------
def _best_base_combined_private():
    """(best_row, prediction_slice) for the best combined-trained base model on private."""
    m = ag.load_matrix()
    preds = ag.load_naive_predictions()
    cell = m[(m["train_group"] == "combined") & (m["test_set"] == "private")]
    best = cell.loc[cell[_METRIC].idxmax()]
    slc = preds[(preds["train_group"] == "combined") & (preds["test_set"] == "private")
                & (preds["model"] == best["model"]) & (preds["loss"] == best["loss"])]
    return best, slc


def _best_hybrid_combined_private():
    """(best_row, prediction_slice) for the best combined-trained hybrid, or None.

    None when the large hybrid prediction log is an unfetched git-LFS pointer.
    """
    hyb = ag.load_hybrid_matrix()
    hyb_preds = ag.load_hybrid_predictions()
    if hyb_preds is None:
        return None
    cell = hyb[(hyb["train_group"] == "combined") & (hyb["test_set"] == "private")]
    best = cell.loc[cell[_METRIC].idxmax()]
    slc = hyb_preds[(hyb_preds["train_group"] == "combined")
                    & (hyb_preds["test_set"] == "private")
                    & (hyb_preds["model_type"] == best["model"])
                    & (hyb_preds["arch_key"] == best["arch_key"])
                    & (hyb_preds["loss"] == best["loss"])]
    return (best, slc) if len(slc) else None


def _plot_private_cm(ax, cm, title: str):
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_SHORT))); ax.set_xticklabels(_SHORT)
    ax.set_yticks(range(len(_SHORT))); ax.set_yticklabels(_SHORT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_title(title, fontsize=9)
    return im


def _base_private_confusion():
    """(confusion, title) of the best combined-trained base model on the private set."""
    best, slc = _best_base_combined_private()
    cm = ag.confusion_from_predictions(slc).to_numpy()
    title = (f"Best base: {display_model_name(best['model'])}/{best['loss']}\n"
             f"QWK={float(best[_METRIC]):.3f}, macro-MAE={float(best['macro_mae']):.3f}")
    return cm, title


def _hybrid_private_confusion():
    """(confusion, title) of the best combined-trained hybrid on the private set, or None."""
    selection = _best_hybrid_combined_private()
    if selection is None:
        return None
    best, slc = selection
    cm = ag.confusion_from_predictions(slc).to_numpy()
    title = (f"Best hybrid: {best['variant']} {best['arch_key']}\n"
             f"QWK={float(best[_METRIC]):.3f}, macro-MAE={float(best['macro_mae']):.3f}")
    return cm, title


def fig_private_confusion(directory: Path | None = None) -> Path:
    """Side-by-side private-set confusion: best base vs best hybrid (both combined-trained).

    Mirrors the in-domain base-vs-hybrid confusion figure so the two regimes read the same
    way. Falls back to the base panel alone when the hybrid prediction log is unavailable
    (e.g. the large LFS file has not been fetched).
    """
    panels = [_base_private_confusion()]
    hybrid_panel = _hybrid_private_confusion()
    if hybrid_panel is not None:
        panels.append(hybrid_panel)

    fig, axes = new_fig(1, len(panels), figsize=(4.7 * len(panels) + 0.6, 4.2))
    axes = np.atleast_1d(axes)
    for ax, (cm, title) in zip(axes, panels):
        im = _plot_private_cm(ax, cm, title)
    fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.04, label="Row-normalized")
    fig.suptitle("Private set (combined-trained, row-normalized)", y=1.04, fontweight="bold")
    return save(fig, "private_confusion_combined", directory=directory)


def fig_private_per_class_f1(directory: Path | None = None) -> Path:
    """Per-class F1 on the private set: best combined-trained base vs best hybrid.

    The bar-chart companion to :func:`fig_private_confusion`; it localises where the
    hybrid's out-of-domain QWK gain lands across the four engagement classes. Falls back to
    the base bars alone when the hybrid prediction log is an unfetched LFS pointer.
    """
    best_base, base_slc = _best_base_combined_private()
    base_f1 = ag.per_class_f1_from_predictions(base_slc)

    width = 0.38
    fig, ax = new_fig(figsize=(6.6, 4.4))
    x = np.arange(len(CLASS_LABELS))
    selection = _best_hybrid_combined_private()
    offset = width / 2 if selection is not None else 0.0
    b1 = ax.bar(x - offset, [base_f1.get(c, 0) for c in CLASS_LABELS], width,
                color=FAMILY_COLORS["base"], edgecolor="black", linewidth=0.4,
                label=f"Best base ({display_model_name(best_base['model'])})")
    ax.bar_label(b1, fmt="%.2f", fontsize=6.5, padding=1)
    if selection is not None:
        _, hyb_slc = selection
        hyb_f1 = ag.per_class_f1_from_predictions(hyb_slc)
        b2 = ax.bar(x + width / 2, [hyb_f1.get(c, 0) for c in CLASS_LABELS], width,
                    color=FAMILY_COLORS["hybrid"], edgecolor="black", linewidth=0.4,
                    label="Best hybrid")
        ax.bar_label(b2, fmt="%.2f", fontsize=6.5, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(_SHORT)
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1: best base vs best hybrid (private set, combined-trained)")
    ax.legend(loc="upper left")
    return save(fig, "private_per_class_f1", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_private_confusion(directory), fig_private_per_class_f1(directory)]
