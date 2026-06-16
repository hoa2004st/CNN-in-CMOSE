"""Private-set figure — the real-world, self-collected, test-only generalization story.

The private set is labeled by hand and never used for training, so for it there is *no*
in-domain option: the only lever is the choice of training source. This figure shows the
private-set confusion of the best combined-trained base model versus the best hybrid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import (
    CLASS_LABEL_SHORT,
    CLASS_LABELS,
    display_model_name,
)

_METRIC = "quadratic_weighted_kappa"
_SHORT = [CLASS_LABEL_SHORT[c] for c in CLASS_LABELS]


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
    m = ag.load_matrix()
    preds = ag.load_naive_predictions()
    cell = m[(m["train_group"] == "combined") & (m["test_set"] == "private")]
    best = cell.loc[cell[_METRIC].idxmax()]
    slc = preds[(preds["train_group"] == "combined") & (preds["test_set"] == "private")
                & (preds["model"] == best["model"]) & (preds["loss"] == best["loss"])]
    cm = ag.confusion_from_predictions(slc).to_numpy()
    title = (f"Best base: {display_model_name(best['model'])}/{best['loss']}\n"
             f"QWK={float(best[_METRIC]):.3f}, macro-MAE={float(best['macro_mae']):.3f}")
    return cm, title


def _hybrid_private_confusion():
    """(confusion, title) of the best combined-trained hybrid on the private set, or None."""
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
    if not len(slc):
        return None
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


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_private_confusion(directory)]
