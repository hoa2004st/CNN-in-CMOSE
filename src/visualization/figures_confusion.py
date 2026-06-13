"""Error-analysis figures: confusion matrices and per-class F1 for the best models."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import CLASS_LABEL_SHORT, CLASS_LABELS, FAMILY_COLORS

_METRIC = "quadratic_weighted_kappa"
_BASE_MODELS = {"openface_mlp", "temporal_cnn", "lstm", "transformer", "i3d_mlp"}
_SHORT = [CLASS_LABEL_SHORT[c] for c in CLASS_LABELS]


def _best_runs(hist):
    """Return (best_base_row, best_hybrid_row) for the in-domain CMOSE training runs."""
    cmose = hist[hist["dataset"] == "cmose"].dropna(subset=[_METRIC])
    base = cmose[cmose["model"].isin(_BASE_MODELS)]
    hyb = cmose[cmose["model"].isin({"openface_temporal_hybrid", "openface_temporal_i3d_hybrid"})]
    best_base = base.loc[base[_METRIC].idxmax()]
    best_hyb = hyb.loc[hyb[_METRIC].idxmax()]
    return best_base, best_hyb


def _plot_cm(ax, row, title: str) -> None:
    cm = np.asarray(row["confusion_matrix_normalized"], dtype=float)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_SHORT)))
    ax.set_xticklabels(_SHORT)
    ax.set_yticks(range(len(_SHORT)))
    ax.set_yticklabels(_SHORT)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    return im


def fig_confusion(directory: Path | None = None) -> Path:
    hist = ag.load_training_histories()
    best_base, best_hyb = _best_runs(hist)
    fig, axes = new_fig(1, 2, figsize=(9, 4.2))
    _plot_cm(axes[0], best_base,
             f"Best base: {best_base['model_display']} / {best_base['loss']}\nQWK={best_base[_METRIC]:.3f}")
    im = _plot_cm(axes[1], best_hyb,
                  f"Best hybrid: {best_hyb['model']}\n[{best_hyb['loss']}]  QWK={best_hyb[_METRIC]:.3f}")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Row-normalized")
    fig.suptitle("Confusion matrices, in-domain CMOSE (row-normalized)", y=1.04, fontweight="bold")
    return save(fig, "confusion_best_models", directory=directory)


def fig_per_class_f1(directory: Path | None = None) -> Path:
    hist = ag.load_training_histories()
    best_base, best_hyb = _best_runs(hist)
    base_f1 = ag.per_class_f1(best_base["classification_report_dict"])
    hyb_f1 = ag.per_class_f1(best_hyb["classification_report_dict"])
    width = 0.38
    fig, ax = new_fig(figsize=(6.6, 4.4))
    x = np.arange(len(CLASS_LABELS))
    b1 = ax.bar(x - width / 2, [base_f1.get(c, 0) for c in CLASS_LABELS], width,
                color=FAMILY_COLORS["base"], edgecolor="black", linewidth=0.4,
                label=f"Best base ({best_base['model_display']})")
    b2 = ax.bar(x + width / 2, [hyb_f1.get(c, 0) for c in CLASS_LABELS], width,
                color=FAMILY_COLORS["hybrid"], edgecolor="black", linewidth=0.4,
                label="Best hybrid")
    ax.bar_label(b1, fmt="%.2f", fontsize=6.5, padding=1)
    ax.bar_label(b2, fmt="%.2f", fontsize=6.5, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(_SHORT)
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1: best base vs best hybrid (in-domain CMOSE)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    return save(fig, "per_class_f1", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [fig_confusion(directory), fig_per_class_f1(directory)]
