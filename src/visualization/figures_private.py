"""Private-set figures — the real-world, self-collected, test-only generalization story.

The private set is labeled by hand and never used for training, so for it there is *no*
in-domain option: the only lever is the choice of training source. These figures show that
(a) combined-corpus training + the hybrid architecture jointly give the best private-set
result, and (b) why the architecture ablation is run on CMOSE (DaiSEE in-domain signal is weak).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.analysis import aggregate as ag
from src.visualization.figbase import new_fig, save
from src.visualization.style import CLASS_LABEL_SHORT, CLASS_LABELS, display_model_name

_METRIC = "quadratic_weighted_kappa"
_SHORT = [CLASS_LABEL_SHORT[c] for c in CLASS_LABELS]
_SRC_COLOR = {"cmose": "#0072B2", "daisee": "#D55E00", "combined": "#009E73"}


def _best_private(frame, src: str, metric: str) -> "pd.Series":
    """Best (by QWK) row for a training source on the private set; metrics read off that row.

    Selection stays anchored to QWK so all three primary metrics describe the *same* model
    rather than three differently-chosen ``best'' models.
    """
    cell = frame[(frame["train_group"] == src) & (frame["test_set"] == "private")]
    return cell.loc[cell[_METRIC].idxmax()]


def fig_private_by_source(directory: Path | None = None) -> Path:
    """Best base vs best hybrid on the private set, per training source: QWK and macro-MAE."""
    m = ag.load_matrix()
    h = ag.load_hybrid_matrix()
    sources = ["cmose", "daisee", "combined"]
    base_rows = [_best_private(m, s, _METRIC) for s in sources]
    hyb_rows = [_best_private(h, s, _METRIC) for s in sources]

    x = np.arange(len(sources))
    width = 0.38
    fig, (axL, axR) = new_fig(1, 2, figsize=(11, 4.6))

    def _panel(ax, metric, ylabel, title):
        base = [float(r[metric]) for r in base_rows]
        hyb = [float(r[metric]) for r in hyb_rows]
        b1 = ax.bar(x - width / 2, base, width, color="#BBBBBB",
                    edgecolor="black", linewidth=0.4, label="Best base model")
        b2 = ax.bar(x + width / 2, hyb, width, color=[_SRC_COLOR[s] for s in sources],
                    edgecolor="black", linewidth=0.4, label="Best semantic-group hybrid")
        ax.bar_label(b1, fmt="%.3f", fontsize=7, padding=1)
        ax.bar_label(b2, fmt="%.3f", fontsize=7, padding=1)
        ax.set_xticks(x)
        ax.set_xticklabels([ag.TRAIN_GROUP_DISPLAY[s] for s in sources])
        ax.set_xlabel("Training source (private set is test-only)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    _panel(axL, _METRIC, "QWK on private set", "QWK (higher is better)")
    _panel(axR, "macro_mae", "Macro-MAE on private set", "Macro-MAE (lower is better)")
    axL.legend(loc="upper left", fontsize=8)
    fig.suptitle("Private-set generalization by training source\n"
                 "(self-collected, never seen in training)", y=1.04, fontweight="bold")
    return save(fig, "private_by_source", directory=directory)


def fig_indomain_cmose_vs_daisee(directory: Path | None = None) -> Path:
    """Justify the CMOSE-centric ablation: DaiSEE in-domain signal is much weaker."""
    m = ag.load_matrix()
    pairs = [("cmose", "cmose_test", "CMOSE"), ("daisee", "daisee_test", "DaiSEE")]
    # Single best-QWK model per dataset; read all three primary metrics from that row (matches T7).
    best_rows = [m[(m["train_group"] == tr) & (m["test_set"] == te)]
                 .pipe(lambda c: c.loc[c["quadratic_weighted_kappa"].idxmax()])
                 for tr, te, _ in pairs]

    fig, (axL, axR) = new_fig(1, 2, figsize=(10.5, 4.4))
    for i, best in enumerate(best_rows):
        for metric, dx, color in (("quadratic_weighted_kappa", -0.18, "#0072B2"),
                                   ("macro_accuracy", 0.18, "#56B4E9")):
            bar = axL.bar(i + dx, float(best[metric]), 0.34, color=color,
                          label=ag.METRIC_DISPLAY[metric] if i == 0 else None,
                          edgecolor="white", linewidth=0.4)
            axL.bar_label(bar, fmt="%.3f", fontsize=8, padding=1)
        bar = axR.bar(i, float(best["macro_mae"]), 0.5, color="#D55E00",
                      edgecolor="white", linewidth=0.4)
        axR.bar_label(bar, fmt="%.3f", fontsize=8, padding=1)
        axR.annotate(f"{display_model_name(best['model'])}/{best['loss']}",
                     (i, 0.02), ha="center", va="bottom", fontsize=7, color="white")
    for ax in (axL, axR):
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([p[2] for p in pairs])
        ax.set_xlabel("In-domain dataset (train = test)")
    axL.set_ylabel("Score")
    axL.set_title("QWK and Macro-Accuracy (higher is better)")
    axL.set_ylim(0, max(float(r["macro_accuracy"]) for r in best_rows) * 1.4)
    axL.legend(loc="upper left", ncol=2, fontsize=8)
    axR.set_ylabel("Macro-MAE (class-index units)")
    axR.set_title("Macro-MAE (lower is better)")
    fig.suptitle("In-domain signal: CMOSE vs DaiSEE (best-QWK model per dataset)",
                 y=1.02, fontweight="bold")
    return save(fig, "indomain_cmose_vs_daisee", directory=directory)


def fig_private_confusion(directory: Path | None = None) -> Path:
    """Confusion on the private set of the best combined-trained model.

    Features the headline model — the best combined-trained *hybrid* — when its prediction
    log is available; falls back to the best combined *base* model otherwise (e.g. when the
    large hybrid LFS file has not been fetched).
    """
    hyb = ag.load_hybrid_matrix()
    hyb_preds = ag.load_hybrid_predictions()
    title_model = None
    cm = None

    if hyb_preds is not None:
        cell = hyb[(hyb["train_group"] == "combined") & (hyb["test_set"] == "private")]
        best = cell.loc[cell[_METRIC].idxmax()]
        slc = hyb_preds[(hyb_preds["train_group"] == "combined")
                        & (hyb_preds["test_set"] == "private")
                        & (hyb_preds["model_type"] == best["model"])
                        & (hyb_preds["arch_key"] == best["arch_key"])
                        & (hyb_preds["loss"] == best["loss"])]
        if len(slc):
            cm = ag.confusion_from_predictions(slc).to_numpy()
            title_model = f"{best['variant']} {best['arch_key']}"
            qwk = float(best[_METRIC])
            macro_mae = float(best["macro_mae"])

    if cm is None:  # fallback: best combined base model
        m = ag.load_matrix()
        preds = ag.load_naive_predictions()
        cell = m[(m["train_group"] == "combined") & (m["test_set"] == "private")]
        best = cell.loc[cell[_METRIC].idxmax()]
        slc = preds[(preds["train_group"] == "combined") & (preds["test_set"] == "private")
                    & (preds["model"] == best["model"]) & (preds["loss"] == best["loss"])]
        cm = ag.confusion_from_predictions(slc).to_numpy()
        title_model = f"{display_model_name(best['model'])}/{best['loss']}"
        qwk = float(best[_METRIC])
        macro_mae = float(best["macro_mae"])

    fig, ax = new_fig(figsize=(5.4, 4.6))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(_SHORT))); ax.set_xticklabels(_SHORT)
    ax.set_yticks(range(len(_SHORT))); ax.set_yticklabels(_SHORT)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized")
    ax.set_title(f"Private set — combined-trained {title_model}\n"
                 f"(row-normalized; QWK={qwk:.3f}, macro-MAE={macro_mae:.3f})", fontsize=9)
    return save(fig, "private_confusion_combined", directory=directory)


def make_all(directory: Path | None = None) -> list[Path]:
    return [
        fig_private_by_source(directory),
        fig_indomain_cmose_vs_daisee(directory),
        fig_private_confusion(directory),
    ]
