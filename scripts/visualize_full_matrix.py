"""Comprehensive model-assessment visualizations from full_matrix.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import MODEL_DISPLAY_NAMES, MODEL_ORDER, MODEL_COLORS

FULL_MATRIX_CSV = REPO_ROOT / "outputs" / "model_assessment" / "full_matrix.csv"
OUT_DIR = REPO_ROOT / "outputs" / "model_assessment" / "full_matrix_analysis"

# Normalize CSV model names to match style keys
_MODEL_KEY_MAP = {"tcn": "temporal_cnn"}

TRAIN_ORDER = ["cmose", "daisee", "combined"]
TEST_ORDER  = ["cmose_test", "daisee_test", "private"]
LOSS_ORDER  = ["ce", "weighted_ce", "ordinal"]

TRAIN_LABELS = {"cmose": "CMOSE", "daisee": "DaiSEE", "combined": "Combined"}
TEST_LABELS  = {"cmose_test": "CMOSE Test", "daisee_test": "DaiSEE Test", "private": "Private"}
LOSS_LABELS  = {"ce": "CE", "weighted_ce": "Weighted CE", "ordinal": "Ordinal"}
TRAIN_COLORS = {"cmose": "#0072B2", "daisee": "#D55E00", "combined": "#009E73"}
LOSS_COLORS  = {"ce": "#0072B2", "weighted_ce": "#D55E00", "ordinal": "#009E73"}
TEST_COLORS  = {"cmose_test": "#0072B2", "daisee_test": "#D55E00", "private": "#009E73"}

METRICS = [
    ("accuracy",               "Accuracy",        True),
    ("macro_accuracy",         "Macro Accuracy",  True),
    ("mae",                    "MAE",             False),
    ("macro_mae",              "Macro MAE",       False),
    ("cohen_kappa",            "Cohen's Kappa",   True),
    ("quadratic_weighted_kappa", "QWK",           True),
]
HIGHER = {m for m, _, h in METRICS if h}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(FULL_MATRIX_CSV)
    df["model"] = df["model"].map(lambda m: _MODEL_KEY_MAP.get(m, m))
    df["model_display"] = df["model"].map(lambda m: MODEL_DISPLAY_NAMES.get(m, m))
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_color(val: float, vmin: float, vmax: float) -> str:
    """White text on dark cells, dark text on light cells."""
    norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    return "white" if norm > 0.55 else "#111111"


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ---------------------------------------------------------------------------
# 1. Cross-dataset heatmaps  (train_group × test_set, mean over models+losses)
# ---------------------------------------------------------------------------

def plot_cross_dataset_heatmaps(df: pd.DataFrame) -> None:
    """2×3 panel: one 3×3 heatmap per metric showing mean metric for every
    training-group × test-set combination."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    train_labels = [TRAIN_LABELS[t] for t in TRAIN_ORDER]
    test_labels  = [TEST_LABELS[t]  for t in TEST_ORDER]

    for ax, (metric, label, higher) in zip(axes, METRICS):
        pivot = (
            df.pivot_table(index="train_group", columns="test_set",
                           values=metric, aggfunc="mean")
            .reindex(index=TRAIN_ORDER, columns=TEST_ORDER)
        )
        vals = pivot.to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        cmap = "viridis" if higher else "viridis_r"
        im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(TEST_ORDER)),  test_labels,  fontsize=9,  rotation=18, ha="right")
        ax.set_yticks(range(len(TRAIN_ORDER)), train_labels, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")

        for r in range(len(TRAIN_ORDER)):
            for c in range(len(TEST_ORDER)):
                v = vals[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color=_text_color(v, vmin, vmax))

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Mean Metrics: Training Group × Test Set\n"
        "(averaged over 6 models × 3 losses — diagonal ≈ in-domain)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "cross_dataset_heatmaps.png")


# ---------------------------------------------------------------------------
# 2. Training group comparison  (bars per test set, grouped by train_group)
# ---------------------------------------------------------------------------

def plot_training_group_comparison(df: pd.DataFrame) -> None:
    """2×3 panel: for each metric, grouped bars showing how CMOSE / DaiSEE /
    Combined training compares on each test set (mean over models and losses)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    x = np.arange(len(TEST_ORDER))
    width = 0.25

    for ax, (metric, label, _) in zip(axes, METRICS):
        for i, tg in enumerate(TRAIN_ORDER):
            means = [
                df.loc[(df["train_group"] == tg) & (df["test_set"] == ts), metric].mean()
                for ts in TEST_ORDER
            ]
            ax.bar(x + (i - 1) * width, means, width=width,
                   label=TRAIN_LABELS[tg], color=TRAIN_COLORS[tg], alpha=0.85)

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Training Group Comparison Across Test Sets\n"
        "(averaged over 6 models × 3 losses)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "training_group_comparison.png")


# ---------------------------------------------------------------------------
# 3. Loss effect on metrics
# ---------------------------------------------------------------------------

def plot_loss_effect(df: pd.DataFrame) -> None:
    """2×3 panel: for each metric, show how CE / Weighted-CE / Ordinal
    loss affects performance on each test set (mean over models and train groups)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    x = np.arange(len(TEST_ORDER))
    width = 0.25

    for ax, (metric, label, _) in zip(axes, METRICS):
        for i, loss in enumerate(LOSS_ORDER):
            means = [
                df.loc[(df["loss"] == loss) & (df["test_set"] == ts), metric].mean()
                for ts in TEST_ORDER
            ]
            ax.bar(x + (i - 1) * width, means, width=width,
                   label=LOSS_LABELS[loss], color=LOSS_COLORS[loss], alpha=0.85)

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Loss Function Effect on Metrics by Test Set\n"
        "(averaged over 6 models × 3 training groups)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "loss_effect_by_metric.png")


# ---------------------------------------------------------------------------
# 4. Model comparison: line chart across test sets
# ---------------------------------------------------------------------------

def plot_model_comparison(df: pd.DataFrame) -> None:
    """2×3 panel: for each metric, one line per model showing mean performance
    across the three test sets (averaged over losses and training groups)."""
    ordered = [m for m in MODEL_ORDER if m in df["model"].values]
    x = np.arange(len(TEST_ORDER))

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    for ax, (metric, label, _) in zip(axes, METRICS):
        for model in ordered:
            means = [
                df.loc[(df["model"] == model) & (df["test_set"] == ts), metric].mean()
                for ts in TEST_ORDER
            ]
            color = MODEL_COLORS.get(model, "#4D4D4D")
            disp  = MODEL_DISPLAY_NAMES.get(model, model)
            ax.plot(x, means, marker="o", markersize=7, linewidth=2.2,
                    color=color, label=disp)

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.2)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Model Comparison Across Test Sets\n"
        "(averaged over 3 losses × 3 training groups)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    _save(fig, "model_comparison_by_testset.png")


# ---------------------------------------------------------------------------
# 5. Model × Loss interaction heatmap
# ---------------------------------------------------------------------------

def plot_model_loss_interaction(df: pd.DataFrame) -> None:
    """2×3 panel: heatmap of model × loss for each metric,
    averaged across training groups and test sets."""
    ordered = [m for m in MODEL_ORDER if m in df["model"].values]
    model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in ordered]
    loss_labels  = [LOSS_LABELS[l] for l in LOSS_ORDER]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=150)
    axes = axes.flatten()

    for ax, (metric, label, higher) in zip(axes, METRICS):
        pivot = (
            df.pivot_table(index="model", columns="loss",
                           values=metric, aggfunc="mean")
            .reindex(index=ordered, columns=LOSS_ORDER)
        )
        vals = pivot.to_numpy(dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        cmap = "viridis" if higher else "viridis_r"
        im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(len(LOSS_ORDER)), loss_labels,  fontsize=9)
        ax.set_yticks(range(len(ordered)),    model_labels, fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")

        for r in range(len(ordered)):
            for c in range(len(LOSS_ORDER)):
                v = vals[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color=_text_color(v, vmin, vmax))

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Model × Loss Interaction\n"
        "(averaged over 3 training groups × 3 test sets)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "model_loss_interaction_heatmap.png")


# ---------------------------------------------------------------------------
# 6. Rank stability bump chart
# ---------------------------------------------------------------------------

def plot_rank_stability(df: pd.DataFrame) -> None:
    """Bump charts showing how model rankings change across the three test sets
    for Accuracy and QWK (averaged over losses and training groups)."""
    ordered = [m for m in MODEL_ORDER if m in df["model"].values]
    x = np.arange(len(TEST_ORDER))

    key_metrics = [
        ("accuracy",               "Accuracy"),
        ("quadratic_weighted_kappa", "QWK"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    for ax, (metric, label) in zip(axes, key_metrics):
        ranks_by_ts: dict[str, dict[str, int]] = {}
        for ts in TEST_ORDER:
            means = {
                m: df.loc[(df["model"] == m) & (df["test_set"] == ts), metric].mean()
                for m in ordered
            }
            sorted_models = sorted(means, key=means.__getitem__,
                                   reverse=(metric in HIGHER))
            ranks_by_ts[ts] = {m: sorted_models.index(m) + 1 for m in ordered}

        for model in ordered:
            ranks = [ranks_by_ts[ts][model] for ts in TEST_ORDER]
            color = MODEL_COLORS.get(model, "#4D4D4D")
            disp  = MODEL_DISPLAY_NAMES.get(model, model)
            ax.plot(x, ranks, marker="o", markersize=10, linewidth=2.5,
                    color=color, label=disp)
            ax.annotate(disp, (x[-1] + 0.06, ranks[-1]),
                        fontsize=7.5, color=color, va="center")

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=10)
        ax.set_yticks(range(1, len(ordered) + 1))
        ax.set_ylim(0.5, len(ordered) + 0.5)
        ax.invert_yaxis()
        ax.set_title(f"Model Rankings — {label}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Rank  (1 = Best)", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xlim(-0.3, len(TEST_ORDER) - 0.3)

    fig.suptitle(
        "Model Rank Stability Across Test Sets\n"
        "(averaged over 3 losses × 3 training groups)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "model_rank_stability.png")


# ---------------------------------------------------------------------------
# 7. Generalization profile  (diagonal = in-domain highlighted)
# ---------------------------------------------------------------------------

def plot_generalization_profile(df: pd.DataFrame) -> None:
    """For each training group, show per-test-set performance on key metrics.
    The 'own' test set (CMOSE↔CMOSE, DaiSEE↔DaiSEE) is highlighted with a star."""
    in_domain = {"cmose": "cmose_test", "daisee": "daisee_test", "combined": None}
    x = np.arange(len(TEST_ORDER))
    width = 0.22

    key_metrics = [
        ("accuracy",               "Accuracy"),
        ("mae",                    "MAE"),
        ("quadratic_weighted_kappa", "QWK"),
        ("cohen_kappa",            "Cohen's Kappa"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

    for ax, (metric, label) in zip(axes, key_metrics):
        for i, tg in enumerate(TRAIN_ORDER):
            means = [
                df.loc[(df["train_group"] == tg) & (df["test_set"] == ts), metric].mean()
                for ts in TEST_ORDER
            ]
            bars = ax.bar(x + (i - 1) * width, means, width=width,
                          label=TRAIN_LABELS[tg], color=TRAIN_COLORS[tg], alpha=0.85)

            # Star on in-domain bar
            own_ts = in_domain[tg]
            if own_ts in TEST_ORDER:
                ts_idx = TEST_ORDER.index(own_ts)
                bar_x  = ts_idx + (i - 1) * width
                bar_y  = means[ts_idx]
                offset = 0.02 if metric not in {"mae", "macro_mae"} else -0.06
                ax.annotate("★", (bar_x, bar_y + offset),
                            ha="center", fontsize=10, color=TRAIN_COLORS[tg])

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Generalization Profile by Training Group\n"
        "(★ = in-domain evaluation; averaged over 6 models × 3 losses)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    _save(fig, "generalization_profile.png")


# ---------------------------------------------------------------------------
# 8. Best configuration summary table
# ---------------------------------------------------------------------------

def plot_best_config_table(df: pd.DataFrame) -> None:
    """Summary table: best model+loss+train_group for each (metric, test_set)."""
    col_headers = ["Metric"] + [TEST_LABELS[t] for t in TEST_ORDER]
    rows = []

    for metric, label, higher in METRICS:
        row = [label]
        for ts in TEST_ORDER:
            sub = df[df["test_set"] == ts]
            idx = sub[metric].idxmax() if higher else sub[metric].idxmin()
            best = sub.loc[idx]
            model = MODEL_DISPLAY_NAMES.get(best["model"], best["model"])
            loss  = LOSS_LABELS.get(best["loss"], best["loss"])
            train = TRAIN_LABELS.get(best["train_group"], best["train_group"])
            val   = best[metric]
            row.append(f"{model}\n{loss} | {train}\n{val:.4f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(15, 5), dpi=150)
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.1, 3.0)

    for c in range(len(col_headers)):
        tbl[(0, c)].set_facecolor("#2D3748")
        tbl[(0, c)].set_text_props(color="white", fontweight="bold")
    for r in range(1, len(rows) + 1):
        bg = "#F7FAFC" if r % 2 == 0 else "white"
        for c in range(len(col_headers)):
            tbl[(r, c)].set_facecolor(bg)

    fig.suptitle("Best Single Configuration per Metric and Test Set",
                 fontsize=13, fontweight="bold")
    _save(fig, "best_config_summary.png")

    # Plain CSV version
    summary = pd.DataFrame(rows, columns=col_headers)
    summary.to_csv(OUT_DIR / "best_config_summary.csv", index=False)
    print("  Saved: best_config_summary.csv")


# ---------------------------------------------------------------------------
# 9. Rank correlation across test sets (Spearman)
# ---------------------------------------------------------------------------

def plot_rank_correlation(df: pd.DataFrame) -> None:
    """For each pair of test sets, compute Spearman rank correlation of model
    mean accuracies.  High correlation → robust model selection signal."""
    ordered = [m for m in MODEL_ORDER if m in df["model"].values]
    test_mean_acc: dict[str, list[float]] = {}
    for ts in TEST_ORDER:
        test_mean_acc[ts] = [
            df.loc[(df["model"] == m) & (df["test_set"] == ts), "accuracy"].mean()
            for m in ordered
        ]

    n = len(TEST_ORDER)
    corr_matrix = np.zeros((n, n))
    for i, ts_i in enumerate(TEST_ORDER):
        for j, ts_j in enumerate(TEST_ORDER):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                rho, _ = spearmanr(test_mean_acc[ts_i], test_mean_acc[ts_j])
                corr_matrix[i, j] = rho

    test_labels = [TEST_LABELS[t] for t in TEST_ORDER]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(n), test_labels, fontsize=10)
    ax.set_yticks(range(n), test_labels, fontsize=10)
    ax.set_title(
        "Spearman Rank Correlation of Model Rankings\nacross Test Sets (by Accuracy)",
        fontsize=11, fontweight="bold",
    )
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    fig.tight_layout()
    _save(fig, "rank_correlation_across_testsets.png")


# ---------------------------------------------------------------------------
# 10. Combined-training delta: combined vs best single
# ---------------------------------------------------------------------------

def plot_combined_training_delta(df: pd.DataFrame) -> None:
    """Bar chart: combined-training mean minus best-single-training mean per
    (metric, test_set).  Positive = combined is better."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    x = np.arange(len(TEST_ORDER))
    width = 0.45

    for ax, (metric, label, _) in zip(axes, METRICS):
        deltas = []
        for ts in TEST_ORDER:
            combined_mean = df.loc[
                (df["train_group"] == "combined") & (df["test_set"] == ts), metric
            ].mean()
            single_best = max(
                df.loc[(df["train_group"] == tg) & (df["test_set"] == ts), metric].mean()
                for tg in ["cmose", "daisee"]
            ) if metric in HIGHER else min(
                df.loc[(df["train_group"] == tg) & (df["test_set"] == ts), metric].mean()
                for tg in ["cmose", "daisee"]
            )
            deltas.append(combined_mean - single_best)

        colors = ["#009E73" if d >= 0 else "#D55E00" for d in deltas]
        ax.bar(x, deltas, width=width, color=colors, alpha=0.85)
        ax.axhline(0, color="#333333", linewidth=1.0)
        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Δ {label}", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        for xi, d in zip(x, deltas):
            ax.annotate(f"{d:+.3f}", (xi, d), xytext=(xi, d + 0.002 * np.sign(d)),
                        ha="center", fontsize=8)

    fig.suptitle(
        "Combined Training vs Best Single-Dataset Training\n"
        "(Δ = combined mean − best(CMOSE,DaiSEE) mean; green = combined wins)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "combined_vs_single_training_delta.png")


# ---------------------------------------------------------------------------
# 11. Ordinal-loss benefit: does it improve ordinal metrics?
# ---------------------------------------------------------------------------

def plot_ordinal_loss_benefit(df: pd.DataFrame) -> None:
    """Scatter/bar comparing CE vs Ordinal on accuracy (nominal) and MAE+QWK
    (ordinal) to test whether ordinal loss selectively helps ordinal metrics."""
    compare_metrics = [
        ("accuracy",               "Accuracy",  True),
        ("mae",                    "MAE",        False),
        ("quadratic_weighted_kappa", "QWK",      True),
    ]
    pairs = [("ce", "ordinal")]
    ordered = [m for m in MODEL_ORDER if m in df["model"].values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    for ax, (metric, label, higher) in zip(axes, compare_metrics):
        ce_vals      = []
        ordinal_vals = []
        for model in ordered:
            ce_mean  = df.loc[(df["model"] == model) & (df["loss"] == "ce"),      metric].mean()
            ord_mean = df.loc[(df["model"] == model) & (df["loss"] == "ordinal"), metric].mean()
            ce_vals.append(ce_mean)
            ordinal_vals.append(ord_mean)

        model_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in ordered]
        x = np.arange(len(ordered))
        width = 0.35

        ax.bar(x - width / 2, ce_vals,      width, label="CE",      color="#0072B2", alpha=0.85)
        ax.bar(x + width / 2, ordinal_vals, width, label="Ordinal", color="#009E73", alpha=0.85)
        ax.set_xticks(x, model_labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "CE vs Ordinal Loss: Nominal vs Ordinal Metrics\n"
        "(averaged over 3 training groups × 3 test sets per model)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "ordinal_loss_benefit.png")


# ---------------------------------------------------------------------------
# 12. Modality comparison: OpenFace vs I3D vs Fusion
# ---------------------------------------------------------------------------

def plot_modality_comparison(df: pd.DataFrame) -> None:
    """Compare openface_mlp, i3d_mlp and fusion model across metrics and test
    sets to isolate the contribution of each feature modality."""
    modalities = {
        "openface_mlp":             "OpenFace MLP",
        "i3d_mlp":                  "I3D MLP",
        "openface_tcn_i3d_fusion":  "Fusion (TCN+I3D)",
    }
    colors = {
        "openface_mlp":            "#0072B2",
        "i3d_mlp":                 "#D55E00",
        "openface_tcn_i3d_fusion": "#E69F00",
    }
    x = np.arange(len(TEST_ORDER))
    width = 0.25

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    for ax, (metric, label, _) in zip(axes, METRICS):
        for i, (key, disp) in enumerate(modalities.items()):
            means = [
                df.loc[(df["model"] == key) & (df["test_set"] == ts), metric].mean()
                for ts in TEST_ORDER
            ]
            ax.bar(x + (i - 1) * width, means, width=width,
                   label=disp, color=colors[key], alpha=0.85)

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Modality Ablation: OpenFace vs I3D vs Fusion\n"
        "(averaged over 3 losses × 3 training groups)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "modality_comparison.png")


# ---------------------------------------------------------------------------
# 13. Temporal architecture benefit: MLP vs TCN / LSTM / Transformer
# ---------------------------------------------------------------------------

def plot_temporal_architecture_benefit(df: pd.DataFrame) -> None:
    """Compare openface_mlp (baseline) with TCN, LSTM, Transformer (all on
    OpenFace features) to show temporal modelling benefit."""
    openface_models = {
        "openface_mlp":  "MLP (baseline)",
        "temporal_cnn":  "TCN",
        "lstm":          "LSTM",
        "transformer":   "Transformer",
    }
    temporal_colors = {
        "openface_mlp": "#0072B2",
        "temporal_cnn": "#56B4E9",
        "lstm":         "#009E73",
        "transformer":  "#8BC34A",
    }
    x = np.arange(len(TEST_ORDER))
    width = 0.2

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), dpi=150)
    axes = axes.flatten()

    for ax, (metric, label, _) in zip(axes, METRICS):
        present = [k for k in openface_models if k in df["model"].values]
        for i, key in enumerate(present):
            disp  = openface_models[key]
            color = temporal_colors.get(key, "#4D4D4D")
            shift = (i - len(present) / 2 + 0.5) * width
            means = [
                df.loc[(df["model"] == key) & (df["test_set"] == ts), metric].mean()
                for ts in TEST_ORDER
            ]
            ax.bar(x + shift, means, width=width, label=disp, color=color, alpha=0.85)

        ax.set_xticks(x, [TEST_LABELS[t] for t in TEST_ORDER], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Temporal Architecture Benefit (OpenFace features only)\n"
        "(averaged over 3 losses × 3 training groups)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "temporal_architecture_benefit.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"Loaded {len(df)} rows.")
    print(f"  train_groups : {sorted(df['train_group'].unique())}")
    print(f"  test_sets    : {sorted(df['test_set'].unique())}")
    print(f"  models       : {sorted(df['model'].unique())}")
    print(f"  losses       : {sorted(df['loss'].unique())}")
    print()

    plot_cross_dataset_heatmaps(df)
    plot_training_group_comparison(df)
    plot_loss_effect(df)
    plot_model_comparison(df)
    plot_model_loss_interaction(df)
    plot_generalization_profile(df)
    plot_best_config_table(df)
    plot_rank_stability(df)
    plot_rank_correlation(df)
    plot_combined_training_delta(df)
    plot_ordinal_loss_benefit(df)
    plot_modality_comparison(df)
    plot_temporal_architecture_benefit(df)

    print(f"\nAll done — output in {OUT_DIR}")


if __name__ == "__main__":
    main()
