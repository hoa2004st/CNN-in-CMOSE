"""Visualize best configs per group and cross-group comparison.

Groups:
  - naive                   : naive_matrix.csv (all models × losses × train_groups)
  - openface_temporal_hybrid: hybrid_matrix.csv filtered to model_type == 'openface_temporal_hybrid'
  - openface_temporal_i3d_hybrid: hybrid_matrix.csv filtered to model_type == 'openface_temporal_i3d_hybrid'

Outputs (all in outputs/model_assessment/comparison/):
  best_configs_tables.png         — per-group table: best config for each metric × test_set
  group_best_comparison.png       — grouped bars: best score of each group per metric × test_set
  improvement_delta_heatmap.png   — (6 metrics × 3 test_sets) delta heatmap for each hybrid vs naive
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NAIVE_CSV  = REPO_ROOT / "outputs" / "model_assessment" / "naive"   / "naive_matrix.csv"
HYBRID_CSV = REPO_ROOT / "outputs" / "model_assessment" / "hybrid"  / "hybrid_matrix.csv"
OUT_DIR    = REPO_ROOT / "outputs" / "model_assessment" / "comparison"

METRICS = [
    ("accuracy",                "Accuracy",     True),
    ("macro_accuracy",          "Macro Acc",    True),
    ("mae",                     "MAE",          False),
    ("macro_mae",               "Macro MAE",    False),
    ("cohen_kappa",             "Cohen κ",      True),
    ("quadratic_weighted_kappa","QWK",          True),
]
HIGHER = {m for m, _, h in METRICS if h}

TEST_SETS   = ["cmose_test", "daisee_test", "private"]
TEST_LABELS = {"cmose_test": "CMOSE Test", "daisee_test": "DaiSEE Test", "private": "Private"}

GROUP_KEYS = [
    "naive",
    "openface_temporal_hybrid",
    "openface_temporal_i3d_hybrid",
]
GROUP_SHORT = {
    "naive":                        "Naive",
    "openface_temporal_hybrid":     "OF-Hybrid",
    "openface_temporal_i3d_hybrid": "OF+I3D-Hybrid",
}
GROUP_COLOR = {
    "naive":                        "#0072B2",
    "openface_temporal_hybrid":     "#D55E00",
    "openface_temporal_i3d_hybrid": "#009E73",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    naive  = pd.read_csv(NAIVE_CSV)
    hybrid = pd.read_csv(HYBRID_CSV)
    return naive, hybrid


def _best_row(df: pd.DataFrame, test_set: str, metric: str,
              model_type: str | None, higher: bool) -> pd.Series | None:
    sub = df[df["test_set"] == test_set].copy()
    if model_type is not None:
        sub = sub[sub["model_type"] == model_type]
    sub = sub[sub[metric].notna()]
    if sub.empty:
        return None
    idx = sub[metric].idxmax() if higher else sub[metric].idxmin()
    return sub.loc[idx]


def _config_naive(row: pd.Series) -> str:
    return f"{row['train_group']} / {row['model']} / {row['loss']}"


def _config_hybrid(row: pd.Series) -> str:
    return f"{row['train_group']} / {row.get('arch_key', '?')}"


def build_best_table(naive: pd.DataFrame, hybrid: pd.DataFrame) -> dict:
    """
    Returns nested dict: best[group_key][test_set][metric] = (value, config_str)
    """
    group_src = {
        "naive":                        (naive,  None),
        "openface_temporal_hybrid":     (hybrid, "openface_temporal_hybrid"),
        "openface_temporal_i3d_hybrid": (hybrid, "openface_temporal_i3d_hybrid"),
    }
    best: dict = {g: {} for g in GROUP_KEYS}

    for group_key, (df, model_type) in group_src.items():
        for ts in TEST_SETS:
            best[group_key][ts] = {}
            for metric, _, higher in METRICS:
                row = _best_row(df, ts, metric, model_type, higher)
                if row is None:
                    best[group_key][ts][metric] = (float("nan"), "—")
                else:
                    cfg = _config_naive(row) if group_key == "naive" else _config_hybrid(row)
                    best[group_key][ts][metric] = (float(row[metric]), cfg)

    return best


# ---------------------------------------------------------------------------
# Figure 1 — per-group best-config tables
# ---------------------------------------------------------------------------

def plot_best_configs_tables(best: dict) -> None:
    """Three stacked matplotlib tables, one per group."""
    metric_labels = [lbl for _, lbl, _ in METRICS]
    test_labels   = [TEST_LABELS[ts] for ts in TEST_SETS]

    fig, axes = plt.subplots(3, 1, figsize=(18, 24), dpi=120)
    fig.subplots_adjust(hspace=0.35)

    for ax, group_key in zip(axes, GROUP_KEYS):
        group_data = best[group_key]

        cell_texts = []
        for metric, _, _ in METRICS:
            row = []
            for ts in TEST_SETS:
                val, cfg = group_data[ts][metric]
                if np.isnan(val):
                    row.append("—")
                else:
                    row.append(f"{val:.4f}\n{cfg}")
            cell_texts.append(row)

        ax.axis("off")
        tbl = ax.table(
            cellText=cell_texts,
            rowLabels=metric_labels,
            colLabels=test_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.35, 3.8)

        color = GROUP_COLOR[group_key]
        for c in range(len(test_labels)):
            tbl[(0, c)].set_facecolor(color)
            tbl[(0, c)].set_text_props(color="white", fontweight="bold")

        for r in range(1, len(METRICS) + 1):
            if (r, -1) in tbl._cells:
                tbl[(r, -1)].set_facecolor("#DDEEFF")
                tbl[(r, -1)].set_text_props(fontweight="bold")
            for c in range(len(test_labels)):
                tbl[(r, c)].set_facecolor("#F8F8F8" if r % 2 == 0 else "white")

        ax.set_title(
            f"{GROUP_SHORT[group_key]} — Best Config per Metric × Test Set\n"
            f"(cell: best value  ↵  train_group / model_or_arch_key / loss)",
            fontsize=11, fontweight="bold", pad=8, color=color,
        )

    fig.suptitle("Group-wise Best Configurations  (6 metrics × 3 test sets)",
                 fontsize=14, fontweight="bold", y=1.005)
    path = OUT_DIR / "best_configs_tables.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 2 — grouped bar comparison of best scores per metric
# ---------------------------------------------------------------------------

def plot_group_best_comparison(best: dict) -> None:
    """2×3 panel: one subplot per metric, 3 grouped bars per test_set (one bar per group)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=120)
    axes = axes.flatten()

    x     = np.arange(len(TEST_SETS))
    width = 0.24

    for ax, (metric, label, higher) in zip(axes, METRICS):
        for i, group_key in enumerate(GROUP_KEYS):
            vals = [best[group_key][ts][metric][0] for ts in TEST_SETS]
            bars = ax.bar(
                x + (i - 1) * width, vals,
                width  = width,
                label  = GROUP_SHORT[group_key],
                color  = GROUP_COLOR[group_key],
                alpha  = 0.88,
            )
            for rect, v in zip(bars, vals):
                if not np.isnan(v):
                    ypos = rect.get_height() + 0.003
                    ax.text(rect.get_x() + rect.get_width() / 2, ypos,
                            f"{v:.3f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90)

        ax.set_xticks(x, [TEST_LABELS[ts] for ts in TEST_SETS], fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylabel(label + ("  ↑ higher=better" if higher else "  ↓ lower=better"), fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        if metric in {"cohen_kappa", "quadratic_weighted_kappa"}:
            ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")

    fig.suptitle(
        "Best Score per Group × Metric × Test Set\n"
        "(each bar = single best row for that group on that metric+test_set)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "group_best_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 3 — improvement delta heatmaps (hybrid − naive)
# ---------------------------------------------------------------------------

def plot_improvement_delta(best: dict) -> None:
    """Two side-by-side heatmaps: (OF-Hybrid − Naive) and (OF+I3D-Hybrid − Naive).
    Each heatmap: rows = 6 metrics, cols = 3 test sets.
    Sign convention: positive (green) always means hybrid wins."""
    hybrid_groups = [
        ("openface_temporal_hybrid",     "OF-Hybrid − Naive"),
        ("openface_temporal_i3d_hybrid", "OF+I3D-Hybrid − Naive"),
    ]

    metric_labels = [lbl for _, lbl, _ in METRICS]
    test_labels   = [TEST_LABELS[ts] for ts in TEST_SETS]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

    for ax, (hybrid_key, title) in zip(axes, hybrid_groups):
        delta = np.full((len(METRICS), len(TEST_SETS)), float("nan"))

        for mi, (metric, _, higher) in enumerate(METRICS):
            for ti, ts in enumerate(TEST_SETS):
                naive_val  = best["naive"][ts][metric][0]
                hybrid_val = best[hybrid_key][ts][metric][0]
                if np.isnan(naive_val) or np.isnan(hybrid_val):
                    continue
                if higher:
                    delta[mi, ti] = hybrid_val - naive_val
                else:
                    # Lower is better → positive delta means hybrid has lower (better) value
                    delta[mi, ti] = naive_val - hybrid_val

        vabs = max(abs(np.nanmin(delta)), abs(np.nanmax(delta)), 1e-6)
        im = ax.imshow(delta, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")

        ax.set_xticks(range(len(TEST_SETS)), test_labels, fontsize=10)
        ax.set_yticks(range(len(METRICS)), metric_labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        for mi in range(len(METRICS)):
            for ti in range(len(TEST_SETS)):
                v = delta[mi, ti]
                if not np.isnan(v):
                    sign = "+" if v >= 0 else ""
                    norm_v = abs(v) / vabs
                    txt_color = "white" if norm_v > 0.55 else "black"
                    ax.text(ti, mi, f"{sign}{v:.4f}",
                            ha="center", va="center",
                            fontsize=9.5, fontweight="bold", color=txt_color)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Δ (+ = hybrid better)", fontsize=8)

    fig.suptitle(
        "Improvement Delta: Hybrid Best vs Naive Best\n"
        "Green = hybrid wins  |  Red = naive wins  "
        "|  MAE/Macro-MAE sign flipped (↓ is better)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "improvement_delta_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Figure 4 — combined summary heatmap (best scores, all groups × all metrics × test sets)
# ---------------------------------------------------------------------------

def plot_combined_score_heatmap(best: dict) -> None:
    """One heatmap per test_set (3 panels).
    Rows = 6 metrics, columns = 3 groups.
    Cell text = best score value.
    Color indicates relative magnitude within each metric row."""
    metric_labels = [lbl for _, lbl, _ in METRICS]
    group_labels  = [GROUP_SHORT[g] for g in GROUP_KEYS]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=120)

    for ax, ts in zip(axes, TEST_SETS):
        matrix = np.zeros((len(METRICS), len(GROUP_KEYS)))
        win_mask = np.zeros((len(METRICS), len(GROUP_KEYS)), dtype=bool)

        for mi, (metric, _, higher) in enumerate(METRICS):
            row_vals = [best[g][ts][metric][0] for g in GROUP_KEYS]
            for gi, v in enumerate(row_vals):
                matrix[mi, gi] = v if not np.isnan(v) else 0.0
            valid_vals = [v for v in row_vals if not np.isnan(v)]
            if valid_vals:
                best_val = max(valid_vals) if higher else min(valid_vals)
                for gi, v in enumerate(row_vals):
                    if not np.isnan(v) and abs(v - best_val) < 1e-9:
                        win_mask[mi, gi] = True

        # Normalize each row separately for color
        norm_matrix = np.zeros_like(matrix)
        for mi, (metric, _, higher) in enumerate(METRICS):
            row = matrix[mi]
            rmin, rmax = row.min(), row.max()
            if rmax > rmin:
                norm_matrix[mi] = (row - rmin) / (rmax - rmin)
                if not higher:
                    norm_matrix[mi] = 1 - norm_matrix[mi]
            else:
                norm_matrix[mi] = 0.5

        cmap = plt.cm.RdYlGn
        rgba = cmap(norm_matrix)
        ax.imshow(rgba, aspect="auto")

        for mi in range(len(METRICS)):
            _, _, higher = METRICS[mi]
            for gi in range(len(GROUP_KEYS)):
                v = matrix[mi, gi]
                norm = norm_matrix[mi, gi]
                txt_color = "white" if norm > 0.65 or norm < 0.25 else "black"
                star = " ★" if win_mask[mi, gi] else ""
                ax.text(gi, mi, f"{v:.4f}{star}",
                        ha="center", va="center",
                        fontsize=8.5, fontweight="bold" if win_mask[mi, gi] else "normal",
                        color=txt_color)

        ax.set_xticks(range(len(GROUP_KEYS)), group_labels, fontsize=9)
        ax.set_yticks(range(len(METRICS)), metric_labels, fontsize=9)
        ax.set_title(TEST_LABELS[ts], fontsize=12, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Best Score per Group × Metric (★ = winner for that row)\n"
        "Color normalized per metric row — green=best, red=worst",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = OUT_DIR / "combined_score_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def save_summary_csv(best: dict) -> None:
    rows = []
    for group_key in GROUP_KEYS:
        for ts in TEST_SETS:
            for metric, label, higher in METRICS:
                val, cfg = best[group_key][ts][metric]
                rows.append({
                    "group":     GROUP_SHORT[group_key],
                    "test_set":  ts,
                    "metric":    metric,
                    "value":     val,
                    "config":    cfg,
                    "higher_is_better": higher,
                })
    df = pd.DataFrame(rows)
    path = OUT_DIR / "best_configs_summary.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    naive, hybrid = load()
    print(f"  naive : {len(naive)} rows | hybrid : {len(hybrid)} rows")

    print("Computing best configs …")
    best = build_best_table(naive, hybrid)

    print("Plotting …")
    plot_best_configs_tables(best)
    plot_group_best_comparison(best)
    plot_improvement_delta(best)
    plot_combined_score_heatmap(best)
    save_summary_csv(best)

    print(f"\nDone — outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
