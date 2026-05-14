"""Visualize cross-model agreement results for private-domain predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.visualization.style import (
    HEATMAP_DIVERGING_CMAP,
    HISTOGRAM_COLOR,
    LOW_AGREEMENT_COLOR,
    REFERENCE_LINE_COLOR,
    SUMMARY_BAR_COLOR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate agreement metric visualizations from CSV outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="outputs/domain_shift_analysis")
    return parser


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_clip_metric_hist(per_clip: pd.DataFrame, metric: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    values = per_clip[metric].to_numpy(dtype=float)
    ax.hist(values, bins=20, color=HISTOGRAM_COLOR, edgecolor="black", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Number of clips")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_pairwise_kappa_heatmap(pairwise: pd.DataFrame, out_path: Path) -> None:
    runs = sorted(set(pairwise["run_a"]).union(set(pairwise["run_b"])))
    matrix = pd.DataFrame(np.nan, index=runs, columns=runs, dtype=float)
    np.fill_diagonal(matrix.values, 1.0)
    for row in pairwise.itertuples(index=False):
        matrix.loc[row.run_a, row.run_b] = row.cohens_kappa
        matrix.loc[row.run_b, row.run_a] = row.cohens_kappa

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix.values, cmap=HEATMAP_DIVERGING_CMAP, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_title("Pairwise Cohen's Kappa Across Models")
    ax.set_xticks(np.arange(len(runs)), labels=runs, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(runs)), labels=runs)
    for i in range(len(runs)):
        for j in range(len(runs)):
            value = matrix.values[i, j]
            text = "nan" if np.isnan(value) else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, out_path)


def plot_pairwise_kappa_bar(pairwise: pd.DataFrame, fleiss_kappa: float, out_path: Path) -> None:
    ordered = pairwise.sort_values("cohens_kappa").copy()
    labels = ordered["run_a"] + " vs " + ordered["run_b"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, ordered["cohens_kappa"], color=LOW_AGREEMENT_COLOR, edgecolor="black", alpha=0.9)
    ax.axhline(fleiss_kappa, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.5, label="Fleiss' kappa")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Pairwise Cohen's Kappa with Fleiss' Kappa Reference")
    ax.set_ylabel("Kappa")
    ax.set_xlabel("Model pair")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out_path)


def plot_summary_metrics(summary: dict, out_path: Path) -> None:
    metrics = [
        ("Mean agreement rate", float(summary["agreement_rate_mean"])),
        ("Mean prediction entropy", float(summary["prediction_entropy_mean"])),
        ("Mean confidence", float(summary["mean_confidence"])),
        ("Fleiss' kappa", float(summary["fleiss_kappa"])),
        ("Mean pairwise Cohen's kappa", float(summary["pairwise_cohens_kappa_mean"])),
    ]
    labels = [item[0] for item in metrics]
    values = [item[1] for item in metrics]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(labels, values, color=SUMMARY_BAR_COLOR, edgecolor="black")
    ax.set_title("Agreement Metric Summary")
    ax.set_xlabel("Value")
    ax.grid(axis="x", alpha=0.3)
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:.3f}", va="center", ha="left", fontsize=9)
    _save(fig, out_path)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    per_clip_path = output_dir / "model_agreement_per_clip.csv"
    pairwise_path = output_dir / "pairwise_cohens_kappa.csv"
    summary_path = output_dir / "agreement_summary.json"

    if not per_clip_path.exists() or not pairwise_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Missing required files. Expected: "
            f"{per_clip_path}, {pairwise_path}, {summary_path}"
        )

    per_clip = pd.read_csv(per_clip_path)
    pairwise = pd.read_csv(pairwise_path)
    summary = pd.read_json(summary_path, typ="series").to_dict()

    plot_clip_metric_hist(
        per_clip,
        metric="agreement_rate",
        title="Clip-Level Agreement Rate Distribution",
        out_path=output_dir / "agreement_rate_distribution.png",
    )
    plot_clip_metric_hist(
        per_clip,
        metric="prediction_entropy",
        title="Clip-Level Prediction Entropy Distribution",
        out_path=output_dir / "prediction_entropy_distribution.png",
    )
    plot_clip_metric_hist(
        per_clip,
        metric="mean_confidence",
        title="Clip-Level Mean Confidence Distribution",
        out_path=output_dir / "mean_confidence_distribution.png",
    )
    plot_pairwise_kappa_heatmap(pairwise, output_dir / "pairwise_cohens_kappa_heatmap.png")
    plot_pairwise_kappa_bar(
        pairwise,
        fleiss_kappa=float(summary["fleiss_kappa"]),
        out_path=output_dir / "pairwise_cohens_kappa_bar.png",
    )
    plot_summary_metrics(summary, output_dir / "agreement_metrics_summary.png")
    print(f"Saved agreement visualizations to {output_dir}")


if __name__ == "__main__":
    main()
