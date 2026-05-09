"""Plot each private clip on a confidence-agreement scatter diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize each private clip as a point in agreement-confidence space.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_csv",
        default="outputs/domain_shift_analysis/model_agreement_per_clip.csv",
        help="Per-clip agreement table.",
    )
    parser.add_argument(
        "--output_png",
        default="outputs/domain_shift_analysis/confidence_agreement_scatter.png",
        help="Output scatter image.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input_csv)
    output_path = Path(args.output_png)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {"clip_id", "agreement_rate", "mean_confidence", "majority_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    color_map = {
        "Highly Disengage": "#1f77b4",
        "Disengage": "#ff7f0e",
        "Engage": "#9467bd",
        "Highly Engage": "#bdbdbd",
    }
    fallback_color = "#444444"

    fig, ax = plt.subplots(figsize=(9.2, 7.2), constrained_layout=True)

    # Main scatter plot
    ax.set_title("Private Clips in Confidence-Agreement Space")
    ax.set_xlabel("Agreement rate (across 6 models)")
    ax.set_ylabel("Mean confidence (across 6 models)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.28)

    # Overlay: agreement-rate count bars in the lowest quarter of scatter.
    bar_ax = ax.inset_axes([0.0, 0.0, 1.0, 0.25], transform=ax.transAxes)
    counts_by_rate = df["agreement_rate"].round(12).value_counts().sort_index()
    x_values = counts_by_rate.index.to_numpy(dtype=float)
    count_values = counts_by_rate.to_numpy(dtype=int)
    if len(x_values) > 1:
        min_spacing = min(
            current - previous
            for previous, current in zip(x_values[:-1], x_values[1:])
            if current > previous
        )
        bar_width = min_spacing * 0.72
    else:
        bar_width = 0.01
    bar_ax.bar(x_values, count_values, width=bar_width, color="#2E86AB", edgecolor="none", alpha=0.62)
    bar_ax.set_xlim(-0.02, 1.02)
    bar_ax.set_facecolor("none")
    bar_ax.set_axis_off()

    for label, group in df.groupby("majority_label"):
        color = color_map.get(label, fallback_color)
        ax.scatter(
            group["agreement_rate"],
            group["mean_confidence"],
            s=28,
            alpha=0.72,
            c=color,
            edgecolors="none",
            label=label,
        )
    ax.legend(title="Majority label", loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"Saved scatter plot: {output_path}")


if __name__ == "__main__":
    main()
