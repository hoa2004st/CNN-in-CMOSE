"""Compare cmose_baseline_paper against the best naive-model run and plot results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRICS_HIGHER_BETTER = ("accuracy", "macro_accuracy", "f1_macro", "f1_weighted")
METRICS_ALL = ("accuracy", "macro_accuracy", "f1_macro", "f1_weighted", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cmose_baseline_paper against best naive model from outputs/*/metrics.json.",
    )
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out_dir", default="outputs/comparisons/baseline_vs_best_naive")
    parser.add_argument(
        "--primary_metric",
        choices=METRICS_HIGHER_BETTER,
        default="f1_macro",
        help="Metric used to pick best naive run and best baseline run.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(outputs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs_dir.rglob("metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        m = payload.get("metrics", {})
        model = str(cfg.get("model", metrics_path.parent.parent.name))
        row = {
            "metrics_path": str(metrics_path),
            "run_name": str(metrics_path.parent.relative_to(outputs_dir)).replace("\\", "/"),
            "model": model,
            "loss": cfg.get("loss"),
            "accuracy": _safe_float(m.get("accuracy")),
            "macro_accuracy": _safe_float(m.get("macro_accuracy")),
            "f1_macro": _safe_float(m.get("f1_macro")),
            "f1_weighted": _safe_float(m.get("f1_weighted")),
            "mae": _safe_float(m.get("mae")),
        }
        rows.append(row)
    return rows


def rank_key(row: dict[str, Any], primary_metric: str) -> tuple[float, float, float, float]:
    return (
        float(row.get(primary_metric) or float("-inf")),
        float(row.get("f1_macro") or float("-inf")),
        float(row.get("macro_accuracy") or float("-inf")),
        float(row.get("accuracy") or float("-inf")),
    )


def plot_head_to_head(comp_df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = comp_df.melt(
        id_vars=["candidate"],
        value_vars=["accuracy", "macro_accuracy", "f1_macro", "f1_weighted"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="metric", y="value", hue="candidate", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("CMOSE Baseline vs Best Naive")
    fig.tight_layout()
    fig.savefig(out_dir / "head_to_head_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_naive_leaderboard(naive_df: pd.DataFrame, out_dir: Path, primary_metric: str) -> None:
    leaderboard = naive_df.sort_values(
        [primary_metric, "f1_macro", "macro_accuracy", "accuracy"],
        ascending=[False, False, False, False],
    ).head(10)
    if leaderboard.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=leaderboard, x=primary_metric, y="run_name", hue="model", dodge=False, ax=ax)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(primary_metric)
    ax.set_ylabel("Run")
    ax.set_title(f"Top Naive Runs by {primary_metric}")
    fig.tight_layout()
    fig.savefig(out_dir / "naive_top10.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    rows = load_rows(outputs_dir)
    if not rows:
        raise SystemExit(f"No metrics.json files found under: {outputs_dir}")

    df = pd.DataFrame.from_records(rows)
    baseline_df = df[df["model"] == "cmose_baseline_paper"].copy()
    naive_df = df[df["model"] != "cmose_baseline_paper"].copy()

    if baseline_df.empty:
        raise SystemExit("No cmose_baseline_paper metrics found.")
    if naive_df.empty:
        raise SystemExit("No naive-model metrics found.")

    best_baseline = max(baseline_df.to_dict("records"), key=lambda row: rank_key(row, args.primary_metric))
    best_naive = max(naive_df.to_dict("records"), key=lambda row: rank_key(row, args.primary_metric))

    comparison_rows = []
    for label, source in (("cmose_baseline_paper", best_baseline), ("best_naive", best_naive)):
        row = {"candidate": label}
        for metric in METRICS_ALL:
            row[metric] = source.get(metric)
        row["model"] = source.get("model")
        row["loss"] = source.get("loss")
        row["run_name"] = source.get("run_name")
        row["metrics_path"] = source.get("metrics_path")
        comparison_rows.append(row)

    comp_df = pd.DataFrame.from_records(comparison_rows)
    comp_df.to_csv(out_dir / "baseline_vs_best_naive.csv", index=False)
    naive_df.sort_values(
        [args.primary_metric, "f1_macro", "macro_accuracy", "accuracy"],
        ascending=[False, False, False, False],
    ).to_csv(out_dir / "naive_leaderboard.csv", index=False)

    delta: dict[str, float | None] = {}
    b_row = comparison_rows[0]
    n_row = comparison_rows[1]
    for metric in METRICS_ALL:
        b_val = b_row.get(metric)
        n_val = n_row.get(metric)
        if b_val is None or n_val is None:
            delta[metric] = None
        else:
            delta[metric] = float(b_val) - float(n_val)

    summary = {
        "primary_metric": args.primary_metric,
        "best_baseline": best_baseline,
        "best_naive": best_naive,
        "delta_baseline_minus_naive": delta,
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Baseline vs Best Naive",
        "",
        f"- Primary ranking metric: `{args.primary_metric}`",
        "",
        "## Best baseline run",
        f"- Run: `{best_baseline['run_name']}`",
        f"- File: `{best_baseline['metrics_path']}`",
    ]
    for metric in METRICS_ALL:
        md_lines.append(f"- {metric}: `{best_baseline.get(metric)}`")

    md_lines += [
        "",
        "## Best naive run",
        f"- Run: `{best_naive['run_name']}`",
        f"- Model: `{best_naive['model']}`",
        f"- Loss: `{best_naive.get('loss')}`",
        f"- File: `{best_naive['metrics_path']}`",
    ]
    for metric in METRICS_ALL:
        md_lines.append(f"- {metric}: `{best_naive.get(metric)}`")

    md_lines += ["", "## Delta (baseline - naive)"]
    for metric in METRICS_ALL:
        md_lines.append(f"- {metric}: `{delta.get(metric)}`")

    (out_dir / "comparison_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_head_to_head(comp_df, out_dir)
    plot_naive_leaderboard(naive_df, out_dir, args.primary_metric)

    print(f"Wrote comparison artifacts to: {out_dir}")


if __name__ == "__main__":
    main()

