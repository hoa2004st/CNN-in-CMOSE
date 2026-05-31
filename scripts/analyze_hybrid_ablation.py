"""Analyze the 2^N hybrid ablation results and compare best hybrid vs best naive model.

Loads all metrics.json files from the ablation run directory, ranks by 6 metrics,
finds the best architecture assignment, then compares it to the best naive model.

Usage:
    python scripts/analyze_hybrid_ablation.py
    python scripts/analyze_hybrid_ablation.py --ablation_root outputs/training_log/combined/semantic_group_fusion
    python scripts/analyze_hybrid_ablation.py --naive_metrics outputs/training_log/combined/transformer/cross_entropy/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS_6 = ["accuracy", "macro_accuracy", "mae", "macro_mae", "cohen_kappa", "quadratic_weighted_kappa"]
PRIMARY_METRIC = "quadratic_weighted_kappa"

HIGHER_IS_BETTER = {"accuracy", "macro_accuracy", "cohen_kappa", "quadratic_weighted_kappa"}


def load_ablation_results(ablation_root: Path) -> list[dict]:
    results = []
    for metrics_path in sorted(ablation_root.glob("*/metrics.json")):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  WARNING: could not read {metrics_path}: {exc}")
            continue
        arch_key = metrics_path.parent.name
        row = {
            "arch_key": arch_key,
            "output_dir": str(metrics_path.parent),
            "group_architectures": data.get("config", {}).get("group_architectures"),
        }
        m = data.get("metrics", {})
        for metric in METRICS_6:
            row[metric] = m.get(metric, float("nan"))
        row[PRIMARY_METRIC] = m.get(PRIMARY_METRIC, float("nan"))
        results.append(row)
    return results


def rank_results(results: list[dict], metric: str) -> list[dict]:
    reverse = metric in HIGHER_IS_BETTER
    return sorted(results, key=lambda r: r.get(metric, float("nan")), reverse=reverse)


def print_table(results: list[dict], title: str, top_n: int = 10) -> None:
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    header = f"{'Rank':<5} {'Arch Key':<30} {'QWK':>7} {'Acc':>7} {'MacAcc':>7} {'MAE':>7} {'MacMAE':>7} {'CK':>7} {'QWK2':>7}"
    print(header)
    print("-" * 80)
    for rank, row in enumerate(results[:top_n], start=1):
        print(
            f"{rank:<5} {row['arch_key']:<30} "
            f"{row.get('quadratic_weighted_kappa', float('nan')):>7.4f} "
            f"{row['accuracy']:>7.4f} "
            f"{row['macro_accuracy']:>7.4f} "
            f"{row['mae']:>7.4f} "
            f"{row.get('macro_mae', float('nan')):>7.4f} "
            f"{row.get('cohen_kappa', float('nan')):>7.4f} "
            f"{row.get('quadratic_weighted_kappa', float('nan')):>7.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze hybrid ablation results and compare to best naive model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ablation_root",
        default="outputs/training_log/combined/semantic_group_fusion",
        help="Directory containing one subfolder per arch_key, each with metrics.json.",
    )
    parser.add_argument(
        "--naive_metrics",
        default="outputs/training_log/combined/transformer/cross_entropy/metrics.json",
        help="Path to the best naive model metrics.json for comparison.",
    )
    parser.add_argument(
        "--output",
        default="outputs/model_assessment/hybrid_vs_naive_comparison.json",
        help="Where to save the comparison JSON.",
    )
    parser.add_argument("--top_n", type=int, default=10, help="How many top variants to print.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ablation_root = Path(args.ablation_root)
    naive_path = Path(args.naive_metrics)
    output_path = Path(args.output)

    print(f"Loading ablation results from: {ablation_root}")
    results = load_ablation_results(ablation_root)
    if not results:
        print("No metrics.json files found. Run the ablation first.")
        return
    print(f"Loaded {len(results)} ablation variants.")

    ranked = rank_results(results, PRIMARY_METRIC)
    best = ranked[0]

    print_table(ranked, f"Top {args.top_n} hybrid variants ranked by {PRIMARY_METRIC}", top_n=args.top_n)

    # Per-metric winners
    print(f"\n{'='*80}")
    print("  Best variant per metric")
    print(f"{'='*80}")
    for metric in [PRIMARY_METRIC] + METRICS_6:
        best_for_metric = rank_results(results, metric)[0]
        direction = "^" if metric in HIGHER_IS_BETTER else "v"
        val = best_for_metric.get(metric, float("nan"))
        print(f"  {metric:<30} {direction}  {val:>8.4f}  ->  {best_for_metric['arch_key']}")

    # Naive model comparison
    naive_row: dict = {}
    if naive_path.exists():
        naive_data = json.loads(naive_path.read_text(encoding="utf-8"))
        naive_m = naive_data.get("metrics", {})
        naive_row = {
            "arch_key": naive_data.get("config", {}).get("model", "naive"),
            "group_architectures": None,
        }
        for metric in METRICS_6:
            naive_row[metric] = naive_m.get(metric, float("nan"))
        naive_row[PRIMARY_METRIC] = naive_m.get(PRIMARY_METRIC, float("nan"))

        print(f"\n{'='*80}")
        print("  Best Hybrid vs Best Naive Model")
        print(f"{'='*80}")
        header2 = f"{'Model':<32} {'QWK':>7} {'Acc':>7} {'MacAcc':>7} {'MAE':>7} {'MacMAE':>7} {'CK':>7}"
        print(header2)
        print("-" * 80)
        for row in [best, naive_row]:
            label = f"Hybrid ({row['arch_key']})" if row.get("group_architectures") else f"Naive ({row['arch_key']})"
            print(
                f"{label:<32} "
                f"{row.get('quadratic_weighted_kappa', float('nan')):>7.4f} "
                f"{row['accuracy']:>7.4f} "
                f"{row['macro_accuracy']:>7.4f} "
                f"{row['mae']:>7.4f} "
                f"{row.get('macro_mae', float('nan')):>7.4f} "
                f"{row.get('cohen_kappa', float('nan')):>7.4f}"
            )

        # Delta row
        print("-" * 80)
        delta_str = f"{'Delta (hybrid - naive)':<32}"
        for metric in METRICS_6:
            bv = best.get(metric, float("nan"))
            nv = naive_row.get(metric, float("nan"))
            delta = bv - nv
            sign = "+" if delta >= 0 else ""
            delta_str += f" {sign}{delta:>6.4f}"
        print(delta_str)
    else:
        print(f"\nNaive metrics not found at {naive_path}. Skipping comparison.")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = {
        "ablation_root": str(ablation_root),
        "total_variants": len(results),
        "best_hybrid": {
            "arch_key": best["arch_key"],
            "group_architectures": best.get("group_architectures"),
            "metrics": {m: best.get(m) for m in [PRIMARY_METRIC] + METRICS_6},
        },
        "naive": {
            "arch_key": naive_row.get("arch_key"),
            "metrics_path": str(naive_path),
            "metrics": {m: naive_row.get(m) for m in [PRIMARY_METRIC] + METRICS_6},
        } if naive_row else None,
        "all_variants_ranked": [
            {
                "rank": i + 1,
                "arch_key": r["arch_key"],
                "group_architectures": r.get("group_architectures"),
                "metrics": {m: r.get(m) for m in [PRIMARY_METRIC] + METRICS_6},
            }
            for i, r in enumerate(ranked)
        ],
    }
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nSaved comparison to {output_path}")


if __name__ == "__main__":
    main()
