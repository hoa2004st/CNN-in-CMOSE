"""Compare best naive, best OpenFaceTemporalHybrid, and best OpenFaceTemporalI3DHybrid per test set.

For each test set finds the absolute best of each model family across all train groups.

Outputs:
  outputs/model_assessment/comparison/naive_vs_hybrid_comparison.csv

Usage:
    python scripts/compare_naive_vs_hybrid.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.output_paths import NAIVE_ASSESSMENT_DIR, HYBRID_ASSESSMENT_DIR, MODEL_COMPARISON_ASSESSMENT_DIR

METRIC_COLS = ["accuracy", "macro_accuracy", "mae", "macro_mae", "cohen_kappa", "quadratic_weighted_kappa"]
PRIMARY = "quadratic_weighted_kappa"
HIGHER_IS_BETTER = {"accuracy", "macro_accuracy", "cohen_kappa", "quadratic_weighted_kappa"}
TEST_SETS = ["cmose_test", "daisee_test", "private"]

MODEL_FAMILIES = [
    ("Naive",         None),                        # from naive CSV, all models
    ("OF-Hybrid",     "openface_temporal_hybrid"),   # OpenFace-only hybrid
    ("OF+I3D-Hybrid", "openface_temporal_i3d_hybrid"),  # OpenFace + I3D hybrid
]


def best_row(df: pd.DataFrame, test_set: str, model_type: str | None) -> pd.Series | None:
    sub = df[df["test_set"] == test_set].copy()
    if model_type is not None:
        col = "model_type" if "model_type" in sub.columns else "model"
        sub = sub[sub[col] == model_type]
    if sub.empty:
        return None
    valid = sub[sub[PRIMARY].notna()]
    if valid.empty:
        return None
    return valid.sort_values(PRIMARY, ascending=False).iloc[0]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--naive_csv",  default=str(NAIVE_ASSESSMENT_DIR  / "full_matrix.csv"))
    p.add_argument("--hybrid_csv", default=str(HYBRID_ASSESSMENT_DIR / "hybrid_matrix.csv"))
    p.add_argument("--output_csv", default=str(MODEL_COMPARISON_ASSESSMENT_DIR / "naive_vs_hybrid_comparison.csv"))
    return p


def _ident(row: pd.Series | None, family: str) -> str:
    if row is None:
        return "—"
    if family == "Naive":
        return f"{row.get('train_group','')}/{row.get('model','')}/{row.get('loss','')}"
    return f"{row.get('train_group','')}/{row.get('arch_key','?')}"


def main(argv=None):
    args = build_parser().parse_args(argv)
    naive_path  = Path(args.naive_csv)
    hybrid_path = Path(args.hybrid_csv)

    if not naive_path.exists():
        print(f"Naive CSV not found: {naive_path}"); return
    if not hybrid_path.exists():
        print(f"Hybrid CSV not found: {hybrid_path}"); return

    naive  = pd.read_csv(naive_path)
    hybrid = pd.read_csv(hybrid_path)
    naive  = naive[naive["model"] != "semantic_group_fusion"]

    # Combine into one searchable frame per family
    family_dfs = {
        "Naive":         naive,
        "OF-Hybrid":     hybrid,
        "OF+I3D-Hybrid": hybrid,
    }

    out_rows = []
    WIDTH = 105
    print(f"\n{'='*WIDTH}")
    print("  BEST NAIVE  vs  BEST OpenFaceTemporalHybrid  vs  BEST OpenFaceTemporalI3DHybrid")
    print("  Absolute best per test set across all train groups")
    print(f"{'='*WIDTH}")

    metric_header = " ".join(f"{m[:7]:>8}" for m in METRIC_COLS)
    print(f"\n  {'Test set':<14} {'Family':<16}  {metric_header}  Identifier")
    print(f"  {'-'*WIDTH}")

    for test_set in TEST_SETS:
        rows_this_ts: list[tuple[str, pd.Series | None]] = []
        for family, model_type in MODEL_FAMILIES:
            df = family_dfs[family]
            r = best_row(df, test_set, model_type)
            rows_this_ts.append((family, r))

        # Print each family
        for i, (family, r) in enumerate(rows_this_ts):
            ts_col = test_set if i == 0 else ""
            if r is None:
                val_str = " ".join(f"{'—':>8}" for _ in METRIC_COLS)
                print(f"  {ts_col:<14} {family:<16}  {val_str}  —")
            else:
                val_str = " ".join(f"{r.get(m, float('nan')):>8.4f}" for m in METRIC_COLS)
                print(f"  {ts_col:<14} {family:<16}  {val_str}  {_ident(r, family)}")

        # Delta rows vs Naive
        naive_r = rows_this_ts[0][1]
        for family, r in rows_this_ts[1:]:
            if naive_r is not None and r is not None:
                deltas = [r.get(m, float("nan")) - naive_r.get(m, float("nan")) for m in METRIC_COLS]
                wins = sum(
                    1 for m, d in zip(METRIC_COLS, deltas)
                    if (m in HIGHER_IS_BETTER and d > 0) or (m not in HIGHER_IS_BETTER and d < 0)
                )
                delta_str = " ".join(f"{('+' if d >= 0 else '')}{d:>7.4f}" for d in deltas)
                print(f"  {'':14} {'vs Naive Delta':<16}  {delta_str}  ({wins}/{len(METRIC_COLS)} wins for {family})")
        print()

        # Collect CSV row
        row: dict = {"test_set": test_set}
        for family, r in rows_this_ts:
            key = family.lower().replace("+", "_").replace("-", "_")
            if r is not None:
                row[f"{key}_train_group"] = r.get("train_group", "")
                row[f"{key}_model"]       = r.get("model", "")
                row[f"{key}_arch_key"]    = r.get("arch_key", "")
                row[f"{key}_loss"]        = r.get("loss", "")
                for m in METRIC_COLS:
                    row[f"{key}_{m}"] = r.get(m, float("nan"))
            else:
                for k in ["train_group", "model", "arch_key", "loss"]:
                    row[f"{key}_{k}"] = ""
                for m in METRIC_COLS:
                    row[f"{key}_{m}"] = float("nan")
        out_rows.append(row)

    result = pd.DataFrame(out_rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
