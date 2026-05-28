import csv

rows = list(csv.DictReader(open("outputs/model_assessment/full_matrix.csv")))

groups   = ["cmose", "daisee", "combined"]
test_sets = ["cmose_test", "daisee_test", "private"]
models   = ["openface_mlp", "tcn", "lstm", "transformer", "i3d_mlp", "openface_tcn_i3d_fusion"]
losses   = ["ce", "weighted_ce", "ordinal"]

def fmt(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

for tg in groups:
    for ts in test_sets:
        subset = [r for r in rows if r["train_group"] == tg and r["test_set"] == ts]
        if not subset:
            continue
        n = subset[0]["n_labeled"]
        print(f"\n=== Train: {tg.upper()}  |  Test: {ts.upper()}  (n={n}) ===")
        header = f"  {'Model':<28} {'Loss':<14} {'Acc':>7} {'MacAcc':>7} {'MAE':>7} {'MacMAE':>7} {'Kappa':>7} {'QWK':>7}"
        sep    = f"  {'-'*28} {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}"
        print(header)
        print(sep)
        for m in models:
            for l in losses:
                r = next((x for x in subset if x["model"] == m and x["loss"] == l), None)
                if r:
                    print(
                        f"  {m:<28} {l:<14}"
                        f" {fmt(r['accuracy']):>7}"
                        f" {fmt(r['macro_accuracy']):>7}"
                        f" {fmt(r['mae']):>7}"
                        f" {fmt(r['macro_mae']):>7}"
                        f" {fmt(r['cohen_kappa']):>7}"
                        f" {fmt(r['quadratic_weighted_kappa']):>7}"
                    )
