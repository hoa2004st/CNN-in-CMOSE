"""Thesis result tables emitted as Markdown previews into outputs/thesis/tables/.

Table file stems follow the LaTeX number they render as (``T<chapter>_<index>_<slug>``,
e.g. ``T5_2_i3d_fusion_effect`` is Table 5.2). Two specs in ``build_all`` are kept for
provenance but not currently ``\\input`` into the thesis, so they retain their legacy
stems (``T2_base_indomain``, ``T9_agreement_stats``).

Each table is built as a ``(name, frame, caption)`` spec (:func:`build_all`); :func:`make_all`
writes the Markdown preview. The LaTeX (``.tex``) rendering of these tables now lives with the
rest of the thesis-project assembly in :mod:`src.analysis.thesis_latex`, which consumes the same
specs — so the tables have a single source of truth and no ``.tex`` is written here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import aggregate as ag
from src.visualization.figbase import TABLE_DIR
from src.visualization.style import display_model_name

# All six reported metrics in display order: the single primary/selection metric (QWK)
# first, then the five secondary metrics. Only QWK ever selects a model; the rest are
# reported for analysis and comparability (see aggregate.SELECTION_METRIC).
_REPORTED_METRICS = ["quadratic_weighted_kappa", "macro_accuracy", "macro_mae", "accuracy", "mae", "cohen_kappa"]


def _fmt(v) -> str:
    if isinstance(v, float):
        if v == 0:  # collapse -0.0 / 0.0
            v = 0.0
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _to_markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(_fmt(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([head, sep, *body])


def _spec(frame: pd.DataFrame, name: str, caption: str) -> tuple[str, pd.DataFrame, str]:
    """A table spec: filename stem, data, caption. No I/O."""
    return name, frame, caption


def table_dataset_stats() -> tuple[str, pd.DataFrame, str]:
    overall = ag.load_class_distribution_overall()
    by_split = ag.load_class_distribution_by_split()
    totals = by_split.groupby(["dataset", "split"])["split_total"].first().unstack().fillna(0).astype(int)
    rows = []
    for dataset in ["CMOSE", "DAiSEE", "Combined"]:
        if dataset not in set(overall["dataset"]):
            continue
        sub = overall[overall["dataset"] == dataset].set_index("class_label")
        row = {
            "Dataset": dataset,
            "Total": int(sub["count"].sum()),
            "Train": int(totals.loc[dataset].get("train", 0)) if dataset in totals.index else 0,
            "Val": int(totals.loc[dataset].get("unlabel", 0)) if dataset in totals.index else 0,
            "Test": int(totals.loc[dataset].get("test", 0)) if dataset in totals.index else 0,
        }
        for cls, short in [("Highly Disengage", "HD%"), ("Disengage", "DE%"),
                           ("Engage", "EG%"), ("Highly Engage", "HE%")]:
            row[short] = round(float(sub.loc[cls, "proportion"]) * 100, 1) if cls in sub.index else 0.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    return _spec(frame, "T3_1_dataset_stats", "Dataset statistics and class balance (%).")


def table_base_indomain() -> tuple[str, pd.DataFrame, str]:
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")].copy()
    cell = cell.sort_values("quadratic_weighted_kappa", ascending=False)
    frame = cell[["model_display", "loss_display", *_REPORTED_METRICS]].rename(
        columns={"model_display": "Model", "loss_display": "Loss",
                 **{k: ag.METRIC_DISPLAY[k] for k in _REPORTED_METRICS}}
    )
    return _spec(frame, "T2_base_indomain",
                 "Base models × losses, in-domain (CMOSE→CMOSE), sorted by QWK.")


def table_hybrid_topk(k: int = 5) -> tuple[str, pd.DataFrame, str]:
    h = ag.load_hybrid_matrix()
    cell = h[(h["train_group"] == "cmose") & (h["test_set"] == "cmose_test")].copy()
    cell = cell.sort_values("quadratic_weighted_kappa", ascending=False).head(k)
    frame = cell[["variant", "arch_key", "quadratic_weighted_kappa", "macro_accuracy",
                  "macro_mae", "accuracy"]].rename(
        columns={"variant": "Variant", "arch_key": "Arch (gaze_eye_face_head_au)",
                 "quadratic_weighted_kappa": "QWK", "macro_accuracy": "Macro-Acc",
                 "macro_mae": "Macro-MAE", "accuracy": "Accuracy"})
    return _spec(frame, "T5_3_hybrid_topk",
                 f"Top-{k} hybrid configs in-domain (CMOSE), sorted by QWK.")


def table_i3d_fusion_effect() -> tuple[str, pd.DataFrame, str]:
    """Paired effect of adding the I3D stream, by evaluation regime.

    Each OpenFace-only hybrid is paired with its exact +I3D twin (same five per-group
    encoders, same train source, test set and loss), so the distribution of paired QWK
    differences isolates the I3D contribution free of the configuration as a confounder.
    The three regimes split by whether the test corpus is represented in training: ``seen
    target`` (the four train×test cells whose test corpus is in the training pool),
    ``cross-corpus`` (train on one public corpus, test on the other), and the unseen
    ``private`` set.
    """
    h = ag.load_hybrid_matrix()
    qwk = "quadratic_weighted_kappa"
    seen = ~ag.unseen_target_mask(h)
    cross = (((h["train_group"] == "cmose") & (h["test_set"] == "daisee_test"))
             | ((h["train_group"] == "daisee") & (h["test_set"] == "cmose_test")))
    private = h["test_set"] == "private"
    regimes = [
        ("Seen target", seen),
        ("Cross-corpus (public)", cross),
        ("Private (unseen)", private),
    ]
    rows = []
    for label, mask in regimes:
        piv = h[mask].pivot_table(
            index=["train_group", "test_set", "loss", "arch_key"],
            columns="has_i3d", values=qwk).dropna()
        delta = piv[True] - piv[False]
        rows.append({
            "Regime": label,
            "Pairs (n)": int(len(delta)),
            "Mean ΔQWK": f"{delta.mean():+.3f}",
            "Pairs improving (%)": int(round(float((delta > 0).mean()) * 100)),
        })
    frame = pd.DataFrame(rows)
    return _spec(frame, "T5_2_i3d_fusion_effect",
                 "Paired effect of adding the I3D stream on QWK, by evaluation regime "
                 "(each OpenFace-only hybrid against its exact +I3D twin).")


def _group_marginal_frame(cell: pd.DataFrame) -> pd.DataFrame:
    """Per-group marginal mean QWK per encoder token over the configs in ``cell``."""
    rows = []
    for g in ag.OPENFACE_GROUP_ORDER:
        means = {
            token: float(cell[cell[f"arch_{g}"] == token]["quadratic_weighted_kappa"].mean())
            for token in ag.ARCH_TOKENS
        }
        best_token = max(means, key=means.get)
        spread = max(means.values()) - min(means.values())
        row = {"Group": ag.GROUP_DISPLAY[g]}
        for token in ag.ARCH_TOKENS:
            row[f"Mean QWK ({ag.ARCH_TOKEN_DISPLAY[token]})"] = round(means[token], 3)
        row["Spread"] = round(spread, 3)
        row["Prefers"] = ag.ARCH_TOKEN_DISPLAY[best_token]
        rows.append(row)
    return pd.DataFrame(rows)


def table_group_marginal() -> tuple[str, pd.DataFrame, str]:
    h = ag.load_hybrid_matrix()
    cell = h[(h["train_group"] == "cmose") & (h["test_set"] == "cmose_test")]
    return _spec(_group_marginal_frame(cell), "T5_group_marginal",
                 "Per-group marginal effect of encoder choice on in-domain QWK.")


def table_group_marginal_combined() -> tuple[str, pd.DataFrame, str]:
    """T5 and T10 merged: the per-group marginal in-domain AND on the unseen-target cells.

    One table instead of two so the across-regime comparison (does the head-pose -> TCN
    preference survive shift?) is read in place.
    """
    h = ag.load_hybrid_matrix()
    regimes = [
        ("In-domain", h[(h["train_group"] == "cmose") & (h["test_set"] == "cmose_test")]),
        ("Unseen targets", h[ag.unseen_target_mask(h)]),
    ]
    rows = []
    for g in ag.OPENFACE_GROUP_ORDER:
        for regime, cell in regimes:
            means = {
                token: float(cell[cell[f"arch_{g}"] == token]["quadratic_weighted_kappa"].mean())
                for token in ag.ARCH_TOKENS
            }
            row = {"Group": ag.GROUP_DISPLAY[g], "Regime": regime}
            for token in ag.ARCH_TOKENS:
                row[ag.ARCH_TOKEN_DISPLAY[token]] = round(means[token], 3)
            row["Spread"] = round(max(means.values()) - min(means.values()), 3)
            row["Prefers"] = ag.ARCH_TOKEN_DISPLAY[max(means, key=means.get)]
            rows.append(row)
    frame = pd.DataFrame(rows)
    return _spec(frame, "T5_1_group_marginal_combined",
                 "Per-group marginal effect of encoder choice on mean QWK, in-domain "
                 "(CMOSE) and pooled over the unseen-target cells.")


def table_private_by_source() -> tuple[str, pd.DataFrame, str]:
    """Private-set (test-only) best base vs best hybrid per training source."""
    m = ag.load_matrix()
    h = ag.load_hybrid_matrix()
    rows = []
    for src in ["cmose", "daisee", "combined"]:
        cb = m[(m["train_group"] == src) & (m["test_set"] == "private")]
        ch = h[(h["train_group"] == src) & (h["test_set"] == "private")]
        bb = cb.loc[cb["quadratic_weighted_kappa"].idxmax()]
        bh = ch.loc[ch["quadratic_weighted_kappa"].idxmax()]
        rows.append({
            "Train source": ag.TRAIN_GROUP_DISPLAY[src],
            "Best base (model/loss)": f"{display_model_name(bb['model'])}/{bb['loss']}",
            "Base QWK": round(float(bb["quadratic_weighted_kappa"]), 3),
            "Base macro-MAE": round(float(bb["macro_mae"]), 3),
            "Best hybrid (arch)": f"{bh['variant']} {bh['arch_key']}",
            "Hybrid QWK": round(float(bh["quadratic_weighted_kappa"]), 3),
            "Hybrid macro-acc": round(float(bh["macro_accuracy"]), 3),
            "Hybrid macro-MAE": round(float(bh["macro_mae"]), 3),
        })
    frame = pd.DataFrame(rows)
    return _spec(frame, "T5_4_private_by_source",
                 "Private set (test-only): best base vs best hybrid by training source.")


def table_indomain_cmose_vs_daisee() -> tuple[str, pd.DataFrame, str]:
    """Best in-domain result per dataset (justifies CMOSE-centric ablation)."""
    m = ag.load_matrix()
    rows = []
    for tr, te, label in [("cmose", "cmose_test", "CMOSE"), ("daisee", "daisee_test", "DAiSEE")]:
        cell = m[(m["train_group"] == tr) & (m["test_set"] == te)]
        best = cell.loc[cell["quadratic_weighted_kappa"].idxmax()]
        rows.append({
            "In-domain dataset": label,
            "Best model": display_model_name(best["model"]),
            "Loss": best["loss_display"],
            "QWK": round(float(best["quadratic_weighted_kappa"]), 3),
            "Macro-Acc": round(float(best["macro_accuracy"]), 3),
            "Macro-MAE": round(float(best["macro_mae"]), 3),
            "Accuracy": round(float(best["accuracy"]), 3),
        })
    frame = pd.DataFrame(rows)
    return _spec(frame, "T4_2_indomain_datasets",
                 "Best in-domain result per dataset (CMOSE vs DAiSEE).")


def table_per_metric_winner() -> tuple[str, pd.DataFrame, str]:
    """Which (model, loss) each of the six metrics declares best, in-domain CMOSE.

    The disagreement between rows is the point: it motivates fixing the primary metrics
    before any architecture comparison.
    """
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")]
    rows = []
    for metric in _REPORTED_METRICS:
        best = cell.loc[cell[metric].idxmin() if metric in ag.LOWER_BETTER_METRICS
                        else cell[metric].idxmax()]
        rows.append({
            "Metric": ag.METRIC_DISPLAY[metric],
            "Best model": best["model_display"],
            "Loss": best["loss_display"],
            "Value": round(float(best[metric]), 3),
        })
    frame = pd.DataFrame(rows)
    return _spec(frame, "T4_1_per_metric_winner",
                 "The winning base configuration according to each metric "
                 "(in-domain CMOSE).")


def table_agreement_stats() -> tuple[str, pd.DataFrame, str]:
    """Prediction-agreement and ensemble-headroom statistics, in-domain CMOSE."""
    from src.analysis import agreement as agr  # local import: loads the 24 MB prediction log

    wide, true = agr.indomain_prediction_table()
    oracle = agr.oracle_stats(wide, true)
    blocks = agr.kappa_block_means(agr.pairwise_kappa(wide))
    of_vs_i3d = agr.pair_overlap(wide, true, ("temporal_cnn", "ce"), ("i3d_mlp", "ce"))

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    rows = [
        ("Configs compared / test clips", f"{oracle['n_configs']} / {oracle['n_clips']}"),
        ("Best single-config accuracy", pct(oracle["best_single_accuracy"])),
        ("Majority-vote accuracy (all configs)", pct(oracle["majority_vote_accuracy"])),
        ("Oracle accuracy (>=1 config correct)", pct(oracle["oracle_accuracy"])),
        ("Clips every config gets wrong", pct(oracle["all_wrong_share"])),
        ("Mean pairwise Cohen κ — same model, different loss", f"{blocks['same model']:.3f}"),
        ("Mean pairwise Cohen κ — same loss, different model", f"{blocks['same loss']:.3f}"),
        ("Mean pairwise Cohen κ — different model and loss",
         f"{blocks['different model & loss']:.3f}"),
        ("openface_tcn/CE vs i3d_mlp/CE — Cohen κ", f"{of_vs_i3d['kappa']:.3f}"),
        ("openface_tcn/CE vs i3d_mlp/CE — only openface_tcn correct", pct(of_vs_i3d["only_a"])),
        ("openface_tcn/CE vs i3d_mlp/CE — only i3d_mlp correct", pct(of_vs_i3d["only_b"])),
        ("openface_tcn/CE vs i3d_mlp/CE — both wrong", pct(of_vs_i3d["both_wrong"])),
        ("openface_tcn/CE + i3d_mlp/CE — pair oracle accuracy",
         pct(of_vs_i3d["pair_oracle_accuracy"])),
    ]
    frame = pd.DataFrame(rows, columns=["Statistic", "Value"])
    return _spec(frame, "T9_agreement_stats",
                 "Prediction-level agreement and ensemble headroom of the base "
                 "configurations (in-domain CMOSE).")


def table_openface_vs_i3d() -> tuple[str, pd.DataFrame, str]:
    """Best OpenFace encoder vs the I3D MLP on QWK and the balanced secondary metrics (in-domain CMOSE, CE).

    Replaces the former bar chart: two rows x three metrics (QWK plus the two balanced
    secondary metrics) read more honestly as a table. The contrast (complementary feature
    families) motivates fusing the two streams. The OpenFace winner is selected on QWK.
    """
    m = ag.load_matrix()
    cell = m[(m["train_group"] == "cmose") & (m["test_set"] == "cmose_test")
             & (m["loss"] == "ce")]
    of = cell[cell["model"].isin(["openface_mlp", "temporal_cnn", "lstm", "transformer"])]
    best_of = of.loc[of["quadratic_weighted_kappa"].idxmax()]
    i3d = cell[cell["model"] == "i3d_mlp"].iloc[0]
    rows = []
    for family, row in [("OpenFace (behaviour descriptors)", best_of),
                        ("I3D (appearance/motion)", i3d)]:
        rows.append({
            "Feature family": family,
            "Model": display_model_name(row["model"]),
            "QWK": round(float(row["quadratic_weighted_kappa"]), 3),
            "Macro-Acc": round(float(row["macro_accuracy"]), 3),
            "Macro-MAE": round(float(row["macro_mae"]), 3),
        })
    frame = pd.DataFrame(rows)
    return _spec(frame, "T4_3_openface_vs_i3d",
                 "Best OpenFace encoder (selected on QWK) vs the I3D MLP on QWK and the "
                 "balanced secondary metrics (in-domain CMOSE, cross-entropy).")


def table_group_marginal_unseen() -> tuple[str, pd.DataFrame, str]:
    """T5's marginal recomputed on the unseen-target cells (does the preference transfer?)."""
    h = ag.load_hybrid_matrix()
    cell = h[ag.unseen_target_mask(h)]
    return _spec(_group_marginal_frame(cell), "T10_group_marginal_unseen",
                 "Per-group marginal effect of encoder choice on QWK, pooled over the "
                 "unseen-target cells.")


def build_all() -> list[tuple[str, pd.DataFrame, str]]:
    """Every result table as a ``(name, frame, caption)`` spec (no I/O)."""
    return [
        table_dataset_stats(),
        table_base_indomain(),
        table_hybrid_topk(),
        table_i3d_fusion_effect(),
        table_group_marginal_combined(),
        table_private_by_source(),
        table_indomain_cmose_vs_daisee(),
        table_per_metric_winner(),
        table_agreement_stats(),
        table_openface_vs_i3d(),
    ]


def make_all(directory: Path | None = None) -> list[Path]:
    """Write the Markdown preview of every table into ``directory`` (default TABLE_DIR)."""
    directory = directory or TABLE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame, caption in build_all():
        md_path = directory / f"{name}.md"
        md_path.write_text(f"**{caption}**\n\n{_to_markdown(frame)}\n", encoding="utf-8")
        paths.append(md_path)
    return paths
