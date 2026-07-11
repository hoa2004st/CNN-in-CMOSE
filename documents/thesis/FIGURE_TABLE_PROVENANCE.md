# Thesis figure & table provenance

For every figure and table in the thesis: its **raw data**, the **analysis code** that
shapes that data, and the **visualization / table-writing code** that renders it.

## Pipeline overview

```
TRAINING / EVAL                RAW DATA (outputs/)              ANALYSIS LAYER             FIGURE / TABLE CODE
─────────────────              ──────────────────              ──────────────             ───────────────────
evaluate_full_matrix.py    →   model_assessment/naive/      ┐
evaluate_hybrid_full_      →   model_assessment/hybrid/     ├→ src/analysis/aggregate.py ┐→ src/visualization/figures_*.py
  matrix.py                    training_log/.../metrics.json │  src/analysis/agreement.py └→ src/analysis/tables.py
main.py (training)         →   dataset_analysis/*.csv       ┘                                + thesis_latex.py (→ .tex)
```

- **Master orchestrator:** `src/analysis/make_thesis_artifacts.py`
  (`python -m src.analysis.make_thesis_artifacts`) regenerates every data-driven figure + table.
- **Shared plot style / saving:** `src/visualization/figbase.py`, `src/visualization/style.py`.
- **LaTeX rendering:** `src/analysis/thesis_latex.py` (+ `src/analysis/latexfmt.py`) renders the
  table specs to `documents/thesis/Table/*.tex` and publishes figures to `documents/thesis/Figure/`.

Raw-data paths below are relative to `outputs/`.

---

## A. Schematic architecture diagrams (no data — hand-drawn)

These have **no raw data, no analysis code, and no plotting code**. They are drawn by hand from
Mermaid source and exported as PNG.

- **Source:** `documents/thesis/diagrams_mermaid.md` (layer dims taken from `src/models/models.py`)

| Figure no. | Figure (PNG) | Label | Chapter |
|---|---|---|---|
| 2.1 | `openface_tcn_block.png` | `fig:tcn_block` | Ch2 |
| 2.2 | `openface_lstm_cell.png` | `fig:lstm_cell` | Ch2 |
| 2.3 | `openface_transformer_encoder_layer.png` | `fig:transformer_encoder_layer` | Ch2 |
| 3.2 | `openface_mlp.png`, `openface_tcn.png`, `openface_lstm.png`, `openface_transformer.png`, `i3d_mlp.png` | `fig:baselines` | Ch3 |
| 3.3 | `hybrid.png` | `fig:hybrid` | Ch3 |
| 3.4 | `hybrid_tcn_encoder.png`, `hybrid_transformer_encoder.png`, `hybrid_lstm_encoder.png` | `fig:group_encoders` | Ch3 |

---

## B. Data-driven figures

All generated into `outputs/thesis/figures/` and auto-published to `documents/thesis/Figure/`.

| Figure no. | Figure (PNG) | Raw data | Analysis code | Plotting code |
|---|---|---|---|---|
| 3.1 | `dataset_class_distribution_overall.png` | `dataset_analysis/class_distribution_overall.csv` | `aggregate.load_class_distribution_overall` | `figures_dataset.fig_overall` |
| 4.1 | `metric_correlation_base.png` | `model_assessment/naive/full_matrix.csv` | `aggregate.load_matrix` → `agreement.metric_rank_correlation` (Kendall τ) | `figures_agreement.fig_metric_correlation_base` |
| 4.2 | `loss_metric_tradeoff.png` | `naive/full_matrix.csv` | `aggregate.load_matrix` | `figures_loss.fig_loss_metric_tradeoff` |
| 4.3 | `base_models_all_metrics.png` | `naive/full_matrix.csv` | `aggregate.load_matrix` | `figures_models.fig_base_all_metrics` |
| 4.4 | `agreement_base_models.png` | `naive/full_matrix_predictions.csv` | `agreement.indomain_prediction_table` → `agreement.pairwise_kappa` (Cohen κ) | `figures_agreement.fig_agreement_base` |
| 4.5 | `crossdomain_base.png` | `naive/full_matrix.csv` | `aggregate.load_matrix` → `best_per_cell` / `cell_matrix` | `figures_crossdomain.fig_crossdomain` |
| 5.1 | `hybrid_ablation_all_metrics.png` | `hybrid/hybrid_matrix.csv` + `naive/full_matrix.csv` (baseline ref) | `aggregate.load_hybrid_matrix` / `load_matrix` | `figures_hybrid.fig_ablation_all_metrics` |
| 5.2 | `crossdomain_hybrid.png` | `hybrid/hybrid_matrix.csv` | `aggregate.load_hybrid_matrix` → `cell_matrix` | `figures_crossdomain.fig_crossdomain` |
| 5.3 | `crossdomain_delta.png` | `naive/full_matrix.csv` + `hybrid/hybrid_matrix.csv` | `aggregate.cell_matrix` on both (subtraction) | `figures_crossdomain.fig_crossdomain_delta` |
| 5.4 | `private_confusion_combined.png` | `naive/full_matrix.csv` + `naive/full_matrix_predictions.csv` + `hybrid/hybrid_matrix.csv` + `hybrid/hybrid_matrix_predictions.csv` (LFS) | `aggregate.load_*` + `confusion_from_predictions` | `figures_private.fig_private_confusion` |
| 5.5 | `indomain_vs_generalization_hybrid.png` | `naive/full_matrix.csv` + `hybrid/hybrid_matrix.csv` | `figures_crossdomain._scatter_points` + scipy `spearmanr` (Spearman ρ) | `figures_crossdomain.fig_indomain_vs_generalization` |
| — | `private_per_class_f1.png` | `naive/full_matrix.csv` + `naive/full_matrix_predictions.csv` + `hybrid/hybrid_matrix.csv` + `hybrid/hybrid_matrix_predictions.csv` (LFS) | `aggregate.load_*` + `per_class_f1_from_predictions` | `figures_private.fig_private_per_class_f1` — still generated, no longer shown in the thesis (its per-class F1 numbers are folded into the §5.7.2 confusion-matrix discussion) |

---

## C. Tables

Built as `(name, frame, caption)` specs in `src/analysis/tables.py`; Markdown preview →
`outputs/thesis/tables/*.md`; LaTeX → `documents/thesis/Table/*.tex` via
`thesis_latex.generate_tables` + `latexfmt.dataframe_to_latex`. File stems follow the rendered
LaTeX number (`T<chapter>_<index>_<slug>`).

| Table no. | File stem (`\input`) | Raw data | Builder function (in `tables.py`) |
|---|---|---|---|
| 3.1 | `T3_1_dataset_stats` | `dataset_analysis/class_distribution_overall.csv` + `class_distribution_by_split.csv` | `table_dataset_stats` |
| 4.1 | `T4_1_per_metric_winner` | `naive/full_matrix.csv` | `table_per_metric_winner` |
| 4.2 | `T4_2_indomain_datasets` | `naive/full_matrix.csv` | `table_indomain_cmose_vs_daisee` |
| 4.3 | `T4_3_openface_vs_i3d` | `naive/full_matrix.csv` | `table_openface_vs_i3d` |
| 5.1 | `T5_1_group_marginal_combined` | `hybrid/hybrid_matrix.csv` | `table_group_marginal_combined` |
| 5.2 | `T5_2_i3d_fusion_effect` | `hybrid/hybrid_matrix.csv` | `table_i3d_fusion_effect` |
| 5.3 | `T5_3_hybrid_topk` | `hybrid/hybrid_matrix.csv` | `table_hybrid_topk` |
| 5.4 | `T5_4_private_by_source` | `naive/full_matrix.csv` + `hybrid/hybrid_matrix.csv` | `table_private_by_source` |

**Hand-authored (not generated):** Table 3.2 `tab:groups` in `3_Methodology.tex` — the 709-feature
semantic decomposition is a hardcoded `tabular`, no data/code behind it.

**Generated but NOT `\input` into the thesis** (kept for provenance; retain legacy stems):
`T2_base_indomain` (`table_base_indomain`) and `T9_agreement_stats` (`table_agreement_stats`, which
drives `agreement.oracle_stats` / `pairwise_kappa` / `pair_overlap`). The functions
`table_group_marginal` and `table_group_marginal_unseen` exist but are not in `build_all`, so they
render nothing.

---

## D. Upstream producers of the raw data

- `model_assessment/naive/full_matrix.csv` + `full_matrix_predictions.csv` ← `src/evaluation/evaluate_full_matrix.py`
- `model_assessment/hybrid/hybrid_matrix.csv` + `hybrid_matrix_predictions.csv` ← `src/evaluation/evaluate_hybrid_full_matrix.py` (+ `generate_hybrid_clip_predictions.py`); metrics in `src/evaluation/metrics.py`
- `training_log/<dataset>/<model>/<loss>/metrics.json` ← training via `main.py` / `src/training/full_training_process.py`
- `dataset_analysis/*.csv` ← committed dataset-analysis artifacts (consumed by `aggregate.load_class_distribution_*`)
