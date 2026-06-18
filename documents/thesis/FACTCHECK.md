# Thesis Fact-Check Provenance Ledger

Every numeric / factual claim in `documents/thesis/` and how to verify it independently.
Protocol that governs this file: see **AGENTS.md → "Fact-Check Protocol"**.

- **Source of truth wins.** If a row disagrees with its source file, the thesis is wrong, not the file
  (unless `git log outputs/<file>` shows the CSV is stale relative to the last retrain).
- **Status:** `OK` = verified equal at stated rounding · `MISMATCH` = differs · `TODO` = not yet checked · `N/A` = prose, no number.
- **Manual check:** paste the command into PowerShell from the repo root; round to the thesis's precision.

## Source files (single source of truth)
| Alias | Path |
| --- | --- |
| `BASE` | `outputs/model_assessment/naive/full_matrix.csv` |
| `HYB` | `outputs/model_assessment/hybrid/hybrid_matrix.csv` |
| `BASE_PRED` | `outputs/model_assessment/naive/full_matrix_predictions.csv` |
| `HYB_PRED` | `outputs/model_assessment/hybrid/hybrid_matrix_predictions.csv` |
| `DIST` | `outputs/dataset_analysis/class_distribution_overall.csv` |
| `SPLIT` | `outputs/dataset_analysis/class_distribution_by_split.csv` |
| `PRIV` | `outputs/dataset_analysis/private/dataset_summary.json` |
| `TABLES` | `outputs/thesis/tables/T*.md` |
| `PDF` | `documents/references/*.pdf` |
| `CODE` | `src/models/models.py`, `src/evaluation/metrics.py`, `src/training/*` |

Reusable lookup helper (PowerShell):
```powershell
function Cell($csv,$tr,$te,$mdl,$ls){ Import-Csv $csv |
  Where-Object { $_.train_group -eq $tr -and $_.test_set -eq $te -and $_.model -eq $mdl -and $_.loss -eq $ls } }
# e.g. (Cell BASE 'cmose' 'cmose_test' 'tcn' 'ce').quadratic_weighted_kappa
```

---

## Verified headline numbers (seed set)

| # | Claim (thesis) | Where (.tex) | Source | Manual check | Expected | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | OpenFace TCN, CE, in-domain CMOSE **QWK 0.537** | Ch.4 §4.3; Ch.1; Ch.5 | `BASE` | `(Import-Csv $BASE \| ? {$_.train_group -eq 'cmose' -and $_.test_set -eq 'cmose_test' -and $_.model -eq 'tcn' -and $_.loss -eq 'ce'}).quadratic_weighted_kappa` | 0.5372 → **0.537** | OK |
| 2 | …same model **macro-acc 0.535**, **macro-MAE 0.547** | Ch.4 §4.3 | `BASE` | same row, `.macro_accuracy` / `.macro_mae` | 0.5349 / 0.5469 | OK |
| 3 | I3D MLP, CE, CMOSE **QWK 0.519** | Ch.4 §4.3 | `BASE` | row `cmose/cmose_test/i3d_mlp/ce` `.quadratic_weighted_kappa` | 0.5192 → 0.519 | OK |
| 4 | CMOSE **69.4% Engage**, **2.8% HD** | Ch.1; Ch.3 §3.2; T1 | `DIST` | `Import-Csv $DIST \| ? {$_.dataset -eq 'CMOSE'}` | EG 0.6944, HD 0.0285 | OK |
| 5 | CMOSE **12,197** clips (8,783/2,193/1,221) | Ch.3 §3.2; T1 | `SPLIT` | sum `split_total` per split for CMOSE | 8783+2193+1221 = 12197 | OK |
| 6 | DAiSEE **8,571** (5,358/1,429/1,784) | Ch.3 §3.2; T1 | `SPLIT` | split totals for DaiSEE | 5358+1429+1784 = 8571 | OK |
| 7 | Private **366** clips; 58.2/30.6/8.5/2.7% | Ch.3 §3.2; Ch.5 §5.6 | `PRIV` | `Get-Content $PRIV \| ConvertFrom-Json` → `total_clips`, `class_distribution` | 366; 0.582/0.306/0.085/0.027 | OK |
| 8 | Best hybrid **QWK 0.605** in-domain CMOSE (`TCN_T_TCN_LSTM_T`, +I3D) | Ch.1; Ch.5 §5.5 | `HYB` | max `quadratic_weighted_kappa` over `cmose/cmose_test`; confirm `arch_key` | 0.6053 → 0.605, arch `TCN_T_TCN_LSTM_T` ✓ | OK |
| 9 | Median I3D-fused hybrid **QWK 0.553** vs OpenFace-only **0.522**; **82%** vs **26%** clear the 0.537 bar | Ch.1; Ch.5 §5.4 | `HYB` | median QWK of fused vs non-fused families, `cmose/cmose_test`; frac > 0.5372 each | 0.553 / 0.522; 0.82 / 0.26 | OK |
| 10 | Best private model **QWK 0.379**, macro-MAE 0.877 (`T_T_T_LSTM_T`, combined) | Ch.1; Ch.5 §5.6 | `HYB` | row `combined/private` max QWK; confirm `arch_key`,`.macro_mae` | 0.3791 / 0.8773, arch `T_T_T_LSTM_T` ✓ | OK |
| 11 | Best base private **QWK 0.285** (combined) > DAiSEE 0.256 > CMOSE 0.190 | Ch.4 §4.6; Ch.5 §5.6 | `BASE` | max QWK per `train_group` on `private` | 0.2850/0.2563/0.1904 | OK |
| 12 | In-domain hybrid gain **+0.068** (0.605−0.537) | Ch.5 §5.5 | derived | rows #8 − #1 | 0.6053−0.5372 = +0.068 | OK |
| 13 | Private hybrid gain **+0.094** (0.379−0.285) | Ch.1; Ch.5 §5.6 | derived | rows #10 − #11(combined) | 0.3791−0.2850 = +0.094 | OK |
| 14 | Oracle **97.6%**, best single **77.3%**, majority-vote 73.5% | Ch.4 §4.4 | `BASE_PRED` | recompute over 15 cfg on `cmose_test`; or `outputs/thesis/tables/T9_agreement_stats.md` | 0.976 / 0.773 / 0.735 | TODO |
| 15 | I3D-only exclusive-correct **8.6%**; pair oracle 84.7% | Ch.4 §4.4 | `BASE_PRED` | recompute TCN vs I3D on `cmose_test` | 0.086 / 0.847 | OK |
| 16 | Paired I3D gain: seen-target **+0.060 / 94% of 972**; cross-corpus **−0.021 / 26% of 486**; private **+0.005 / 52% of 729** | Ch.5 §5.4; Ch.6; T13 | `HYB` paired | pair each cfg with its +I3D twin by regime; mean & frac>0 (see `tables.table_i3d_fusion_effect`) | +0.060/0.94; −0.021/0.26; +0.005/0.52 | OK |
| 17 | Head pose prefers TCN: in-domain QWK 0.545>0.541>0.523 | Ch.5 §5.3; T12 | `outputs/thesis/tables/T12_group_marginal_combined.md` | read table | 0.545/0.541/0.523 | OK |
| 19 | Combined→CMOSE best-base **QWK 0.477** (near-in-domain under pooling) | Ch.4 §4.5.1; `crossdomain_base` fig | `BASE` | max QWK `combined/cmose_test` | 0.4773 → 0.477 (fig shows 0.48) | OK |
| 18 | ~**1,500** trained runs; **243** = 3^5 configs | Ch.1; Ch.3 §3.9 | `CODE` + counts | distinct `arch_key` in `HYB` = 243; run-count narrative 5×3×3 + 243×2 | 243 distinct arch_key ✓; run total TODO | OK (243) |

## Equations (form + implementation)
| Eq | Claim | Check against | Status |
| --- | --- | --- | --- |
| eq:emd | ordinal/EMD loss = class-weighted mean sq. CDF distance (`w_y` outside class sum) | `OrdinalEMDLoss` in `src/training/train.py` (cumsum of softmax vs one-hot, mean over classes, `*weight[targets]`) | OK |
| eq:qwk | QWK weight `w_ij=(i-j)^2/(C-1)^2` | `src/evaluation/metrics.py` + Cohen 1968 | TODO |
| LSTM gates | Eq. in Ch.2 | Hochreiter & Schmidhuber 1997 (standard form) | TODO |
| attention | softmax(QK^T/√d_k)V | Vaswani et al. 2017 | TODO |
| inv-freq weights | `w_c=(1/n_c)·C/Σ(1/n_k)`, mean 1 | training code | TODO |

## Citations (final published version + attributed claim)
All entries upgraded to most-reliable version (journal > conference > preprint); DOIs/pages
web-verified. Source of truth = `documents/thesis/reference.bib`.
| Cite key | Final version (verified) | Attributed claim verified | Status |
| --- | --- | --- | --- |
| `cmose` | CVPR-W 2024, pp.4636–4645, DOI 10.1109/CVPRW63382.2024.00466 | — | metadata OK |
| `daisee` | arXiv:1609.01885, 2016 — **no peer-reviewed version exists** | "in the wild", 4 levels | metadata OK (arXiv unavoidable) |
| `openface2` | FG 2018, pp.59–66, DOI 10.1109/FG.2018.00019 | toolkit channels | metadata OK |
| `i3d` | CVPR 2017, pp.4724–4733, DOI 10.1109/CVPR.2017.502 | inflated 3D, Kinetics | metadata OK |
| `tcn` | arXiv:1803.01271 — **ICLR 2018 submission rejected; no formal pub** | causal dilated conv | metadata OK (arXiv unavoidable) |
| `transformer` | NeurIPS 2017, pp.5998–6008 | self-attention | metadata OK |
| `efficientnetv2` | ICML 2021, PMLR 139, pp.10096–10106 | backbone | metadata OK |
| `lstm` / `smote` / `cohenkappa` | journals, DOIs added | standard | metadata OK |
| `review_engagement_cv` | IEEE Access 13, 2025, pp.140519–140545, DOI 10.1109/ACCESS.2025.3596885 | "imbalance + subjectivity are open problems" | metadata OK; claim TODO |
| `openface_pca_smote` | IJACSA 14(3), 2023, pp.617–626, DOI 10.14569/IJACSA.2023.0140371 | "PCA/SVD + SMOTE before CNN" | metadata OK; claim TODO |
| `openface_bilstm` | **UPGRADED** MSc thesis → ICIAP 2022, LNCS 13233, pp.411–422, DOI 10.1007/978-3-031-06433-3_35 (Çopur, Nakip, Scardapane, Slowack) | **BiLSTM specifically** | metadata OK; claim TODO |
| `efficientnet_lstm_engagement` | J. on AI 6(1), 2024, pp.85–103, DOI 10.32604/jai.2024.048911 | "EfficientNetV2-L + LSTM, best 62.11%" | metadata OK |
| `classical_ml_engagement` | arXiv:2405.04251, 2024 — **no peer-reviewed version found** | **"LR/SVM/MLP/KNN/XGBoost comparison"** | metadata OK (arXiv unavoidable); **claim AT RISK, verify in paper body** |
| `context_aware_emotion_3d` | TNNLS 36(7), 2025, pp.13567–13578, DOI 10.1109/TNNLS.2024.3476249 | "context-aware affect / spatial relations" | metadata OK |

## Open flags (highest priority)
- **`classical_ml_engagement`**: thesis (Ch.1, Ch.2) attributes a classical-ML bake-off (LR/SVM/MLP/KNN/XGBoost); the paper abstract describes a single lightweight sequential model (68.57%). **Confirm the comparison exists in the paper body or re-point the citation.**
- **`openface_bilstm`**: cited document is a master's thesis; confirm BiLSTM usage and consider citing the peer-reviewed version.
- **DAiSEE 8,571 vs paper's 9,068**: this is the tracking-filtered subset — verify against `SPLIT`, never against the DAiSEE paper.
