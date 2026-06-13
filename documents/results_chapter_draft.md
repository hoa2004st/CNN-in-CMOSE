# Results chapters — question-driven draft (v3, protocol-first)

> Supersedes `results_interpretation.md` and the current narrative of
> `Chapter/4_Baseline_models.tex` / `Chapter/5_Semantic_group_hybrid.tex`.
> Ordering convention: an **Evaluation protocol** section (A.2) fixes metrics, corpus,
> and loss *before any results*, using a-priori reasoning plus the evidence that is
> genuinely in-domain; the two evidence legs that require the generalization study
> (accuracy's blindness to collapse, the stability of the loss ranking) are
> **forward-referenced** there and **closed** in A.5.2. The question each section
> answers is noted as a small tag (A1–A8, B1–B4) and mapped in the inventory at the end.
> Every number is taken from the regenerated artifacts in `outputs/thesis/`
> (tables T1–T10, figures). Artifacts marked **NEW** were added for this draft.
>
> **Single-seed caveat (state once in A.1):** every configuration was trained once
> (seed 42). Wherever the draft says *mean* or *max*, the aggregation is over the
> hyperparameter grid (losses, the 243 encoder assignments), never over random seeds.

---

## Chapter A — Baseline models

### A.1 Experimental setup

Frame: 5 architectures (`openface_mlp`, `openface_tcn`, `openface_lstm`,
`openface_transformer`, `i3d_mlp`) × 3 losses (CE, Weighted CE, Ordinal) × 6 metrics,
evaluated in-domain and on the 3×3 train×test matrix (CMOSE / DaiSEE / Combined →
CMOSE-test / DaiSEE-test / Private).

**Aggregation policy (used everywhere below):**

- **Max** when crowning a winner. The loss (and, in Chapter B, the encoder assignment)
  is a hyperparameter selected on validation data, so "which architecture is best" means
  "best achievable when tuned" — that is the max over the tunable axis. Averaging over
  losses would punish an architecture for losses it would never ship with, and mixes
  objectives that deliberately optimise different metrics.
- **Mean / distribution** when characterising a *family* (Chapter B's 243-config
  ablation, the per-group marginals): no one tunes 243 configs, so the claim "the family
  is better" must hold for the population, not its lucky maximum.
- With one seed per config, mean-over-grid measures robustness to design choices, not
  noise — say this explicitly.

### A.2 Evaluation protocol

Before any model is compared, three conventions must be fixed: which metrics to read,
which corpus to develop on, and which loss to develop with. Each is justified here on
a-priori grounds plus the evidence that is itself in-domain; the two arguments that
need out-of-domain evidence are stated as claims and *proved in A.5.2* — the reader is
told exactly where the loop closes.

#### A.2.1 Metrics *(A6, sets up A2)*

**A-priori:** engagement is an *ordinal*, *heavily imbalanced* 4-class problem
(class shares in `T1_dataset_stats`). Raw accuracy ignores both properties: it rewards
majority-class collapse and counts a one-level miss the same as a three-level miss. QWK
is the standard agreement metric for exactly this setting (ordinal, chance-corrected,
quadratically distance-weighted).

**In-domain evidence:** the data say the same thing twice.

1. *The leaderboard winner depends on the metric* (Table `T8_per_metric_winner`, **NEW**):

   | Metric | Winner |
   |---|---|
   | QWK | openface_tcn / CE (0.537) |
   | Macro-accuracy | i3d_mlp / Ordinal (0.635) |
   | Macro-MAE | i3d_mlp / Ordinal (0.458) |
   | Accuracy, MAE, Cohen κ | i3d_mlp / CE (0.773 / 0.244 / 0.436) |

   No metric choice is innocent — each crowns a different champion, so the choice must
   be argued, not defaulted.
2. *The six metrics form two nearly orthogonal blocks* (Figure
   `metric_correlation_base.png`, **NEW**): a *micro* block (Accuracy↔MAE, Kendall
   τ = 0.98) and a *macro* block (Macro-accuracy↔Macro-MAE, τ = 0.89), with cross-block
   τ ≈ 0.01–0.10. A metric from either block alone discards the other half of the
   picture. QWK is the only metric with moderate-to-strong agreement with *both* blocks
   (τ = 0.49–0.60, plus τ = 0.79 with Cohen κ): the compromise metric, sensitive to
   ordinal distance without being enslaved to the majority class.

**Protocol:** primaries are **QWK (emphasised), macro-accuracy, macro-MAE**; accuracy is
retained only as a foil. *Deferred evidence:* the decisive argument against accuracy —
that it cannot even detect cross-corpus collapse — requires the generalization study and
is delivered in **A.5.2**.

#### A.2.2 Development corpus *(A1)*

This choice needs no deferred evidence: it rests entirely on the two in-domain
(diagonal) cells. The best in-domain CMOSE model reaches QWK 0.537 / macro-accuracy
0.535; the best in-domain DaiSEE model only QWK 0.166 / macro-accuracy 0.289 — barely
above chance on every chance-corrected metric, while its raw accuracy (0.548) hides the
failure (a first taste of A.2.1's foil). A corpus on which no model rises above chance
cannot separate good architectures from bad ones: whatever ranking it produces is noise.

**Protocol:** **CMOSE is the development corpus**; DaiSEE is kept only as a transfer
target, where its difficulty is informative rather than obstructive.

- Table `T7_indomain_datasets`, Figure `indomain_cmose_vs_daisee.png`.
- Context: class imbalance figures `dataset_class_distribution_overall/by_split.png`, `T1`.

#### A.2.3 Loss *(A7)*

In-domain, the loss axis is a clean trade: best-architecture QWK by loss is CE 0.537 /
Weighted CE 0.500 / Ordinal 0.487, while the rebalanced losses buy macro-accuracy and
macro-MAE at that known QWK cost (the Pareto story: Figures `loss_metric_tradeoff.png`,
`loss_pareto_macroacc.png`, `loss_pareto_mae.png`). Under a QWK-led protocol the choice
is therefore CE; the rebalanced losses remain a deliberate option when macro recall is
the deployment goal, not a more reliable default.

**Protocol:** **CE is the development loss.** *Deferred evidence:* **A.5.2** shows the
CE > Weighted CE > Ordinal ranking is preserved intact under domain shift — the only
loss effect that survives — which retroactively makes this the reliable choice, not just
the convenient one.

### A.3 In-domain leaderboard *(A2)*

With the protocol fixed, the architecture question has a clean answer. Winner under the
primaries: **openface_tcn/CE, QWK 0.537** (max-over-loss per A.1). The full 15-config
leaderboard (Table `T2_base_indomain`, Figure `base_models_all_metrics.png`) adds two
structural observations:

- The **loss axis decides which metric family you win** (CE → micro and ordinal metrics,
  rebalanced → macro metrics) — the per-metric winners of T8 are this pattern seen from
  the other side.
- The **architecture axis decides the feature story**: OpenFace-TCN tops the ordinal
  primary while i3d_mlp/CE sits statistically adjacent at 0.519 and tops everything
  frequency-weighted. Two near-tied models built on disjoint feature families is the
  setup for A.4.2 — the score table cannot tell whether they are redundant or
  complementary.

### A.4 Prediction agreement

#### A.4.1 Agreement between configurations *(A3)*

On the 1,221 in-domain CMOSE test clips (Figure `agreement_base_models.png`, Table
`T9_agreement_stats`, both **NEW**), mean pairwise Cohen κ between configs is only
0.27–0.39 — far below what their clustered scores suggest. Sharing the architecture
(κ = 0.391) or the loss (κ = 0.374) raises agreement only mildly over sharing neither
(κ = 0.273): disagreement is driven by *both* axes, so the configs are genuinely
different hypotheses about the data, not reparametrisations of one another.

The practical consequence is headroom: the best single config gets 77.3% of clips
right, but at least one of the 15 is right on **97.6%** (oracle); only 2.4% of clips
defeat every model. Yet plain majority voting lands at 73.5% — *below* the best single
model — because the errors are too decorrelated for unweighted voting to harvest. The
headroom exists, but claiming it requires a *learned* combination. That is precisely the
design premise of the Chapter B hybrid.

#### A.4.2 OpenFace versus I3D *(A5)*

Against `openface_tcn/CE` (Figure `feature_error_overlap.png`, **NEW**), `i3d_mlp/CE`
has the largest exclusive-correct share (8.6% of clips that only I3D gets right, vs
5–6% for the OpenFace partners) and the smallest both-wrong share (15.3% vs 18–19%
within the OpenFace family). The pair's oracle accuracy is 84.7% — +7.4 points over the
best single model, the largest pair gain in the pool. The behaviour descriptors and the
appearance features are looking at genuinely different evidence; fusing them (Chapter B)
is motivated by this error structure, not by the score table, where i3d_mlp's 0.519
looks redundant next to 0.537.

### A.5 Generalization

#### A.5.1 The 3×3 cross-dataset matrix *(A4)*

Best-per-cell QWK (Figure `crossdomain_base.png`):

| train \ test | CMOSE | DaiSEE | Private |
|---|---|---|---|
| CMOSE | **0.537** | 0.024 | 0.190 |
| DaiSEE | 0.045 | **0.166** | 0.256 |
| Combined | 0.477 | 0.136 | 0.285 |

The off-diagonal collapse spares no factor combination. Along the architecture axis the
in-domain ranking is *reshuffled* (**NEW** Figure `generalization_by_arch.png`):
in-domain `openface_tcn` 0.537 > `i3d_mlp` 0.519 > `openface_transformer` 0.443 …; on
unseen targets `openface_tcn` 0.112 ≈ `openface_transformer` 0.110 > `openface_lstm`
0.096 > `i3d_mlp` 0.085 > `openface_mlp` 0.065. **I3D falls from 2nd to 4th**:
appearance features bind to the corpus, while the OpenFace behaviour descriptors carry
more corpus-invariant signal. In-domain and generalization are thus *complementary*
selection criteria along the architecture axis, not one criterion measured twice.

For an unknown target, pooling corpora is the best simple mitigation: Combined training
gives the best private-set result (0.285) while keeping near-in-domain CMOSE
performance (0.477).

#### A.5.2 Closing the loop: metric and loss reliability *(A6, A7)*

The two claims deferred from the protocol are now provable.

- **Accuracy cannot see the collapse (completes A.2.1).** Across the off-diagonal cells
  accuracy stays at 0.49–0.62 while QWK falls to 0.02–0.29 (Figure
  `crossdomain_base.png`). A metric that reports business-as-usual while the model has
  lost all ordinal signal is not a safety net but a blindfold. Combined with its
  in-domain blindness to imbalance (DaiSEE's 0.548, A.2.2), accuracy is the least
  reliable metric in the study; QWK — the only block-bridging metric (A.2.1) and the one
  that detects collapse — is the most. The protocol's metric choice stands confirmed.
- **The loss ranking survives shift (completes A.2.3).** Best-architecture QWK by loss
  on the unseen-target mean: CE 0.160 / Weighted CE 0.099 / Ordinal 0.093 — the same
  order as in-domain (Figure `generalization_by_loss.png`, **NEW**). The loss is the
  only factor whose ranking transfers; CE is reliable in the strict sense that choosing
  it in-domain remains the right choice out of domain.

#### A.5.3 In-domain versus transfer *(A8)*

One point per trained model, x = in-domain QWK, y = mean QWK over its unseen-target
cells (**NEW** Figure `indomain_vs_generalization_base.png`): Spearman ρ = **0.28**
over the 30 base points — a link too weak to select on. The hybrid overlay
(`indomain_vs_generalization_hybrid.png`, revisited in B.5) sharpens the warning:
within the 972-point ablation population ρ = **−0.31**, i.e. among closely related
configs, squeezing out more in-domain QWK mildly *hurts* transfer — the extra fit is
corpus fit. Model selection for deployment therefore cannot read the in-domain
leaderboard alone; it must lean on the pooling result of A.5.1.

### A.6 Chapter summary

1. Protocol: QWK-led primaries, CMOSE as development corpus, CE as development loss —
   each fixed in A.2 on a-priori + in-domain grounds, and confirmed where deferred
   (A.5.2).
2. CMOSE (0.537) is the only usable development corpus; DaiSEE (0.166) evaluates noise.
3. Each metric crowns its own winner (T8); under the primaries the baseline bar is
   openface_tcn/CE = 0.537 (max-over-loss; single-seed caveat).
4. Configs agree far less than their scores suggest (κ 0.27–0.39); oracle 97.6% vs best
   single 77.3% — headroom only a learned combiner can claim.
5. OpenFace and I3D err on different clips (both-wrong 15.3%) — complementary families.
6. Off-diagonal collapse is universal; the architecture ranking reshuffles (I3D drops);
   pooling is the best simple defence.
7. QWK is the most reliable metric (block-bridging, collapse-detecting); accuracy the
   least. CE is the most reliable loss (ranking preserved under shift).
8. In-domain rank predicts generalization weakly (ρ = 0.28) and inverts within a family
   (ρ = −0.31).

---

## Chapter B — Semantic-group hybrid

### B.1 Overview

243 encoder assignments (TCN/Transformer/LSTM per semantic group: gaze, eye, face, head,
AU) × {with, without I3D} = 486 configs, all CE loss, evaluated on the same frame as
Chapter A. Family-level claims use means/distributions; headline comparisons use the max
(A.1 policy).

### B.2 Consistency with Chapter A *(B1)*

The chapter opens by re-validating the A.2 protocol on a 32-fold larger model
population — if the conventions were artifacts of five hand-picked baselines, this is
where they would break. They do not:

- *Metric structure reproduced*: the same two-block Kendall-τ pattern (micro block 0.87,
  macro block 0.73, cross-block 0.26–0.33, QWK again the bridge at 0.63–0.73) —
  **NEW** Figure `metric_correlation_hybrid.png` vs `metric_correlation_base.png`.
- *Accuracy still blind*: in the ablation, accuracy barely separates configs that differ
  by 0.1 QWK (Figure `hybrid_ablation_all_metrics.png`).
- *Cross-domain collapse reproduced*: hybrid best-per-cell QWK matrix (Figure
  `crossdomain_hybrid.png`): diagonal 0.605 / 0.228, off-diagonal 0.065–0.379 — the same
  shape as A.5.1 at a higher level.

The Chapter A frame (QWK-led primaries, CE loss, accuracy as foil) therefore carries
over without amendment, and the findings it produced are properties of the task, not of
any architecture.

### B.3 Encoder choice per semantic group *(B2)*

In-domain (Table `T5_group_marginal`, Figure `hybrid_group_marginal.png`): head pose
shows a 0.022-QWK spread between encoders with TCN best; every other group's spread is
≤ 0.006 — encoder-agnostic. On the pooled unseen-target cells (**NEW** Table
`T10_group_marginal_unseen`, Figure `hybrid_group_marginal_unseen.png`) head pose again
has the largest spread (0.010) and again prefers TCN, while the other groups' spreads
(0.002–0.004) flip winners arbitrarily — noise. The consistency across regimes is what
elevates this from a tuning accident to a property of the signal: head dynamics carry
local, rhythm-like temporal structure that a TCN's receptive field matches; the
remaining groups' information is encoder-invariant, so their encoder budget can be spent
freely.

### B.4 Effect of I3D fusion *(B3)*

Pairing each OpenFace-only config with its exact +I3D twin isolates the fusion effect
from the configuration (**NEW** Figure `i3d_paired_delta.png`):

| Regime | mean ΔQWK | pairs with Δ>0 |
|---|---|---|
| Seen target (4 cells, n=972) | **+0.060** | 94% |
| Cross-corpus (2 cells, n=486) | −0.021 | 26% |
| Private set (3 cells, n=729) | +0.005 (huge variance, ±0.2) | 52% |

In-domain, this is the cash value of A.4.2's complementarity: the +I3D family median is
0.553 vs 0.522 without, and 83% of +I3D configs clear the 0.537 baseline bar vs 27% of
OpenFace-only. Out of domain the gain evaporates and slightly reverses — I3D's
appearance features drag corpus bias along, the same mechanism behind i3d_mlp's rank
drop in A.5.1. The two results are one finding seen at two scales, and they yield a
practical rule: fuse I3D when the deployment domain is represented in training; rely on
the OpenFace groups otherwise.

### B.5 Hybrid versus baseline *(B4)*

- *In-domain*: best hybrid (+I3D, `TCN_T_TCN_LSTM_T`) QWK **0.605** vs best base 0.537
  (**+0.068**); Table `T4_hybrid_topk`, Figures `hybrid_best_comparison.png`,
  `confusion_best_models.png`, `per_class_f1.png`. Gains concentrate in the
  upper-engagement classes; Highly-Disengage remains the failure mode.
- *Every cell*: hybrid ≥ base in all nine train×test cells, e.g. CMOSE→Private 0.309 vs
  0.190, DaiSEE→DaiSEE 0.228 vs 0.166 (compare `crossdomain_hybrid.png` with
  `crossdomain_base.png`). The improvement is not a single tuned point but a uniform
  shift of the whole matrix.
- *Private set* (Table `T6_private_by_source`, Figures `private_by_source.png`,
  `private_confusion_combined.png`): hybrid beats base from every training source; under
  Combined training **0.379 vs 0.285 (+0.094)** — a larger margin than in-domain. The
  two contributions (corpus pooling, hybrid architecture) stack on the hardest, most
  realistic target.
- *Why it generalizes better* (closing the loop with A.5.3): the hybrid points sit above
  the base trend in `indomain_vs_generalization_hybrid.png` — the architecture lifts the
  whole in-domain↔transfer trade-off instead of riding the weak (or inverted)
  correlation along it.

### B.6 Chapter summary

1. All Chapter A findings reproduce on the 486-config population (metric blocks,
   accuracy blindness, cross-domain collapse) — the protocol and its findings are task
   properties.
2. Encoder choice matters only for head pose (TCN), in-domain *and* out; the other four
   groups are encoder-agnostic.
3. I3D fusion: +0.060 QWK paired gain when the test corpus is seen (94% of pairs),
   neutral-to-harmful beyond it.
4. The hybrid beats the best baseline in every cell of the matrix; +0.068 in-domain
   grows to +0.094 on the private set under pooled training. Highly-Disengage remains
   the open problem (data/objective, not architecture).

---

## Artifact inventory and question map

| Section | Answers | Tables | Figures |
|---|---|---|---|
| A.2.1 | A6 (in-domain leg) | **T8**, T1 | **metric_correlation_base** |
| A.2.2 | A1 | T7, T1 | indomain_cmose_vs_daisee, dataset_class_distribution_* |
| A.2.3 | A7 (in-domain leg) | — | loss_metric_tradeoff, loss_pareto_* |
| A.3 | A2 | T2 | base_models_all_metrics |
| A.4 | A3, A5 | **T9** | **agreement_base_models**, **feature_error_overlap**, openface_vs_i3d |
| A.5.1 | A4 | — | crossdomain_base, **generalization_by_arch** |
| A.5.2 | A6, A7 (deferred legs) | — | crossdomain_base, **generalization_by_loss** |
| A.5.3 | A8 | — | **indomain_vs_generalization_base** |
| B.2 | B1 | — | **metric_correlation_hybrid**, hybrid_ablation_all_metrics, crossdomain_hybrid |
| B.3 | B2 | T5, **T10** | hybrid_group_marginal, **hybrid_group_marginal_unseen** |
| B.4 | B3 | — | **i3d_paired_delta** |
| B.5 | B4 | T4, T6 | hybrid_best_comparison, confusion_best_models, per_class_f1, private_by_source, private_confusion_combined, **indomain_vs_generalization_hybrid** |

(**bold** = new in this revision; all live in `outputs/thesis/` and are already
published to `documents/thesis/Figure|Table/` by `make_thesis_artifacts`.
A1–A8 = the eight baseline-chapter questions, B1–B4 = the four hybrid-chapter
questions, in the order they were posed.)
