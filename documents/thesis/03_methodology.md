# Chapter 3. Methodology

## 3.1 Overview

We frame engagement recognition as 4-class ordinal classification of a fixed-length clip.
Figure 3.1 shows the conceptual pipeline: (1) per-frame OpenFace descriptors and I3D motion
features are extracted and resampled to fixed length; (2) OpenFace features are split into five
**semantic groups**; (3) each group — and, in the multimodal variant, the I3D stream — is
encoded by its own temporal encoder into a fixed embedding; (4) embeddings are concatenated and
classified, with an auxiliary head supervising each stream. Baselines replace steps (2)–(3)
with a single monolithic encoder. The same models are trained and evaluated under a **3×3
cross-dataset protocol** and four loss functions.

> TODO: draw Figure 3.1 (conceptual framework). A clean block diagram of streams → encoders →
> fusion → classifier, mirroring `src/models/models.py:OpenFaceTemporalI3DHybrid`.

## 3.2 Data and Preprocessing

**Datasets.** We use three training sources and three test sets (Table T1):

- **CMOSE** [@cmose]: 12,197 clips; splits `train` (8,783) / `unlabel` (2,193, used as
  validation) / `test` (1,221).
- **DAiSEE** [@daisee]: 8,571 clips; `Train/Validation/Test` mapped to `train/unlabel/test`.
- **Combined**: CMOSE ∪ DAiSEE (20,768 clips), ids prefixed to avoid collision.
- **Private (self-collected)**: a held-out set of **366 clips that we collected and manually
  labeled** ourselves, used **for testing only** (never in training or model selection). Its
  label prior differs again from both public corpora — 58.2% Engage, 30.6% Highly-Engage, 8.5%
  Disengage, and only 2.7% Highly-Disengage. Because it is independently sourced and labeled, it
  is the most realistic measure of how the models would behave in deployment, and it anchors
  Section 4.6.

  > TODO: describe the private-set collection methodology — recording setup, number of
  > participants/sessions, clip segmentation, the labeling protocol and rubric you applied, and
  > any inter-rater or self-consistency checks. This is a genuine contribution and deserves a
  > full paragraph; the pipeline/feature processing is identical to CMOSE.

All four share the four ordinal labels HD/DE/EG/HE. Class proportions (Table T1, Figure
`dataset_class_distribution_overall`) are heavily skewed and skewed *differently* across CMOSE
(Engage-dominated) and DAiSEE (Engage + Highly-Engage), which is central to Sections 4.5–4.6.

**Features.** Per-frame OpenFace 2.0 [@openface2] descriptors (709-dim) and precomputed
I3D [@i3d] features (1024-dim). Each clip is resampled along time to a fixed length
(**300 frames** for OpenFace in single-dataset runs, **150** for the larger Combined runs due
to memory; **75** frames for the I3D stream).

**Normalization.** Per-feature z-score using statistics fit on the **training split only**;
the validation, test, and private sets are transformed with the train statistics to prevent
leakage. OpenFace and I3D are normalized independently.

> TODO: confirm the exact resampling function and any clipping; cite `extract_openface.py` /
> `extract_i3d.py`.

## 3.3 Semantic-Group Decomposition

The 709 OpenFace features are partitioned by name prefix into five behaviorally meaningful
groups, in a fixed order (`OPENFACE_GROUP_ORDER`):

| Group | Dim | Content |
|---|---|---|
| Gaze | 8 | gaze direction vectors / angles |
| Eye landmarks | 280 | 2D/3D eye-region landmarks |
| Face landmarks | 340 | facial landmark coordinates |
| Head pose | 46 | head translation/rotation |
| Action units | 35 | AU presence/intensity |

This decomposition is the inductive prior of the method: the five subsystems have different
temporal dynamics (e.g. rapid micro-saccades vs. slow head turns), so each can be matched to a
suitable encoder.

## 3.4 Hybrid Architecture (Proposed)

**Per-group encoder.** Each group is encoded by a `GroupEncoder` that is either a **TCN** (a
stack of dilated 1-D convolutions [@tcn]) or a **Transformer** encoder [@transformer], followed
by temporal pooling and a LayerNorm, producing a fixed **64-dim** embedding.

**Fusion and heads.** The five group embeddings are concatenated (320-dim) and passed through a
2-layer MLP classifier (320→128→4) with dropout. Each group additionally has an **auxiliary
classification head**; the total loss is the main-head loss plus `aux_weight = 0.2` times the
mean of the per-group auxiliary losses. The auxiliary supervision encourages every stream to be
individually discriminative (`OpenFaceTemporalHybrid`, `src/models/models.py`).

**Multimodal variant.** `OpenFaceTemporalI3DHybrid` adds a sixth stream: an I3D TCN encoder
(also 64-dim) with its own auxiliary head; the fusion MLP then takes 384-dim input. The I3D
encoder is always a TCN.

**Configuration / arch_key.** A hybrid configuration is specified by the encoder chosen for
each of the five groups, written as an `arch_key` of five tokens in group order, e.g.
`T_TCN_TCN_T_TCN` (`T` = Transformer, `TCN` = TCN). With two choices per group there are
**2⁵ = 32** configurations, evaluated both without and with I3D (Section 4.4).

## 3.5 Baseline Models

Five monolithic baselines (allow-list `MODEL_ORDER`): `openface_mlp` (flattened OpenFace →
MLP), `openface_tcn` (`temporal_cnn`), `openface_lstm`, `openface_transformer` (all on the full
709-dim OpenFace sequence), and `i3d_mlp` (I3D features → MLP). The `openface_tcn_i3d_fusion`
model exists in the codebase but is **excluded from assessment** to keep the comparison focused
on the semantic-group hypothesis.

## 3.6 Loss Functions

Each model is trained under **cross-entropy**, **weighted cross-entropy** (inverse-frequency
weights), and an **ordinal (EMD) loss**; **focal loss** [@focal] is available. Weights for the
weighted/ordinal variants are computed from the training split.

## 3.7 Training Protocol

Adam [@adam] with learning rate 1e-4; mini-batch training with early stopping on **validation
loss** (the `unlabel` split), patience 10, restoring the best-epoch checkpoint. Per the project
convention, only train and validation **loss** are logged per epoch (no per-epoch
classification metrics); final metrics are computed once on the held-out test split. Fixed seed
42. Figure `loss_curves_cmose_ce` shows representative curves and confirms stable convergence
and early stopping.

## 3.8 Evaluation Protocol

**Cross-dataset matrix.** Every model/loss is trained on each of {CMOSE, DAiSEE, Combined} and
evaluated on each of {CMOSE-test, DAiSEE-test, Private} — a 3×3 matrix per configuration.

**Metrics.** Six metrics (Section 2.4): accuracy, macro-accuracy, MAE, macro-MAE, Cohen's
kappa, and QWK; **QWK and macro-accuracy are primary**. Confusion matrices and per-class F1 are
used for error analysis.

**Aggregation.** All ~237 runs are aggregated by `src/analysis/aggregate.py`; figures/tables
are produced by `python -m src.analysis.make_thesis_artifacts` into `outputs/thesis/`.
