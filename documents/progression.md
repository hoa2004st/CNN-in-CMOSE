# Thesis Progress Tracker
## Cross-Domain Engagement Recognition Without Target Labels

**Last updated**: _fill in date_  
**GPU**: vast.ai — _fill in instance type_  
**Overall status**: 🔴 Not started

---

## Quick Status Board

| Phase | Name | Status | Blocker |
|---|---|---|---|
| 0 | Environment & Data Setup | 🔴 Not started | — |
| 1 | CMOSE Baseline Reproduction | 🔴 Not started | Needs Phase 0 |
| 2 | Dataset Comparison (Feature-Space) | 🔴 Not started | Needs Phase 1 features |
| 3 | Domain Shift Analysis | 🔴 Not started | Needs Phase 1 model |
| 4 | Pseudo-Labeling | 🔴 Not started | Needs Phase 3 |
| 5 | Naive vs Baseline Comparison | 🔴 Not started | Needs Phase 1 |
| 6 | Proposed Pipeline (Optional) | ⚪ Optional | Needs Phase 3–4 |

**Status legend**: 🔴 Not started · 🟡 In progress · 🟢 Done · ⛔ Blocked · ⚪ Optional/Skipped

---

## Phase 0 — Environment & Data Setup

**Goal**: All features extractable, project structure in place, vast.ai instance confirmed.

- [ ] **0.1** vast.ai instance rented and SSH access confirmed
- [ ] **0.2** Conda/venv environment created with all dependencies installed
- [ ] **0.3** OpenFace binary compiled and tested on one sample clip
- [ ] **0.4** I3D weights downloaded (Kinetics-400 pretrained RGB stream)
- [ ] **0.5** CMOSE assumptions verified: `data/CMOSE/openface-features/secondFeature/*.csv`, `data/CMOSE/labels.csv`, and `data/CMOSE/final_data_1.json` available
- [ ] **0.6** Private clips verified under `data/private/clips/*.mp4`
- [ ] **0.7** `labels.csv` for CMOSE verified/created with `clip_id, label, split` columns
- [ ] **0.8** Build `data/private/accepted.csv` after OpenFace QA with columns: `clip_id,clip_path,openface_csv,is_accepted,reject_reason`
- [ ] **0.9** One end-to-end test: materialize CMOSE I3D + extract private OpenFace/I3D on accepted clips, verify expected shapes

**Notes / blockers**:
> _Fill in as you go_

---

## Phase 1 — CMOSE Baseline Reproduction

**Goal**: Reproduce video-only baseline. Target: Acc ≥ 75%, AvgAcc ≥ 53% on CMOSE test set.

### 1a — Feature Extraction

- [ ] **1a.1** Validate CMOSE OpenFace feature inventory (`data/CMOSE/openface-features/secondFeature`) and missing-file report
- [ ] **1a.2** Materialize/validate CMOSE I3D `.npy` features from `final_data_1.json`
- [ ] **1a.3** Run `extract_openface.py` on all private clips
- [ ] **1a.4** Build/update `accepted.csv` and exclude rejected private clips
- [ ] **1a.5** Run `extract_i3d.py` on accepted private clips only
- [ ] **1a.6** Sanity check: plot AU12 distribution for a sample of CMOSE clips, confirm it varies with engagement labels

### 1b — Model Implementation

- [ ] **1b.1** Implement `TCN` block (from locuslab/TCN or from scratch)
- [ ] **1b.2** Implement `MLP1` (I3D → attention logits)
- [ ] **1b.3** Implement `MLP2` (I3D projection)
- [ ] **1b.4** Implement `MLP3` (normalized FC score head)
- [ ] **1b.5** Implement full `CMOSEBaseline` forward pass — unit test with random input
- [ ] **1b.6** Removed: paper-baseline ranking-loss implementation is no longer part of this repo
- [ ] **1b.7** Removed: paper-baseline score pool is no longer part of this repo
- [ ] **1b.8** Removed: paper-baseline multi-margin loss is no longer part of this repo

### 1c — Training

- [ ] **1c.1** Set up `DataLoader` for CMOSE (train split)
- [ ] **1c.2** First training run — 100 epochs, confirm loss is decreasing
- [ ] **1c.3** Full training run — 1200 epochs with CosineAnnealing
- [ ] **1c.4** Evaluate on CMOSE val set every 50 epochs, save best checkpoint
- [ ] **1c.5** Final evaluation on CMOSE test set

**Results checkpoint**:

| Metric | Paper | Reproduced |
|---|---|---|
| Overall Accuracy | 77.48% | |
| Average Accuracy | 60.94% | |
| HD Recall | ~high | |
| DE Recall | ~high | |
| EG Recall | ~high | |
| HE Recall | ~high | |

> _Paper-baseline reproduction code has been removed from this repository._

### 1d — Naive Model Baselines

- [ ] **1d.1** Train MLP on OpenFace features (CMOSE train, evaluate on test)
- [ ] **1d.2** Train LSTM on OpenFace
- [ ] **1d.3** Train TCN on OpenFace
- [ ] **1d.4** Train Transformer on OpenFace
- [ ] **1d.5** Train MLP on I3D features
- [ ] **1d.6** Train MLP on concatenated TCN(OpenFace) + I3D
- [ ] **1d.7** Fill in comparison table in `implementation_spec.md` Phase 5

**Notes / blockers**:
> _Fill in as you go_

---

## Phase 2 — Dataset Comparison (Feature-Space)

**Goal**: Quantify the domain gap between CMOSE and private dataset at the feature level — no model needed.

- [ ] **2.1** Compute per-feature mean/std for OpenFace features: CMOSE vs private
- [ ] **2.2** Compute Wasserstein distance for each of 4 feature groups (gaze, head pose, AU intensities, AU presence)
- [ ] **2.3** Plot box plots for top 10 most different features
- [ ] **2.4** UMAP of I3D features: all CMOSE clips + accepted private clips, colored by dataset
- [ ] **2.5** Compute cosine distance between CMOSE centroid and private dataset centroid (I3D space)
- [ ] **2.6** Compute CMOSE class centroids in I3D space — which class centroid is closest to private dataset center?
- [ ] **2.7** Report Domain Gap Score (composite)
- [ ] **2.8** Write 1-paragraph interpretation: "The gap is primarily driven by ___"

**Key finding** (fill in):
> _Which modality shows larger gap: OpenFace or I3D? Is the gap larger than the gap between CMOSE and DAiSEE/EngageWild (from paper Table 6)?_

**Notes / blockers**:
> _Fill in as you go_

---

## Phase 3 — Domain Shift Analysis

**Goal**: Apply trained CMOSE model to accepted private clips (`is_accepted=1`), diagnose what breaks.

- [ ] **3.1** Run CMOSE baseline inference on all accepted private clips — save scores and predicted labels
- [ ] **3.2** Plot histogram of raw scores s ∈ [−1, 1]: private vs CMOSE test
- [ ] **3.3** Report predicted class distribution on private clips
- [ ] **3.4** Compute mean prediction entropy on private clips vs CMOSE test clips
- [ ] **3.5** Extract `X_attn` attention weights for private clips — are they uniform or structured?
- [ ] **3.6** Extract `X_vis` embeddings for CMOSE test + private clips
- [ ] **3.7** UMAP of embeddings: CMOSE (colored by true label) + private (colored by predicted label)
- [ ] **3.8** Compute distance from each private clip embedding to each CMOSE class centroid
- [ ] **3.9** Identify "unstable classes": which classes have private clips falsely mapped to them?
- [ ] **3.10** Run same analysis with OpenFace-only model (TCN without I3D) — compare domain shift severity

**Key findings** (fill in):
> _Which class dominates predictions on private data?_  
> _Is entropy higher on private vs CMOSE test? By how much?_  
> _Do attention weights degrade on private data?_  
> _OpenFace-only vs I3D-only: which shows less domain shift?_

**Notes / blockers**:
> _Fill in as you go_

---

## Phase 4 — Pseudo-Labeling

**Goal**: Try 3 strategies, evaluate stability, report which (if any) helps.

### Strategy 1 — Confidence Threshold Self-Training

- [ ] **4.1.1** Implement `is_confident(score, margin=0.3)` function
- [ ] **4.1.2** Run on private clips — how many clips pass the threshold?
- [ ] **4.1.3** Fine-tune model on CMOSE + confident pseudo-labeled clips (250 epochs)
- [ ] **4.1.4** Re-evaluate on CMOSE test set (catastrophic forgetting check)
- [ ] **4.1.5** Repeat for 2 more iterations, track: # pseudo-labels, CMOSE acc, private entropy
- [ ] **4.1.6** Try margins: 0.2, 0.3, 0.4 — report sensitivity

**Results**:
> _Iteration 1: N pseudo-labels = ___, CMOSE acc = ___, private entropy = ___  
> Iteration 2: N = ___, CMOSE acc = ___, private entropy = ___  
> Iteration 3: N = ___, CMOSE acc = ___, private entropy = ____

### Strategy 2 — Teacher-Student (EMA)

- [ ] **4.2.1** Implement EMA teacher (momentum=0.999)
- [ ] **4.2.2** Implement consistency loss between student score and teacher score on private clips
- [ ] **4.2.3** Train student with `λ=0.1` consistency weight for 250 epochs
- [ ] **4.2.4** Evaluate CMOSE retention and private entropy change
- [ ] **4.2.5** Try `λ ∈ {0.05, 0.1, 0.2}` — report sensitivity

**Results**:
> _Best λ = ___, CMOSE acc = ___, private entropy = ___  
> Student-teacher agreement at end of training = ___%_

### Strategy 3 — k-NN Label Propagation

- [ ] **4.3.1** Extract `X_vis` embeddings for CMOSE train + private clips
- [ ] **4.3.2** Fit kNN classifier (k=5, cosine metric) on CMOSE embeddings
- [ ] **4.3.3** Predict pseudo-labels for all accepted private clips
- [ ] **4.3.4** Report confidence distribution (`predict_proba` max values)
- [ ] **4.3.5** Try k ∈ {3, 5, 10, 15} — report sensitivity
- [ ] **4.3.6** Compute per-class pseudo-label distribution

**Results**:
> _Best k = ___, pseudo-label distribution: HD=_%, DE=_%, EG=_%, HE=__%_

### Cross-Strategy Analysis

- [ ] **4.4.1** Compute agreement rate between Strategy 1 and Strategy 3 labels
- [ ] **4.4.2** Compute agreement rate between Strategy 2 and Strategy 3 labels
- [ ] **4.4.3** Identify "consensus clips": clips where all 3 strategies agree — what % of accepted private clips?
- [ ] **4.4.4** Report label stability: run each strategy 3x with different seeds, measure % of stable assignments

**Consensus pseudo-labels**:
> _N clips with 3-way agreement = ___ out of accepted private clips  
> Class distribution of consensus clips: HD=__, DE=__, EG=__, HE=___

**Notes / blockers**:
> _Fill in as you go_

---

## Phase 5 — Results & Discussion

**Goal**: Write up all findings into clear thesis discussion.

- [ ] **5.1** Final comparison table (naive models vs CMOSE baseline on CMOSE test)
- [ ] **5.2** Discussion: "Which features transfer better?" — supported by Phase 3 findings
- [ ] **5.3** Discussion: "Which classes are unstable?" — supported by Phase 3 findings
- [ ] **5.4** Discussion: "Does pseudo-labeling help?" — supported by Phase 4 findings
- [ ] **5.5** Discussion: "Limitations of no target labels" — explicit and honest, reference indirect metrics
- [ ] **5.6** Thesis claim refined with actual numbers (fill in template from thesis_direction.md)

**Notes / blockers**:
> _Fill in as you go_

---

## Phase 6 — Proposed New Pipeline (Optional)

**Goal**: One motivated improvement, implementable in ≤1 week.

**Chosen option**: _fill in after Phase 3-4 (see thesis_direction.md for options)_

- [ ] **6.1** Implement proposed pipeline
- [ ] **6.2** Run on private clips — compare pseudo-label stability to Phase 4 baseline
- [ ] **6.3** Report delta metrics vs CMOSE baseline
- [ ] **6.4** Discuss why it works / doesn't work — ground in Phase 2-3 findings

---

## Experiment Log

_Add an entry every time you run a significant experiment. Keep this honest — failed runs are valuable._

| Date | Phase | Experiment | Result | Notes |
|---|---|---|---|---|
| | | | | |

---

## Key Numbers (Fill in as completed)

| Metric | Value | Date confirmed |
|---|---|---|
| CMOSE baseline Acc | | |
| CMOSE baseline AvgAcc | | |
| Best naive model Acc on CMOSE | | |
| Best naive model AvgAcc on CMOSE | | |
| Domain Gap Score | | |
| Private dataset prediction entropy (before PL) | | |
| Private dataset prediction entropy (after best PL) | | |
| Strategy 1/2/3 agreement rate | | |
| Consensus pseudo-labels count | | |

---

## Decisions Log

_Record key decisions made during implementation so you can justify them in the thesis._

| Date | Decision | Reason |
|---|---|---|
| | Using video-only branch of CMOSE (no audio) | Private dataset has no audio track |
| | Paper-baseline training implementation removed from this repo | Keep current work focused on retained models |
| | | |
