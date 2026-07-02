# Defense Speaking Script — Student's Engagement Detection in Online Classes

**Thesis:** Student's engagement detection in online classes
**Author:** Phan Minh Hòa — 20225495
**Audience:** technical committee · **Language:** English

> This script is written to match the **actual `thesis.pptx` deck** (18 content slides + a Thank-you slide and four backup slides). Each entry lists the section chip, the figure/table actually on the slide, the on-slide textbox (verbatim intent), and the spoken lines. Voice is first-person "I", plain scientific register — no figurative language.

---

## Deck map & timing (~11 min)

| # | Slide (on-slide title) | Section | Figure / Table on slide | Time | Mark |
|---|------------------------|---------|-------------------------|------|------|
| 1 | Title | — | logo | 0:10 | |
| 2 | Content | — | — | 0:10 | |
| 3 | Problem Statement (1/3) | 1. Introduction | 4 webcam frames | 0:35 | |
| 4 | Key Challenges (2/3) | 1. Introduction | imbalance table | 0:50 | |
| 5 | Research Gaps & Contributions (3/3) | 1. Introduction | — | 0:40 | ⭐ contributions |
| 6 | Features & Encoders (1/1) | 2. Literature Review | — | 0:35 | express |
| 7 | Datasets (1/4) | 3. Methodology | `dataset_class_distribution_overall.png` | 0:50 | ⭐ private set |
| 8 | Baseline Architectures (2/4) | 3. Methodology | 5-baseline diagrams | 0:30 | |
| 9 | Hybrid Architecture (3/4) | 3. Methodology | `hybrid.png` | 1:00 | ⭐ contribution 1 |
| 10 | Baseline Models — CMOSE in-domain | 4. Experimental Results | `metric_correlation_base.png` | 0:45 | |
| 11 | Baseline Models — Cross-dataset | 4. Experimental Results | `crossdomain_base.png` | 0:50 | ⭐ contribution 3 |
| 12 | Hybrid Ablation — CMOSE in-domain | 4. Experimental Results | Table 5.1 (group marginal) | 0:35 | |
| 13 | Hybrid Ablation — CMOSE in-domain | 4. Experimental Results | `hybrid_ablation_all_metrics.png` | 0:40 | |
| 14 | Hybrid Ablation — Cross-dataset | 4. Experimental Results | `crossdomain_hybrid.png` | 0:50 | ⭐ |
| 15 | Private Dataset Comparison | 4. Experimental Results | `private_per_class_f1.png` | 0:50 | ⭐ climax |
| 16 | Conclusions (1/2) | 5. Conclusions | — | 0:40 | impact |
| 17 | Limitations & Future Work (2/2) | 5. Conclusions | — | 0:35 | impact |
| 18 | Thank you | — | — | — | + backup |

**Story to land:** the metric decides the winner → QWK is the honest metric → baselines transfer poorly (accuracy hides it) → split-and-fuse beats the single encoder → the advantage is widest on the real, unseen private set. Repeat the words **QWK / ordinal agreement**, **transfer**, and **honest**.

---

# 1. INTRODUCTION

## Slide 3 — Problem Statement (1/3)
**FIGURE:** four real webcam frames (different people, rooms, cameras).
**TEXTBOX (on slide):**
- Task: recognize students' engagement level
- Input: 10 s clips from students' webcams
- Output: 4 classes — Highly Engage / Engage / Disengage / Highly Disengage

**SCRIPT:**
> "In an online class the webcam is often the only signal a teacher has about a student, but a teacher watching their own screen cannot monitor every learner. I address one automation task: given a ten-second webcam clip, classify it into one of four ordered engagement levels, from Highly Engage down to Highly Disengage."

---

## Slide 4 — Key Challenges (2/3)
**FIGURE:** per-dataset class-distribution table (imbalance is what needs proof).

| Engagement class | CMOSE | DAiSEE | Private |
|---|---|---|---|
| Highly Engage (HE) | 9.6% | 43.8% | 30.6% |
| Engage (EG) | 69.4% | 50.3% | 58.2% |
| Disengage (DE) | 18.1% | 5.1% | 8.5% |
| Highly Disengage (HD) | 2.8% | 0.7% | 2.7% |

**TEXTBOX (on slide):** Ordinality · Imbalance · Angle Diversity

**SCRIPT:**
> "Three properties make the task hard. First, ordinality: the four labels form an ordered scale, so the size of a mistake matters — confusing Highly Engage with Highly Disengage is far worse than a one-level slip — and plain accuracy cannot see that. Second, imbalance: this table shows each dataset is skewed differently; CMOSE is almost 70% Engage, so a model can look accurate while always predicting the majority class. Third, angle diversity: every learner frames the webcam differently, and the OpenFace features are measured in that camera geometry, so the same expression reaches the model differently. Fitting one dataset says little about the next."

---

## Slide 5 — Research Gaps & Contributions (3/3)
**FIGURE:** none (text).
**TEXTBOX (on slide):**
- *Gaps:* facial features are mixed but encoded as one block; studies mostly use accuracy and cross-dataset generalisation is limited
- *Contributions:* new architecture that divides facial features then fuses them · new private dataset manually labelled from a SOICT online class · generalisation study using QWK instead of accuracy

**SCRIPT:**
> "From the literature I identify two gaps. The mixed facial features are usually pushed through a single encoder, ignoring that they describe different behavioural subsystems; and models are judged by accuracy on one dataset, so their transfer is rarely tested. My thesis makes three contributions that answer these gaps directly: an architecture that splits the facial features and then fuses them, a private test set I labelled from a real SOICT online class, and a cross-dataset generalisation study measured by Quadratic Weighted Kappa rather than accuracy."

---

# 2. LITERATURE REVIEW

## Slide 6 — Features & Encoders (1/1) *(express)*
**FIGURE:** none (two columns: Features / Encoder).
**TEXTBOX (on slide):**
- *OpenFace features:* 709 per frame — gaze, eye landmarks, face landmarks, head pose, action units
- *I3D features:* 1024 per clip, trained on Kinetics-400
- *Encoders:* TCN · LSTM · Transformer

**SCRIPT:**
> "The task is built on two standard feature extractors rather than raw pixels. OpenFace produces 709 numbers per frame describing the face — gaze, eye landmarks, face landmarks, head pose, and action units. These are exactly the five groups my proposed model will use. I3D produces one 1024-dimensional vector per clip capturing motion, from a network trained on Kinetics-400. CMOSE ships both; I extracted the same features for DAiSEE and my private set with the same toolkits. To read these sequences over time I use three standard temporal encoders — a TCN, an LSTM, and a Transformer — and these are the building blocks for both my baselines and my proposed model."

---

# 3. METHODOLOGY

## Slide 7 — Datasets (1/4) ⭐
**FIGURE:** `dataset_class_distribution_overall.png` (stacked class distribution: CMOSE / DAiSEE / Combined).
**TEXTBOX (on slide):**
- *Public datasets:* CMOSE, DAiSEE (and their Combined)
- *Private dataset:* 366 ten-second clips · single annotator · test only

**SCRIPT (slow — contribution 2):**
> "I train on two public datasets, CMOSE and DAiSEE, and on their combination. This figure shows the core difficulty from the previous slide: all three are imbalanced, but in different directions — DAiSEE is dominated by Highly Engage, CMOSE by Engage — so a model that fits one prior meets a different one at test time. I also collected a private dataset: 366 real ten-second webcam clips from a local SOICT online class, labelled by hand, and used only for testing, never for training. It is a small sample of the population a deployed model actually meets. I flag its main limitation now — it has a single annotator — and I return to that under future work."

---

## Slide 8 — Baseline Architectures (2/4)
**FIGURE:** the five single-encoder baseline diagrams (`openface_mlp/tcn/lstm/transformer` + `i3d_mlp`).
**TEXTBOX (on slide):**
- 5 single-encoder baselines: 4 over OpenFace (MLP, TCN, LSTM, Transformer) + 1 over I3D (MLP)
- Used for: choosing the evaluation method · setting the baseline for the proposed model

**SCRIPT:**
> "First the baselines. Four models take the full OpenFace sequence — an MLP, a TCN, an LSTM, and a Transformer — and one MLP takes the pooled I3D vector. Each pushes a single feature block through a single encoder; this is the conventional design. I use these baselines for two purposes: to choose and justify the evaluation metric, and to set the reference level that my proposed model must beat."

---

## Slide 9 — Hybrid Architecture (3/4) ⭐
**FIGURE:** `hybrid.png` (the 709-d vector split into 5 group streams + optional I3D, fused).
**TEXTBOX (on slide):**
- OpenFace features are split into 5 groups
- Each group has its own encoder (TCN / LSTM / Transformer)
- I3D motion stream is optional
- Each stream has an auxiliary head
- Total of 486 configurations

**SCRIPT (slow — contribution 1):**
> "This is the proposed model. A single OpenFace frame mixes signals on different timescales — gaze moves fast and noisily, head pose drifts slowly, action units fire in short bursts — yet the baselines force all 709 features through one encoder. I split the descriptor into five meaningful groups — gaze, eye landmarks, face landmarks, head pose, and action units — and give each group its own encoder, chosen independently from the three families, plus a small auxiliary head so each group is forced to be useful on its own. The I3D motion vector can join as an optional sixth stream, and the group embeddings are fused for the final prediction. Three encoder choices over five groups is 243 configurations, and 486 with and without I3D — I evaluate all of them, so my claims describe the whole family, not one lucky model."

---

# 4. EXPERIMENTAL RESULTS

## Slide 10 — Baseline Models: CMOSE in-domain
**FIGURE:** `metric_correlation_base.png` (correlation of the six metrics across baselines).
**TEXTBOX (on slide):**
- Two groups of metrics appear: {Accuracy, MAE, Cohen's κ} and {Macro-Accuracy, Macro-MAE}
- QWK bridges the two groups

**SCRIPT:**
> "Before comparing any model I fix how I measure. Running all baselines and correlating the six metrics, they fall into two blocks: a micro block — accuracy, micro-MAE, Cohen's kappa — that rewards getting the majority class right, and a macro block — macro-accuracy and macro-MAE — that rewards balanced, per-class performance. The two blocks disagree; in fact accuracy crowns the I3D MLP while QWK crowns the OpenFace TCN. Quadratic Weighted Kappa is the single metric correlated with both blocks — it is chance-corrected and it respects the ordinal scale. So I adopt QWK as the one metric that selects the best model everywhere in the thesis, and report the rest alongside."

---

## Slide 11 — Baseline Models: Cross-dataset ⭐
**FIGURE:** `crossdomain_base.png` (3×3 train×test heatmaps: QWK, macro-acc, macro-MAE, accuracy).
**TEXTBOX (on slide):**
- Accuracy stays high off-diagonal
- QWK collapses to near zero → low transferability
- QWK is chosen as the primary metric

**SCRIPT (slow — contribution 3):**
> "Now the honest test: train on one dataset, evaluate on another. On the diagonal, where the test set matches training, QWK is healthy — around 0.54 on CMOSE. Move off the diagonal, to a dataset the model never saw, and QWK collapses toward zero — the models essentially do not transfer. But look at the accuracy panel in those same off-diagonal cells: it stays high, because a model can score well by predicting the majority class of the new dataset. If I had reported accuracy, as most papers do, I would have hidden a near-total failure. This is exactly why I commit to QWK as the primary metric, and it is the negative finding at the centre of my third contribution."

---

## Slide 12 — Hybrid Ablation: CMOSE in-domain (which encoder per group)
**FIGURE:** Table 5.1 — per-group marginal effect of encoder choice on QWK (in-domain and pooled over unseen targets).
**TEXTBOX (on slide):**
- Feature groups don't show a clear encoder preference
- Head pose lightly prefers the TCN

**SCRIPT:**
> "Turning to the hybrid, the first question is whether the per-group encoder choice matters. Mostly it does not: for four of the five groups the QWK spread across the three encoders is under 0.006 — essentially flat. Only head pose shows a clear preference, for the TCN, and that preference survives the domain shift, so it is real rather than a tuning accident — head motion such as nodding or turning away is short and local, which suits a convolution. The takeaway is that the benefit comes from splitting and fusing, not from searching for a magic configuration, with the TCN a safe default for head pose."

---

## Slide 13 — Hybrid Ablation: CMOSE in-domain (effect of I3D)
**FIGURE:** `hybrid_ablation_all_metrics.png` (OpenFace-only vs Hybrid+I3D across all six metrics; dashed line = QWK-selected baseline).
**TEXTBOX (on slide):**
- I3D motion stream markedly improves performance

**SCRIPT:**
> "This figure shows all 243 configurations on every metric, split into the OpenFace-only family and the I3D-fused family, with the dashed line the QWK-selected baseline. On the QWK panel the I3D-fused family sits clearly above the OpenFace-only one — median 0.553 against 0.522 — and 82% of fused configurations beat the 0.537 baseline, against only 26% without I3D. The best configuration reaches QWK 0.605. I predicted this gain from the earlier analysis that the OpenFace and I3D baselines fail on different clips, so fusing their information should help — and in-domain it clearly does."

---

## Slide 14 — Hybrid Ablation: Cross-dataset ⭐
**FIGURE:** `crossdomain_hybrid.png` (hybrid 3×3 train×test matrix).
**TEXTBOX (on slide):**
- Proposed model outperforms the baseline
- QWK = 0.605 (+0.068, in-domain)
- QWK = 0.379 (+0.094, private set)
- Transferability is still a problem

**SCRIPT (slow):**
> "Out of domain the hybrid matrix has the same shape as the baseline matrix — a healthy diagonal and an off-diagonal collapse — so the transfer problem belongs to the task, not to my model. What changes is that the hybrid beats the baseline in every cell. In-domain the best hybrid reaches QWK 0.605 against the baseline's 0.537, a gain of 0.068. On the unseen private set, trained on the combined data, it reaches 0.379 against 0.285 — a gain of 0.094, wider than the in-domain gain. So the distribution shift amplifies the architecture's advantage rather than eroding it, even though transfer in absolute terms is still an open problem."

---

## Slide 15 — Private Dataset Comparison ⭐
**FIGURE:** `private_per_class_f1.png` (per-class F1: best base — OpenFace Transformer — vs best hybrid, combined-trained).
**TEXTBOX (on slide):**
- Proposed model has higher per-class F1 scores
- Transferability of the HD class remains a challenge under severe class imbalance

**SCRIPT (climax — slow):**
> "This is the decisive comparison: the best baseline against the best hybrid on the private set, both trained on the combined data, broken down by class. The hybrid raises F1 on the populated classes — Engage from 0.52 to 0.70, Highly Engage from 0.49 to 0.56 — and, where the baseline is almost flat, it gives the middle Disengage class real mass, from 0.09 to 0.28. That localises where the +0.094 QWK gain comes from. I am honest about the exception: the rarest class, Highly Disengage, drops to zero F1 — all ten private HD clips are misread. No encoder choice can recover a class the training data barely contains, and that remains the open failure mode of the thesis."

---

# 5. CONCLUSIONS

## Slide 16 — Conclusions (1/2)
**FIGURE:** none (text; the three contributions).
**TEXTBOX (on slide):**
- Architecture — decompose-and-fuse; +0.068 QWK in-domain; interpretable
- Private dataset — 366 clips, the decisive deployment probe
- Generalization — transfer collapses; QWK exposes it; +0.094 on private

**SCRIPT:**
> "Three findings. First, decomposing the face into groups and fusing them beats a single encoder in-domain by 0.068 QWK, and it is interpretable — only head pose needs a specific encoder. Second, the private set of 366 real clips is the decisive deployment probe, because it is the only test on a different population. Third, transfer across datasets collapses, an ordinal metric exposes what accuracy hides, and the architecture's advantage widens to +0.094 exactly on that private set. The contribution is not one new best score; it is evidence that this task should be measured by honest, ordinal agreement, with attention to the rare class."

---

## Slide 17 — Limitations & Future Work (2/2)
**FIGURE:** none (two columns).
**TEXTBOX (on slide):**
- *Limitations:* private dataset (too small to train on / single annotator); modest in-domain gain
- *Future work:* larger, more structured private dataset with a better labelling protocol; target the HD class; learned group encoders

**SCRIPT:**
> "The limitations point to the next work. The private set is small and has a single annotator, so it can serve only as a test set; and the in-domain gain, while real, is modest. For future work I would grow the private set with more annotators and a more structured labelling protocol, target the rare Highly-Disengage class directly — for example with an ordinal loss or resampling — and let the model learn the per-group encoders instead of enumerating all 486 configurations. Thank you — I am happy to take questions."

---

## Slide 18 — Thank you / Q&A + BACKUP

Backup slides carried in the deck (each one figure/table):

| Backup slide | Figure / Table | Use for |
|---|---|---|
| 21 | `base_models_all_metrics.png` | full baseline leaderboard, all six metrics |
| 22 | `crossdomain_delta.png` | best-hybrid − best-base QWK over the 3×3 matrix (every cell positive) |
| 23 | `private_confusion_combined.png` | row-normalised private confusion, base vs hybrid (Engage recall 0.46→0.72; HD→0.00) |
| 24 | `agreement_base_models.png` | pairwise Cohen's κ across baselines (models with similar scores disagree) |

---

## Anticipated Q&A — rehearse

1. **"Why is the in-domain gain so small (+0.068)?"** → Most of the 243-cell design space is insensitive to the encoder choice; the value is robustness under shift (+0.094 on the private set) and an interpretable, transferable design rule — not a new state of the art. *(Backup 22.)*
2. **"Single annotator — isn't the private set unreliable?"** → It is the main limitation. The conclusions rest on QWK trends across many models, not on single clips, and it is still the only test on a different population. Future work adds annotators.
3. **"Why not just ensemble the baselines that disagree?"** → Backup 24 shows the baselines agree only weakly (κ ≈ 0.27–0.39), so their errors are decorrelated; plain majority voting underperforms the best single model. I fuse features instead and leave ensembling to future work. *(Backup 24.)*
4. **"Why develop on CMOSE, not DAiSEE?"** → DAiSEE's in-domain QWK is only 0.166 — too weak to separate architectures from label noise — whereas CMOSE reaches 0.537.
5. **"Why does I3D help in-domain but not under shift?"** → It learns useful but dataset-specific appearance cues; the paired ablation shows +0.060 on seen targets but −0.021 cross-dataset. Fuse I3D when the target is represented in training; rely on the OpenFace groups otherwise.
6. **"Did you compute the features yourself?"** → CMOSE ships OpenFace and I3D precomputed; I extracted both for DAiSEE and the private set with the same toolkits, so the private features are distributionally comparable.
7. **"Only head pose prefers an encoder — is the whole search pointless?"** → The flatness is itself a finding: the TCN is a safe default and the split-and-fuse structure, not the encoder search, is what helps. *(Slide 12 / Table 5.1.)*
8. **"Why five groups and not six — what about shape/PDM?"** → The 709 columns partition into exactly five groups (gaze, eye landmarks, face landmarks, head pose, action units) that sum to 709; the shape/PDM parameters are contained within that partition.

---

## Delivery notes

- Say **"QWK — Quadratic Weighted Kappa, ordinal agreement"** the first two times, for non-CV panellists.
- Name the figure's provenance before reading it: baseline, hybrid, or head-to-head.
- Pre-empt the single-annotator question on slide 7 — do not wait for it.
- Slow down on the ⭐ slides (5, 7, 9, 11, 14, 15). Everything else moves briskly.
- Highest-attention moments: slide 3 (the problem) and slides 14–15 (the private-set result).
- If running long, compress slides 12 and 13 into one spoken beat ("the encoder choice barely matters, but adding I3D clearly helps").
