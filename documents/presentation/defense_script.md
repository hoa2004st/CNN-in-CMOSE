# Defense Deck & Speaking Script — Student's Engagement Detection in Online Classes

**Thesis:** Student's engagement detection in online classes
**Author:** Phan Minh Hòa — 20225495
**Audience:** technical committee · **Language:** English · **Target:** ~11 minutes

> **This script matches the rebuilt deck `20225495_PhanMinhHoa_20252_v2.pptx`**, generated from the rewritten thesis. In the rewrite the **feature-group hybrid model is the central contribution (Chapter 4)** and the single-encoder baseline study is the supporting **Appendix A** that fixes the evaluation protocol and motivates fusion. Every slide's on-slide text (the **TEXTBOX** blocks) is already typed into `…_v2.pptx`; the deck keeps the original theme, fonts, layouts, figures, and the E0–E3 relabelling.
>
> **Ordinal label convention (thesis-wide):** `E0` Highly-Disengage · `E1` Disengage · `E2` Engage · `E3` Highly-Engage. **E0 is the rarest, most-disengaged class.**
>
> Voice in the spoken lines is first-person "I", plain scientific register, no figurative language.

---

## Narrative arc (what the committee must leave with)

Motivation → **two concrete limitations** in prior work (features flattened through one encoder; accuracy on one dataset hides majority-collapse and untested transfer) → **two solution directions** (preserve feature structure; evaluate with QWK across datasets) → **three contributions** (hybrid model, private test set, cross-dataset study). The results walk the protocol: **QWK is the honest metric** → **baselines do not transfer, and accuracy hides it** → the **hybrid beats the baseline in-domain (+0.068 QWK)**, driven by split-and-fuse rather than encoder search → its **advantage widens out of domain, widest on the unseen private set (+0.094)**. Repeat the words **QWK / ordinal agreement**, **transfer**, and **honest**.

---

## Deck map (matches the built `…_v2.pptx`: 23 slides — logo, title, contents, 15 content, thanks, 4 backups)

| Deck # | Slide title | Section | Figure / Table | Time | Mark |
|---|-------------|---------|----------------|------|------|
| 1 | Logo | — | — | — | |
| 2 | Title | — | — | 0:10 | |
| 3 | Contents | — | agenda | 0:10 | |
| 4 | Problem Statement | 1. Introduction | — (text) | 0:40 | |
| 5 | Two Limitations in Prior Work | 1. Introduction | camera-angle frames | 0:50 | ⭐ the gap |
| 6 | Objectives & Contributions | 1. Introduction | — | 0:45 | ⭐ contributions |
| 7 | Features & Encoders | 2. Literature Review | group table | 0:40 | express |
| 8 | Datasets & Imbalance | 3. Methodology | `dataset_class_distribution_overall.png` + Table 3.1 | 0:50 | ⭐ private set |
| 9 | Baseline Models | 3. Methodology | 5 baseline diagrams | 0:30 | |
| 10 | Proposed Hybrid Model | 3. Methodology | `hybrid.png` | 1:00 | ⭐ contribution 1 |
| 11 | Evaluation Protocol: Why QWK | 4. Experimental Results | `metric_correlation_base.png` | 0:45 | |
| 12 | Baselines Do Not Transfer | 4. Experimental Results | `crossdomain_base.png` | 0:50 | ⭐ contribution 3 |
| 13 | Hybrid In-Domain: Encoder Choice | 4. Experimental Results | group-marginal figure | 0:40 | |
| 14 | Hybrid In-Domain: Adding I3D | 4. Experimental Results | `hybrid_ablation_all_metrics.png` | 0:45 | ⭐ contribution 1 |
| 15 | Hybrid: Cross-Dataset | 4. Experimental Results | `crossdomain_hybrid.png` | 0:50 | ⭐ |
| 16 | Private Set: The Deployment Test | 4. Experimental Results | `private_per_class_f1.png` | 0:50 | ⭐ climax |
| 17 | Conclusions | 5. Conclusions | — | 0:40 | impact |
| 18 | Limitations & Future Work | 5. Conclusions | — | 0:35 | impact |
| 19 | Thank you | — | — | — | + backup |

*The agreement / "why fuse the two feature families" point (thesis Appendix A) is folded into the slide 14 talking track and Q&A rather than a standalone slide; its figure `agreement_base_models.png` rides on backup slide 23.*

---

# TITLE & AGENDA

## Slide 2 — Title / Slide 3 — Contents
**Contents (on slide):** 1. Introduction · 2. Literature Review · 3. Methodology · 4. Experimental Results · 5. Conclusions

**SCRIPT:**
> "Good morning. My thesis is on automatic detection of student engagement in online classes. I will state the problem and the two gaps I target, describe the proposed model and the evaluation, then present the results, closing on the deployment test on a private dataset I collected."

---

# 1. INTRODUCTION

## Slide 4 — Problem Statement
**TEXTBOX:**
- Online classes: the webcam is the main student–teacher link
- A teacher facing many video tiles cannot see who has disengaged
- Task: classify a 10-second webcam clip; output 4 ordinal levels
  - E0 – Highly-Disengage · E1 – Disengage · E2 – Engage · E3 – Highly-Engage

**SCRIPT:**
> "In an online class the webcam is often the only signal a teacher has about a student, but a teacher watching dozens of video tiles at once can no longer tell when a learner has stopped attending. This thesis addresses the automatic recognition of student engagement: given a ten-second webcam clip of one student, classify it into one of four ordered levels, from E0, highly disengaged, up to E3, highly engaged. Solving it would let a platform see in real time who is losing focus and, aggregated over a session, give an objective measure of how well the material holds attention."

---

## Slide 5 — Two Limitations in Prior Work ⭐
**FIGURE:** the four camera-angle frames at the foot of the slide (they illustrate the transfer/camera point).
**TEXTBOX:**
- Prior work: encode frames with a facial toolkit, or learn from raw pixels
- **Limitation 1** — the 709 facial features are flattened through one encoder
- **Limitation 2** — single-dataset accuracy hides poor transfer to new cameras

**SCRIPT (⭐ slow — this is the gap):**
> "Prior work follows two families: reduce each frame to a facial-behaviour descriptor with a toolkit such as OpenFace and model the sequence, or learn directly from pixels with a deep video model. Both leave the same two limitations, and these motivate the thesis. First, the features are flattened: a single OpenFace frame is 709 numbers bundling gaze, hundreds of landmark coordinates, head pose, and action units — subsystems that move on completely different timescales — yet they are forced through one encoder. Second, evaluation: studies report accuracy in-domain, but the classes are heavily skewed toward the engaged levels, so a model that just predicts the majority class scores high while learning nothing that transfers — and because facial features are read from the camera's viewpoint, a change of angle at deployment moves the features even when engagement is unchanged. That transfer is almost never measured."

---

## Slide 6 — Objectives & Contributions ⭐
**TEXTBOX:**
- **Objectives:** preserve the structure of the facial features instead of flattening it · evaluate with an ordinal metric (QWK), directly across datasets
- **Contributions:**
  - Hybrid model — split 709 into 5 groups, an encoder per group, optional I3D, auxiliary heads
  - Private dataset — 366 hand-labelled webcam clips, test-only
  - Cross-dataset study on QWK — transfer collapses and accuracy hides it

**SCRIPT (⭐ slow — contributions):**
> "This gives one solution direction for each limitation: preserve the structure of the facial features rather than flattening them, and evaluate with a metric that majority prediction cannot inflate, measured directly across datasets. From these, the thesis makes three contributions. First, a hybrid model that splits the 709 features into five behavioural groups, gives each its own temporal encoder, optionally adds an I3D motion stream, and supervises every stream with a small auxiliary head. Second, a private test set of 366 real webcam clips I collected and hand-labelled, used only for testing. Third, a cross-dataset generalisation study, selected throughout on Quadratic Weighted Kappa — QWK — rather than accuracy."

---

# 2. LITERATURE REVIEW

## Slide 7 — Features & Encoders *(express)*
**TEXTBOX:**
- **OpenFace:** 709 features per frame → gaze 8, eye 280, face 340, head pose 46, action units 35
- **I3D:** 1024 features per clip → motion + appearance, pretrained on Kinetics-400
- **Encoders:** TCN · LSTM · Transformer

**SCRIPT:**
> "The task builds on two standard extractors rather than raw pixels. OpenFace produces 709 numbers per frame, and by meaning these split into exactly five groups — gaze, eye landmarks, face landmarks, head pose, and action units — the same five my model uses. I3D produces one 1024-dimensional vector per clip capturing motion and appearance. To read a sequence over time I use the three standard families — a temporal convolutional network, an LSTM, and a Transformer — which serve both as the single-encoder baselines and as the interchangeable per-group encoders. CMOSE ships both features precomputed; I extracted the same for DAiSEE and my private set."

---

# 3. METHODOLOGY

## Slide 8 — Datasets & Imbalance ⭐
**FIGURE:** `dataset_class_distribution_overall.png` + Table 3.1 (corrected to the E0–E3 order).
**TEXTBOX (table on slide):**

| Class | CMOSE | DAiSEE | Private |
|---|---|---|---|
| E0 | 2.8% | 0.7% | 2.7% |
| E1 | 18.1% | 5.1% | 8.5% |
| E2 | 69.4% | 50.3% | 58.2% |
| E3 | 9.6% | 43.8% | 30.6% |

**SCRIPT (slow — contribution 2):**
> "I train on two public datasets — CMOSE, larger and expert-labelled, and DAiSEE, public but noisier — and on their combination. The table shows the core difficulty: all three are skewed toward the engaged levels, but in different directions — CMOSE concentrates on the single Engage class, DAiSEE splits between Engage and Highly-Engage — so a model that fits one prior meets a different one at test time. The disengaged classes E0 and E1 are rare everywhere, yet they are exactly what a monitoring system must catch. I also collected a private set: 366 real ten-second webcam clips, hand-labelled and used only for testing, sampling the camera angles a deployed model actually meets. Its main limitation, a single annotator, I return to under future work."

---

## Slide 9 — Baseline Models
**FIGURE:** the five single-encoder baseline diagrams.
**TEXTBOX:**
- 5 single-encoder baselines: 4 over OpenFace (MLP, TCN, LSTM, Transformer) + 1 over I3D (MLP)
- One feature block → one encoder → classifier

**SCRIPT:**
> "The baselines are the conventional design: four encoders take the full 709-dimensional OpenFace sequence, and one MLP takes the pooled I3D vector — each pushing a single feature block through a single encoder. In the thesis these form the supporting study in Appendix A: they establish the evaluation protocol and set the reference level the proposed model must exceed."

---

## Slide 10 — Proposed Hybrid Model ⭐
**FIGURE:** `hybrid.png`.
**TEXTBOX:**
- OpenFace split into 5 meaning-based groups
- Each group has its own encoder (TCN / LSTM / Transformer), 64-d
- Optional I3D motion stream
- Auxiliary head per stream
- 243 per-group configs (486 with / without I3D)

**SCRIPT (⭐ slow — contribution 1):**
> "This is the proposed model. Because gaze moves fast and noisily, head pose drifts slowly, and action units fire in short bursts, no single encoder suits all 709 features at once. So I split the descriptor into the five behavioural groups and give each its own encoder, chosen independently from the three families. Each group is encoded to a common 64-dimensional embedding, so a large group like the face landmarks cannot dominate a small one like gaze; the embeddings are concatenated and classified together, and a small auxiliary head on every stream forces each group to be useful on its own. The I3D motion vector can join as an optional sixth stream. Three encoders over five groups is 243 configurations, 486 with and without I3D — I evaluate all of them, so my claims describe the whole family. Reusing the baselines' three encoder families is deliberate: any gain must come from the grouping, not from a stronger encoder."

---

# 4. EXPERIMENTAL RESULTS

> *Deck note:* slides 11–12 present the protocol and baseline evidence (Appendix A in the thesis); slides 13–16 present the proposed hybrid model (Chapter 4 proper). To the committee this is one continuous results story.

## Slide 11 — Evaluation Protocol: Why QWK
**FIGURE:** `metric_correlation_base.png`.
**TEXTBOX:**
- Six metrics form two blocks — micro {Accuracy, MAE, Cohen's Kappa} · macro {Macro-Accuracy, Macro-MAE}
- The blocks disagree — QWK is the only bridge → single primary metric

**SCRIPT:**
> "Before comparing any model I fix how I measure. Scoring the fifteen baseline configurations on six metrics, they fall into two blocks: a micro block — accuracy, MAE, Cohen's kappa — that rewards getting the majority class right, and a macro block — macro-accuracy and macro-MAE — that rewards balanced per-class performance. The blocks disagree: accuracy crowns the I3D MLP, QWK crowns the OpenFace TCN. QWK is the one metric correlated with both — it is chance-corrected and respects the ordinal scale. So QWK becomes the single metric that selects every model, I develop on CMOSE, the only dataset where models rise well above chance, and I train with cross-entropy, which maximises QWK."

---

## Slide 12 — Baselines Do Not Transfer ⭐
**FIGURE:** `crossdomain_base.png`.
**TEXTBOX:**
- On-diagonal QWK healthy (≈ 0.54); off-diagonal collapses to ≈ 0
- Accuracy stays high in the same cells — it hides the collapse
- → QWK exposes it; combined training is the simplest fix

**SCRIPT (⭐ slow — contribution 3):**
> "Now the honest test: train on one dataset, evaluate on another. On the diagonal QWK is healthy — about 0.54 on CMOSE. Move off the diagonal, to a dataset the model never saw, and QWK collapses toward zero — 0.02 from CMOSE to DAiSEE, 0.05 the other way, chance-level agreement. But the accuracy panel in those same cells stays high, because predicting the new dataset's majority class scores well. Had I reported accuracy, as most papers do, I would have hidden a near-total failure. This is the negative finding at the centre of my third contribution, and it is why the whole thesis commits to QWK."

---

## Slide 13 — Hybrid In-Domain: Encoder Choice
**FIGURE:** per-group marginal figure (Table 5.1 / group-marginal chart).
**TEXTBOX:**
- Per-group encoder choice barely matters (QWK spread ≤ 0.006)
- Only head pose prefers an encoder — the TCN (spread 0.022, survives shift)
- → the gain is split-and-fuse, not the encoder search

**SCRIPT:**
> "Turning to the hybrid, the first question is whether the per-group encoder choice matters. Mostly it does not: for four of the five groups the QWK spread across the three encoders is under 0.006 — essentially flat. Only head pose shows a real preference, for the TCN — 0.545 against 0.541 and 0.523, a spread of 0.022 — and it survives the domain shift, so it is signal, not a tuning accident; head motion such as nodding is short and local, which suits a convolution. The takeaway is that the benefit comes from splitting and fusing, not from searching for a magic configuration, with the TCN a safe default."

---

## Slide 14 — Hybrid In-Domain: Adding I3D ⭐
**FIGURE:** `hybrid_ablation_all_metrics.png` (243 configs, ±I3D, all six metrics; dashed = QWK-selected baseline).
**TEXTBOX:**
- I3D fusion lifts the whole family: best QWK 0.605 vs baseline 0.537 (+0.068); median 0.553, 82% beat the baseline

**SCRIPT (⭐):**
> "This figure shows all 243 configurations on every metric, split into the OpenFace-only family and the I3D-enabled family. On the QWK panel the I3D-enabled family sits clearly above the baseline — median 0.553, with eighty-two per cent of configurations beating it — and the best reaches QWK 0.605, a gain of 0.068 over the baseline's 0.537. And I predicted this gain rather than stumbling on it: the agreement analysis in the appendix showed the OpenFace and I3D baselines score almost the same yet fail on different clips — together they are right on 84.7 per cent of clips, seven points above the best single model — so fusing the two feature families should help, and in-domain it clearly does."

---

## Slide 15 — Hybrid: Cross-Dataset ⭐
**FIGURE:** `crossdomain_hybrid.png` (hybrid 3×3 matrix; delta version on backup 22).
**TEXTBOX:**
- Hybrid beats the baseline in every one of the 9 cells
  - in-domain +0.068 (QWK 0.605)
  - private set +0.094 (QWK 0.379) — the widest gap

**SCRIPT (⭐ slow):**
> "Out of domain the hybrid matrix has the same shape — a healthy diagonal and an off-diagonal collapse — so the transfer problem belongs to the task, not to my model. What changes is that the hybrid beats the baseline in every one of the nine cells. The advantage is smallest on the in-domain diagonal, plus-0.068, and largest on the private column, plus-0.094. And the paired ablation shows the I3D stream helps only where the target is seen in training — plus-0.060 there, but neutral to slightly harmful off-domain, because global appearance carries dataset bias — so it is the OpenFace behaviour groups, read in subject-centred units, that carry the out-of-domain gain."

---

## Slide 16 — Private Set: The Deployment Test ⭐
**FIGURE:** `private_per_class_f1.png` (per-class F1: best combined-trained baseline vs best combined-trained hybrid).
**TEXTBOX:**
- Combined-trained hybrid best: QWK 0.379 vs baseline 0.285 (+0.094); F1 up on E1 / E2 / E3
- E0 (Highly-Disengage) collapses to 0 — the open failure mode

**SCRIPT (⭐ climax — slow):**
> "The decisive test is the private set — real clips, used only for testing, so the only lever is the training data. Trained on both datasets, the proposed hybrid reaches QWK 0.379, beating every single-source model and beating the best baseline by 0.094 — a wider margin than in-domain. The per-class F1 localises the gain: the hybrid lifts the Engage class from 0.52 to 0.70 and Highly-Engage from 0.49 to 0.56, and, where the baseline is almost flat, gives the middle Disengage class real mass, from 0.09 to 0.28. I am honest about the exception: the rarest class, E0 Highly-Disengage, drops to zero — all ten private E0 clips are misread. No encoder choice recovers a class the training data barely contains, and that stays the open failure mode."

---

# 5. CONCLUSIONS

## Slide 17 — Conclusions
**TEXTBOX:**
- **Architecture** — decompose-and-fuse beats a single encoder in-domain (+0.068 QWK); interpretable, only head pose needs a specific encoder
- **Private dataset** — 366 hand-labelled clips, the decisive deployment probe
- **Generalization** — transfer collapses; QWK exposes what accuracy hides; the advantage widens to +0.094 on the private set

**SCRIPT:**
> "Three findings. First, decomposing the face into groups and fusing them beats a single encoder in-domain by 0.068 QWK, and it is interpretable — only head pose needs a specific encoder. Second, the private set of 366 real clips is the decisive deployment probe, the only test on a different population. Third, transfer across datasets collapses, an ordinal metric exposes what accuracy hides, and the architecture's advantage widens to plus-0.094 exactly on that private set. The contribution is not one new best score; it is evidence that this task should be measured by honest, ordinal agreement, with attention to the rare disengaged class."

---

## Slide 18 — Limitations & Future Work
**TEXTBOX:**
- **Limitations:** private set too small to train on, single annotator (test-only); E0 not recovered; in-domain gain modest; single seed / fixed hyper-parameters
- **Future work:** target the E0 class directly (ordinal loss, resampling); larger multi-annotator private set with a structured protocol; explicit domain adaptation; learn the group encoders (vs enumerating 486 configs)

**SCRIPT:**
> "The limitations point to the next work. The private set is small and single-annotator, so it can only be a test set; the in-domain gain is modest; and the rarest E0 class is not recovered. For future work I would target E0 directly with an ordinal loss or resampling, grow the private set with multiple annotators and a structured labelling protocol, add explicit domain adaptation for the transfer gap, and let the model learn the per-group encoders rather than enumerating all 486 configurations. Thank you — I am happy to take questions."

---

## Slide 19 — Thank you / Q&A + BACKUP

Backup slides carried in the deck:

| Backup | Figure / Table | Use for |
|---|---|---|
| 20 | `base_models_all_metrics.png` | full baseline grid, 5 models × 3 losses, six metrics |
| 21 | `crossdomain_delta.png` | best-hybrid − best-baseline QWK over the 3×3 matrix (every cell positive) |
| 22 | `private_confusion_combined.png` | row-normalised private confusion, baseline vs hybrid (E0 → 0.00) |
| 23 | `agreement_base_models.png` | pairwise Cohen's κ across the 15 baselines (similar scores, decorrelated errors) — the "why fuse" evidence |

---

## Anticipated Q&A — rehearse

1. **"Why is the in-domain gain only +0.068?"** → Most of the 243-cell space is insensitive to the encoder; the value is robustness under shift (+0.094 on the private set) and an interpretable design rule, not a new state of the art. *(Backup 21.)*
2. **"Single annotator — isn't the private set unreliable?"** → It is the main limitation; conclusions rest on QWK trends across many models, and it is still the only test on a different population. Future work adds annotators.
3. **"Why not just ensemble the disagreeing baselines?"** → Their errors are decorrelated (κ ≈ 0.27–0.39), so plain majority voting (73.5%) underperforms the best single model (77.3%). I fuse features, not finished predictions. *(Backup 23.)*
4. **"Why develop on CMOSE, not DAiSEE?"** → DAiSEE's in-domain QWK is only 0.166 — too weak to separate architecture from label noise — versus 0.537 on CMOSE.
5. **"Why does I3D help in-domain but not under shift?"** → It learns dataset-specific appearance; paired ablation +0.060 on seen targets, −0.021 on the public cross-corpus cells. Fuse I3D only when the target is represented in training.
6. **"Did you compute the features yourself?"** → CMOSE ships them precomputed; I extracted OpenFace and I3D for DAiSEE and the private set with the same toolkits, so features are comparable.
7. **"Only head pose prefers an encoder — is the search pointless?"** → The flatness is the finding: the TCN is a safe default and the split-and-fuse structure, not the search, is what helps. *(Slide 13 / Table 5.1.)*
8. **"Why five groups?"** → The 709 columns partition into exactly five meaning-based groups (8 / 280 / 340 / 46 / 35) that sum to 709.
9. **"Is E0 the most engaged or least?"** → E0 is Highly-*Disengage*, the rarest at 2–3%; the scale runs E0 disengaged up to E3 highly engaged.

*A deeper, equation-level question bank (QWK weights, squared-EMD loss, attention, the leak-free `unlabel` validation split, counting 243/486/972/1500) is in `defense_questions.html`.*

---

## Delivery notes

- Say **"QWK — Quadratic Weighted Kappa, ordinal agreement"** the first two times, for non-CV panellists.
- Name each figure's provenance before reading it: baseline, hybrid, or head-to-head.
- Pre-empt the single-annotator question on slide 8 — do not wait for it.
- Slow down on the ⭐ slides (5, 6, 10, 12, 14, 15, 16). Everything else moves briskly.
- Highest-attention moments: slide 5 (the two limitations) and slides 15–16 (the private-set result).
- If running long, compress slides 13 and 14 into one spoken beat ("the encoder choice barely matters, but adding I3D clearly helps") and lean on slide 16.
