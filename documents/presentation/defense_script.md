# Defense Presentation — Slide Design & Speaking Script

**Thesis:** Student's engagement detection in online classes
**Author:** Phan Minh Hòa
**Audience:** technical committee · **Language:** English · **Target length:** ~10–11 min

---

## How to use this file

- Each slide has: **Visual** (what to put on screen, which figure/table, layout), a **Provenance** tag for any chart/table, and **Script** (the core line to say).
- The deck **follows the thesis chapter order** (Intro → Literature → Methodology → Baseline → Hybrid → Conclusions). Most slides move fast; the **three contributions are marked ⭐ DWELL** — slow down, make eye contact, let the result land.
- Scripts are deliberately short — **only the core idea**. Anyone who wants detail can read the thesis. Say them in your own words; keep the numbers and the logical beats.

### Chart/table provenance legend (use these consistently)

| Tag | Meaning |
|-----|---------|
| 🟦 **BASELINE** | Produced from the 5 monolithic baseline models (Chapter 4). |
| 🟩 **HYBRID** | Produced from the proposed semantic-group hybrid, 243/486 configs (Chapter 5). |
| 🟪 **BASELINE vs HYBRID** | Direct comparison of best baseline against best hybrid. |
| ⬜ **DATA / PROTOCOL** | Dataset statistics or evaluation protocol — model-independent (Chapter 3). |
| 🟨 **ARCHITECTURE** | A design/architecture diagram, not a result. |

> Say the provenance out loud the first time a chart appears: *"this summarises the five baselines"* vs *"this is the proposed hybrid."* The committee must never be unsure which model a number describes.

---

## Timing budget (target ~10:30)

| # | Slide | Chapter | Time | Running | Mark |
|---|-------|---------|------|---------|------|
| 1 | Title | — | 0:15 | 0:15 | |
| 2 | Table of contents | — | 0:15 | 0:30 | |
| 3 | Problem Statement | Ch.1 | 0:35 | 1:05 | |
| 4 | Key Challenges: Ordinality, Imbalance, Angle Diversity | Ch.1 | 0:50 | 1:55 | |
| 5 | Research Gaps & Contributions | Ch.1 | 0:35 | 2:30 | |
| 6 | Feature Extraction & Temporal Architectures | Ch.2 | 0:30 | 3:00 | express |
| 7 | Datasets & the Private Evaluation Set | Ch.3 | 0:55 | 3:55 | ⭐ DWELL (private set) |
| 8 | Semantic-Group Decomposition & Hybrid Architecture (Proposed) | Ch.3 | 1:15 | 5:10 | ⭐ DWELL (architecture) |
| 9 | Evaluation Protocol & Metrics | Ch.3/4 | 0:35 | 5:45 | |
| 10 | Baselines: In-Domain Results & Prediction Agreement | Ch.4 | 0:40 | 6:25 | |
| 11 | Encoder Ablation & In-Domain Comparison | Ch.5 | 0:50 | 7:15 | ⭐ DWELL (architecture) |
| 12 | Cross-Dataset Generalization | Ch.4/5 | 0:45 | 8:00 | ⭐ DWELL (generalization) |
| 13 | Private-Set Generalization | Ch.5 | 0:55 | 8:55 | ⭐ DWELL (private set) |
| 14 | Conclusions | Ch.6 | 0:30 | 9:25 | impact |
| 15 | Limitations & Future Work | Ch.6 | 0:35 | 10:00 | impact |
| 16 | Thank you / Q&A + backup | — | — | — | |

The five ⭐ DWELL slides (7, 8, 11–13) are your contributions — non-negotiable. Everything else can move briskly.

---



## Slide 1 — Title (0:15)

**Visual:** HUST logo (image). Thesis title: "Student's engagement detection in online classes". Author: "Phan Minh Hoa - 20225495". Slide number.

**Script:**
> "Good morning everyone. I’m Phan Minh Hòa. Today, I will present to you my thesis on “Student’s engagement detection in online classes”. Without wasting your time, let’s go right into the main content."
"

---

## Slide 2 — Table of contents (0:15)

**Visual:** Heading "Content". Numbered list:
1. Introduction
2. Literature Review
3. Methodology
4. Experimental Results
5. Conclusions

**Script:**
> "Here's the outline of my presentation."

---
# CHAPTER 1 — INTRODUCTION

## Slide 3 — Problem Statement (0:35)

**Visual:** Section tag "1. Introduction"; title "Problem Statement". One illustration figure plus three example webcam screenshots. Text:
- "Recognize students' engagement level"
- "Input: 10s clips from students' webcams"
- "Output: 4 classes: HE / E / D / HD"

**Script:**
> "In online classes, the webcam became the teacher's only window onto the learner. When teaching and focusing on their own screen sharing, teachers realistically cannot read the room and decide they should change the pace or repeat the points they just made. To assist online teachers our thesis focus on one automation task: classify ten-second clips into four ordered engagement levels, from Highly-Disengaged to Highly-Engaged."

---

## Slide 4 — Key Challenges: Ordinality, Imbalance, Angle Diversity (0:50)

**Visual:** Section tag "1. Introduction"; title "Key Challenges". Three labelled items, each with a small figure:
1. Ordinality
2. Imbalance
3. Angle Diversity

**Script:**
> "This task innately have three key challenges. One: the ordinal scale forms a *stair* — mistaking a disengaged student for a highly-engaged one is a disaster, off-by-one is fine, but plain accuracy can't tell those apart. Two: the dataset is highly imbalance for example, CMOSE dataset have almost 70% of the clips labeled 'Engaged' while only under 3% labeled highly-disengaged. A model can have high accuracy by always guessing 'Engaged' and never doing its job. Three: every learner frames their webcam differently — a laptop looking up, a monitor camera at eye level, a phone off to the side — so the same expression reaches the model at a different angle and head pose. The OpenFace features are measured in that geometry, so the angle alone shifts them even when the engagement is identical. Doing well on one dataset's camera setups tells us little about the angles the model will actually meet."

---

## Slide 5 — Research Gaps & Contributions (0:35)

**Visual:** Section tag "1. Introduction"; titles "Research Gaps" and "Contributions".

Research Gaps (text):
- All facial features are put into a model / pipeline as a block.
- Studies mostly use the accuracy metric, both for in-domain and cross-dataset evaluation.

Contributions (text):
- New architecture that divides the facial features, then fuses them.
- New private dataset, manually labelled from a SOICT online class.
- Generalization study using QWK instead of accuracy.

**Script:**
> "Looking across the engagement-detection literature, two habits keep coming back. First, almost every prior study pours all the facial features into a single encoder — gaze, head pose, muscles, all mixed together — so the natural structure of the face is thrown away. Second, the studies that do test across datasets mostly report plain accuracy — which, as we just saw, can be fooled by the majority class and so says little about real generalization — and none of them looks at a Vietnamese academic setting. These two habits are the gaps my work targets — and my three contributions answer them: an architecture that keeps the structure of the features, a private test set I collected and labelled myself in a local online class, and an honest cross-dataset test measured by ordinal agreement rather than accuracy."

---

# CHAPTER 2 — LITERATURE REVIEW

## Slide 6 — Feature Extraction & Temporal Architectures (0:30) — *express*

**Visual:** Two text columns.
- Features: "OpenFace -> 709 features per frame; I3D -> 1024 features per clip." Note: "CMOSE ships these; DAiSEE and the private set extracted by us."
- Encoders: TCN, LSTM, Transformer.

**Provenance:** 🟨 ARCHITECTURE.

**Script:**
> "Work on this task almost always starts the same way — not from raw pixels, but from two off-the-shelf feature extractors. OpenFace is the standard for the face: 709 numbers per frame — gaze, landmarks, head pose, muscle activations. I3D is the standard for motion: one 1024-number vector per clip. These two are so established that CMOSE ships them pre-computed, and I'll later use the very same tools to extract features for DAiSEE and my private set.Then, to read the sequence over time, the literature converges on three temporal encoders — TCN, LSTM, and Transformer. These are also the common building blocks I build my proposed model on."

---

# CHAPTER 3 — METHODOLOGY

## Slide 7 — ⭐ DWELL — Datasets & the Private Evaluation Set (0:55)

**Visual:** Figure `dataset_class_distribution_overall.png` (left). Private-set details as text (right): "366 clips, 10s, OpenFace + I3D by us, single annotator, TEST ONLY." Optional labelling-interface screenshot. Table T3_1 as backup.

**Provenance:** ⬜ DATA / PROTOCOL.

**Script (slow down — contribution #2):**
> "I train on two public datasets — CMOSE and DAiSEE — and the combination of two together. This figure shows the main problem: all three are imbalanced in different ways, so a model that fits one meets a different one at test time. I also create a private dataset consist of 366 real webcam clips, cut from real online class of SOICT, labelled manually, and used solely for testing — never for training. This dataset is a small sample of the actual population that models will meet when deployed in real life."

*(Pre-empting the single-annotator question here is deliberate.)*

---

## Slide 8 — ⭐ DWELL — Semantic-Group Decomposition & Hybrid Architecture (1:15)

**Visual:** Two-panel diagram. Left ("Standard"): "709 -> one encoder -> class." Right ("Proposed"): Figure `hybrid.png` — the 709-feature vector splitting into 5 streams (gaze, eye landmarks, face landmarks, head pose, action units), each with its own encoder + small head, plus I3D as a sixth motion stream, fused to the final class. Legend listing the three encoder options (TCN, LSTM, Transformer).

**Provenance:** 🟨 ARCHITECTURE (the proposed model — contribution #1).

**Script (slow down — contribution #1):**
> "We designed 5 baseline models, 2 MLP model, each for a set of feature and 3 temporal decoder for OpenFace features alone. One OpenFace frame mixes very different signals that change at different speeds — eyes move fast and noisily, the head turns slowly, muscles fire in short bursts. Putting all 709 into one encoder, as everyone does, forces the same time model on all of them. So I split the face into five meaningful groups — gaze, eye landmarks, face landmarks, head pose, action units — and give each its own encoder: a TCN, Transformer, or LSTM, chosen per group, producing a 64-number embedding. I3D can join as a sixth, motion stream. Every stream also gets its own small head, so each group has to be useful on its own, not only in the mix. Three encoders over five groups gives 243 versions — and I test all of them, with and without I3D: 486 models in total. That lets me make claims about the whole family, not one lucky model.
"

---

## Slide 9 — Evaluation Protocol & Metrics (0:35)

**Visual:** Figure `base_models_accuracy_vs_qwk.png` — two panels, Accuracy (left) and QWK (right), base models x losses, in-domain CMOSE. Caption: "Same models, two metrics, different winner: Accuracy -> I3D MLP (0.77); QWK -> OpenFace TCN (0.54)."

**Provenance:** 🟦 BASELINE.

**Script:**
> "Before comparing models, I fixed how I measure. Accuracy is what almost every prior study reports, so I keep it on screen for reference — but, as the rest of the thesis will show, it is not a reliable metric for this task. Here is why, on the very same baseline models: rank them by accuracy and the I3D MLP wins with 0.77; rank them by QWK — ordinal agreement — and a different model, the OpenFace TCN, wins at 0.54, while that same I3D MLP drops to 0.52. The metric itself changes the answer — so the choice of metric is not innocent, and I commit to one. **The single metric that selects the best model, everywhere in this thesis, is QWK** — and only QWK. Whenever I say 'best' — best baseline, best hybrid, best per cell — I mean the highest QWK, with ties broken by order, not by a second metric. Macro-accuracy and macro-MAE are reported alongside as secondary checks, and plain accuracy stays on screen only for comparison with prior work; none of them ever picks the winner. This one-metric rule is also what later exposes a failure that accuracy hides."

---

# CHAPTER 4–5 — EXPERIMENTAL RESULTS

## Slide 10 — Baselines: In-Domain Results & Prediction Agreement (0:40)

**Visual:** Figure `base_models_all_metrics.png` (left) - best baseline QWK 0.537. Figure `agreement_base_models.png` (right), caption: "different errors -> best-picker 97.6% vs 77.3% best single."

**Provenance:** 🟦 BASELINE.

**Script:**
> "The baseline scores, in-domain on CMOSE: the best single model reaches QWK (Quadratic Weighted Kappa) 0.537 — the bar to beat. The second plot is why I add I3D: the best OpenFace model and the I3D model disagree with each other more than the OpenFace models disagree among themselves — they get different clips right. They make different errors, so I join them. A perfect picker over the two would reach 97.6% versus 77.3% — real room to improve, which I leave as future work."

---

## Slide 11 — ⭐ DWELL — Encoder Ablation & In-Domain Comparison (0:50)

**Visual:** Figure `hybrid_ablation_all_metrics.png`. Caption: "Best baseline 0.537 -> hybrid median 0.553, best 0.605; 82% beat baseline." Note: "Only head pose prefers a specific encoder -> TCN."

**Provenance:** 🟩 HYBRID.

**Script (slow down — architecture result):**
> "The architecture, in-domain — all 243 versions. As a *family*, the I3D hybrid sits above the baseline: median QWK (Quadratic Weighted Kappa) 0.553 versus 0.537, 82% of versions beat the bar, best 0.605. To be honest, that best gain — plus 0.068 — is small. The stronger finding is a design rule: four of the five groups don't care which encoder they get; only *head pose* clearly prefers one — the TCN — because head motion is short and local in time. So the message isn't a magic model; it's that *split-and-join* helps, and a TCN is a safe default."

---

## Slide 12 — ⭐ DWELL — Cross-Dataset Generalization (0:45)

**Visual:** Figures `crossdomain_base.png` (left) and `crossdomain_hybrid.png` (right). Caption: "Off-diagonal QWK approx 0 (CMOSE->DAiSEE 0.02) - yet accuracy looks fine."

**Provenance:** 🟦 BASELINE matrix (left) + 🟩 HYBRID matrix (right). Label both.

**Script (slow down — contribution #3):**
> "The key negative finding — my third contribution. Move any model to a dataset it did not train on, and it falls apart: the off-diagonal QWK (Quadratic Weighted Kappa) drops to near zero — CMOSE to DAiSEE is 0.02. These models don't transfer. And on those same cells, *accuracy still looks fine* — the majority-guessing trap again. If I had reported accuracy like most papers, I would have hidden a complete failure. The simplest fix that works best — train on the datasets pooled together — sets up the final result."

---

## Slide 13 — ⭐ DWELL — Private-Set Generalization (0:55)

**Visual:** Figure `private_confusion_combined.png` (best baseline vs best hybrid). Caption: "Private set - Hybrid QWK 0.379 vs baseline 0.285 (+0.094); hybrid wins all 9 cells."

**Provenance:** 🟪 BASELINE vs HYBRID.

**Script (the climax — slow down most):**
> "Now all three contributions come together on the private set — the unseen, real-student test. Best baseline against best hybrid, on my own data. First, the hybrid wins all *nine* train-by-test cells — not one lucky case. Second, because the set is test-only, my only choice is the training data: training on the pooled datasets with the hybrid gives the best result in the whole thesis — QWK (Quadratic Weighted Kappa) 0.379 versus 0.285. That gap, plus 0.094, is *bigger* than in-domain. The architecture proves its worth exactly where it's hardest — on real, unseen students. That's the practical heart of the thesis."

---

# CHAPTER 6 — CONCLUSIONS

## Slide 14 — Conclusions (0:30) — *impact*

**Visual:** Three points mirroring slide 5:
1. Architecture - decompose-and-fuse; +0.068 in-domain; interpretable.
2. Private dataset - 366 clips, the decisive deployment probe.
3. Generalization - transfer collapses; QWK exposes it; +0.094 on private.

**Script:**
> "To conclude — three findings that compound. This work *showed* that decomposing the face into semantic groups and fusing them beats putting everything into one encoder, and that it does so interpretably. It *showed* that the field's standard practice — reporting accuracy on a single dataset — hides a near-total collapse under transfer, a failure that only an order-aware metric like ordinal agreement reveals. And it *demonstrated*, on real unseen students, that the advantage holds exactly where it is hardest. So the contribution is not one new best score; it is evidence that the task should be measured differently — by ordinal agreement, honestly, with attention to the rare class. That shift is what I'd like you to remember."

---

## Slide 15 — Limitations & Future Work (0:35) — *impact*

**Visual:** Two columns.
- Limitations: single annotator; clip-level I3D; modest in-domain gain.
- Future work: target the rare HD class; domain adaptation; learned group encoders; larger multi-annotator, multi-culture private set.

**Script:**
> "What this work could *not* settle points to where the next work lives. Three honest limits remained: the private set stayed small and had a single annotator, and the in-domain gain proved modest. And the rare Highly-Disengaged class — the one we most want to catch — was still caught worst, dropping to near-zero recall under shift. So the natural next steps are to target that rare class directly, add domain adaptation, let the model *learn* the group encoders rather than enumerating them by hand, and grow the private set — bigger, with more annotators and more cultures. Thank you for listening — I'm happy to take questions."

---

## Slide 16 — Thank you / Q&A + BACKUP SLIDES

**Visual:** "Thank you · Questions?" + contact. Keep these hidden backups:

| Backup | Figure / Table | Provenance | Use for |
|--------|----------------|------------|---------|
| Full in-domain leaderboard | `base_models_all_metrics.png` | 🟦 BASELINE | "show me all baselines" |
| Per-metric winner | Table `T4_1_per_metric_winner` | 🟦 BASELINE | "which metric picks which model" |
| OpenFace vs I3D | Table `T4_3_openface_vs_i3d` | 🟦 BASELINE | feature-family comparison |
| Loss is a Pareto choice | `loss_metric_tradeoff.png` | 🟦 BASELINE | "why cross-entropy?" |
| Group marginal effect | Table `T5_1_group_marginal_combined` | 🟩 HYBRID | "is head pose really special?" |
| I3D fusion effect | Table `T5_2_i3d_fusion_effect` | 🟩 HYBRID | "does fusion help under shift?" |
| Top-k hybrid configs | Table `T5_3_hybrid_topk` | 🟩 HYBRID | best configuration details |
| Private by source | Table `T5_4_private_by_source` | 🟩 HYBRID | per-training-source private results |
| Per-class F1 | `per_class_f1.png` | 🟪 vs | rare-class behaviour |
| In-domain ≠ transfer | `indomain_vs_generalization_hybrid.png` | 🟩 HYBRID | "can you pick a model in-domain?" (ρ=−0.31) |
| Dataset stats | Table `T3_1_dataset_stats` | ⬜ DATA | exact counts |

---

## Anticipated Q&A — rehearse these

1. **"Why is the in-domain gain so small (+0.068)?"** → Most versions score about the same; the value is being robust off-domain (+0.094 on the private set) and a clear, interpretable framework, not a new best score in-domain. *(Backup: `indomain_vs_generalization_hybrid.png`.)*
2. **"Single annotator — isn't the private set unreliable?"** → Yes, the main limit; the model's suggestions while labelling were optional; the conclusions rest on QWK (Quadratic Weighted Kappa) trends, not single clips; future work adds more annotators. It's still the only test on a *different population*.
3. **"Why not ensemble the models that make different errors?"** → A perfect picker shows 97.6% versus 77.3%; I left it as future work to keep the contribution focused. *(Backup: `agreement_base_models.png`.)*
4. **"Why develop on CMOSE, not DAiSEE?"** → DAiSEE's in-domain QWK (Quadratic Weighted Kappa) is 0.166 — too weak to tell architectures apart from label noise.
5. **"Why does I3D help in-domain but not under shift?"** → It learns useful but dataset-specific look cues (+0.060 in-domain, −0.021 cross-public). *(Backup: `T5_2_i3d_fusion_effect`.)*
6. **"Did you compute the features yourselves?"** → CMOSE ships OpenFace + I3D; I extracted both for DAiSEE and the private set with the same toolkits.

---

## Delivery notes

- Say **"QWK (Quadratic Weighted Kappa)"** as **"ordinal agreement"** the first two times — helps any panelist who is not from computer vision.
- The arc to land: **small gain in-domain → collapse on transfer → compounds on the private set.** Repeat the words *honest* and *compound*.
- On every chart, state provenance first (🟦 baseline / 🟩 hybrid / 🟪 head-to-head / ⬜ data).
- Pre-empt the single-annotator question on slide 7; don't wait for it.
- The opening problem (slide 3) and the closing slides 14–15 are your highest-attention moments — land them cleanly and slowly. The five ⭐ DWELL slides (7, 8, 11–13) are where you slow down; everything else moves briskly.
