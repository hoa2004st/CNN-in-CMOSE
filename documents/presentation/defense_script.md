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
| 8 | Semantic-Group Decomposition | Ch.3 | 0:35 | 4:30 | |
| 9 | Hybrid Architecture (Proposed) | Ch.3 | 0:50 | 5:20 | ⭐ DWELL (architecture) |
| 10 | Evaluation Protocol & Metrics | Ch.3/4 | 0:35 | 5:55 | |
| 11 | Baselines: In-Domain Results & Prediction Agreement | Ch.4 | 0:40 | 6:35 | |
| 12 | Encoder Ablation & In-Domain Comparison | Ch.5 | 0:50 | 7:25 | ⭐ DWELL (architecture) |
| 13 | Cross-Dataset Generalization | Ch.4/5 | 0:45 | 8:10 | ⭐ DWELL (generalization) |
| 14 | Private-Set Generalization | Ch.5 | 0:55 | 9:05 | ⭐ DWELL (private set) |
| 15 | Conclusions | Ch.6 | 0:30 | 9:35 | impact |
| 16 | Limitations & Future Work | Ch.6 | 0:35 | 10:10 | impact |
| 17 | Thank you / Q&A + backup | — | — | — | |

The five ⭐ DWELL slides (7, 9, 12–14) are your contributions — non-negotiable. Everything else can move briskly.

---



## Slide 1 — Title (0:15)

**Visual:** Thesis title, your name, supervisor, HUST logo. Faint background: a webcam frame with OpenFace landmark overlay. **Title only — no agenda, no content.**

**Script:**
> "Good morning everyone. I’m Phan Minh Hòa. Today, I will present to you my thesis on “Student’s engagement detection in online classes”. Without wasting your time, let’s go right into the main content."
"

---

## Slide 2 — Table of contents (0:15)

**Visual:** Clean agenda, grouped to mirror the talk's arc:
1. **Introduction**
2. **Literature Review**
3. **Methodology**
4. **Experimental Results**
5. **Conclusions**

**Script:**
> "Here's the outline of my presentation."

---
# CHAPTER 1 — INTRODUCTION

## Slide 3 — Problem Statement (0:35)

**Visual:** 4 examples screenshot. Ordinal scale **HD → DE → EG → HE**.

**Script:**
> "In online classes, the webcam became the teacher's only window onto the learner. When teaching and focusing on their own screen sharing, teachers realistically cannot read the room and decide they should change the pace or repeat the points they just made. To assist online teachers our thesis focus on one automation task: classify ten-second clips into four ordered engagement levels, from Highly-Disengaged to Highly-Engaged."

---

## Slide 4 — Key Challenges: Ordinality, Imbalance, Angle Diversity (0:50)

**Visual:** Three stacked rows:
1. **Ordinal** — HD→HE stair; "off-by-one ≈ fine, opposite end = disaster."
2. **Imbalanced** — mini bar "CMOSE: 69% Engage, <3% Highly-Disengaged."
3. **Angle diversity** — same student, different webcam angles (laptop-up · eye-level · phone-side) → shifted features.

**Script:**
> "This task innately have three key challenges. One: the ordinal scale forms a *stair* — mistaking a disengaged student for a highly-engaged one is a disaster, off-by-one is fine, but plain accuracy can't tell those apart. Two: the dataset is highly imbalance for example, CMOSE dataset have almost 70% of the clips labeled 'Engaged' while only under 3% labeled highly-disengaged. A model can have high accuracy by always guessing 'Engaged' and never doing its job. Three: every learner frames their webcam differently — a laptop looking up, a monitor camera at eye level, a phone off to the side — so the same expression reaches the model at a different angle and head pose. The OpenFace features are measured in that geometry, so the angle alone shifts them even when the engagement is identical. Doing well on one dataset's camera setups tells us little about the angles the model will actually meet."

---

## Slide 5 — Research Gaps & Contributions (0:35)

**Visual:** Top: two red gaps — "(1) 709 facial features put into *one model*" and "(2) almost never tested on a *different* dataset." Bottom: three numbered cards:
1. 🏗️ **Architecture** — *divide the face, then fuse*.
2. 📹 **Private dataset** — 366 self-collected clips, *test only*.
3. 🌍 **Generalization** — 3×3 cross-dataset study; QWK (Quadratic Weighted Kappa), not accuracy.

**Script:**
> "There are two gaps in past work. One: the 709 face features are almost always put into a single encoder, so their structure is lost. Two: models are almost never tested on a *different* dataset — the one thing that matters when you deploy. My three contributions answer both: an architecture that keeps the structure, a private test set I collected and labelled myself, and an honest test across datasets."

---

# CHAPTER 2 — LITERATURE REVIEW

## Slide 6 — Feature Extraction & Temporal Architectures (0:30) — *express*

**Visual:** Two columns.
- **Features:** "OpenFace → 709-d per frame · I3D → 1024-d per clip." Note: *"CMOSE ships these; DAiSEE & private extracted by us."*
- **Encoders:** three icons — **TCN**, **LSTM**, **Transformer**.

**Provenance:** 🟨 ARCHITECTURE.

**Script:**
> "We used two feature extractor. OpenFace gives 709 numbers per frame — gaze, landmarks, head pose, muscle activations. I3D gives one 1024-number vector per clip, for motion. One note: CMOSE comes with these features ready; for DAiSEE and my private set I made them myself, with the same tools. To read the sequence over time I use three standard encoders — TCN, LSTM, and Transformer — the building blocks for everything next."

---

# CHAPTER 3 — METHODOLOGY

## Slide 7 — ⭐ DWELL — Datasets & the Private Evaluation Set (0:55)

**Visual:** Left: **`dataset_class_distribution_overall.png`**. Right: a **private-set card** — "366 clips · 10s · single learner · OpenFace+I3D by us · single annotator · **TEST ONLY**." Optional: labelling-interface screenshot; Table T3_1 as backup.

**Provenance:** ⬜ DATA / PROTOCOL.

**Script (slow down — contribution #2):**
> "I train on two public datasets — CMOSE and DAiSEE — and the combination of two together. This figure shows the main problem: all three are imbalanced in *different ways*, so a model that fits one meets a different one at test time. And here's my second contribution — the private set: 366 real webcam clips, cut from real online class of SOICT, labelled manually, made with the *same* OpenFace and I3D tools, and used *only* for testing — never for training. It directly answers the deployment problem: a local test set recorded on the kind of real webcam setups the model will really meet."

*(Pre-empting the single-annotator question here is deliberate.)*

---

## Slide 8 — Semantic-Group Decomposition (0:35)

**Visual:** Two-panel contrast. Left "Standard": "709 → ONE encoder → class." Right "Proposed": the 709 vector fanning into 5 colored streams — **gaze · eye landmarks · face landmarks · head pose · action units**.

**Provenance:** 🟨 ARCHITECTURE.

**Script:**
> "One OpenFace frame mixes very different signals that change at different speeds — eyes move fast and noisily, the head turns slowly, muscles fire in short bursts. Putting all 709 into one encoder forces the same time model on all of them. So I split the face into five meaningful groups — gaze, eye landmarks, face landmarks, head pose, action units — and give each group its own encoder."

---

## Slide 9 — ⭐ DWELL — Hybrid Architecture (Proposed) (0:50)

**Visual:** **`hybrid.png`** full width. Corner legend with the three encoder options.

**Provenance:** 🟨 ARCHITECTURE (the proposed model — contribution #1).

**Script (slow down — contribution #1):**
> "The full model. Each of the five groups goes into its own encoder — a TCN, Transformer, or LSTM, chosen per group — and becomes a 64-number embedding. I3D can join as a sixth, motion stream. Every stream also gets its own small head, so each group has to be useful on its own, not only in the mix. Three encoders over five groups gives 243 versions — and I test *all* of them, with and without I3D: 486 models in total. That lets me make claims about the whole *family*, not one lucky model."

---

## Slide 10 — Evaluation Protocol & Metrics (0:35)

**Visual:** **`metric_correlation_base.png`**. Caption: "Accuracy can be fooled by majority-guessing → primary metrics: **QWK (Quadratic Weighted Kappa) · macro-accuracy · macro-MAE**."

**Provenance:** 🟦 BASELINE.

**Script:**
> "Before comparing models, I fixed how I measure. This heatmap shows the six metrics *disagree* on which model among baseline models is best — so the metric itself is a choice. Because accuracy can be fooled by always guessing the majority class, I use three order-aware metrics: QWK (Quadratic Weighted Kappa) — ordinal agreement — plus macro-accuracy and macro-MAE. QWK (Quadratic Weighted Kappa) is my main number. This same rule is what later shows a failure that accuracy hides."

---

# CHAPTER 4–5 — EXPERIMENTAL RESULTS

## Slide 11 — Baselines: In-Domain Results & Prediction Agreement (0:40)

**Visual:** Left: **`base_models_all_metrics.png`** — best baseline QWK (Quadratic Weighted Kappa) **0.537**. Right: **`agreement_base_models.png`** with callout "different errors → best-picker 97.6% vs 77.3% best single."

**Provenance:** 🟦 BASELINE.

**Script:**
> "The baseline scores, in-domain on CMOSE: the best single model reaches QWK (Quadratic Weighted Kappa) 0.537 — the bar to beat. The second plot is why I add I3D: the best OpenFace model and the I3D model disagree with each other more than the OpenFace models disagree among themselves — they get different clips right. They make different errors, so I join them. A perfect picker over the two would reach 97.6% versus 77.3% — real room to improve, which I leave as future work."

---

## Slide 12 — ⭐ DWELL — Encoder Ablation & In-Domain Comparison (0:50)

**Visual:** **`hybrid_ablation_all_metrics.png`**. Headline: "Best baseline **0.537** → hybrid median **0.553**, best **0.605** · 82% beat baseline." Sub-note: "Only **head pose** prefers an encoder → **TCN**."

**Provenance:** 🟩 HYBRID.

**Script (slow down — architecture result):**
> "The architecture, in-domain — all 243 versions. As a *family*, the I3D hybrid sits above the baseline: median QWK (Quadratic Weighted Kappa) 0.553 versus 0.537, 82% of versions beat the bar, best 0.605. To be honest, that best gain — plus 0.068 — is small. The stronger finding is a design rule: four of the five groups don't care which encoder they get; only *head pose* clearly prefers one — the TCN — because head motion is short and local in time. So the message isn't a magic model; it's that *split-and-join* helps, and a TCN is a safe default."

---

## Slide 13 — ⭐ DWELL — Cross-Dataset Generalization (0:45)

**Visual:** **`crossdomain_base.png`** (left) + **`crossdomain_hybrid.png`** (right). Big text: "Off-diagonal QWK (Quadratic Weighted Kappa) ≈ 0 (CMOSE→DAiSEE **0.02**) — yet accuracy looks fine."

**Provenance:** 🟦 BASELINE matrix (left) + 🟩 HYBRID matrix (right). Label both.

**Script (slow down — contribution #3):**
> "The key negative finding — my third contribution. Move any model to a dataset it did not train on, and it falls apart: the off-diagonal QWK (Quadratic Weighted Kappa) drops to near zero — CMOSE to DAiSEE is 0.02. These models don't transfer. And on those same cells, *accuracy still looks fine* — the majority-guessing trap again. If I had reported accuracy like most papers, I would have hidden a complete failure. The simplest fix that works best — train on the datasets pooled together — sets up the final result."

---

## Slide 14 — ⭐ DWELL — Private-Set Generalization (0:55)

**Visual:** **`private_confusion_combined.png`** (best baseline vs best hybrid). Callout: "Private set — Hybrid QWK (Quadratic Weighted Kappa) **0.379** vs baseline **0.285** (**+0.094**) · Hybrid wins **all 9** cells."

**Provenance:** 🟪 BASELINE vs HYBRID.

**Script (the climax — slow down most):**
> "Now all three contributions come together on the private set — the unseen, real-student test. Best baseline against best hybrid, on my own data. First, the hybrid wins all *nine* train-by-test cells — not one lucky case. Second, because the set is test-only, my only choice is the training data: training on the pooled datasets with the hybrid gives the best result in the whole thesis — QWK (Quadratic Weighted Kappa) 0.379 versus 0.285. That gap, plus 0.094, is *bigger* than in-domain. The architecture proves its worth exactly where it's hardest — on real, unseen students. That's the practical heart of the thesis."

---

# CHAPTER 6 — CONCLUSIONS

## Slide 15 — Conclusions (0:30) — *impact*

**Visual:** Three checkmark cards mirroring slide 5:
1. ✅ **Architecture** — decompose-and-fuse; +0.068 in-domain; interpretable.
2. ✅ **Private dataset** — 366 clips, the decisive deployment probe.
3. ✅ **Generalization** — transfer collapses; QWK (Quadratic Weighted Kappa) exposes it; +0.094 on private.

**Script:**
> "The three contributions don't just add up — they *compound*: each one makes the others matter more, exactly where it's hardest, on unseen real clips. Together they change the goal of the task — away from chasing accuracy on one dataset, toward measuring ordinal agreement honestly and watching the rare but important class. That change of goal is what I'd like you to remember."

---

## Slide 16 — Limitations & Future Work (0:35) — *impact*

**Visual:** Two columns. **Limitations:** single annotator · clip-level I3D · modest in-domain gain. **Future:** target rare HD class · domain adaptation · learned group encoders · larger multi-annotator, multi-culture private set.

**Script:**
> "What is still weak is where the next work lives. Three honest limits: the private set is small and has a single annotator; and the in-domain gain is small. And the rare Highly-Disengaged class — the one we most want to catch — is still the one we catch worst: it drops to near-zero recall under shift. So future work goes after it directly, adds domain adaptation, lets the model learn the group encoders instead of trying them all by hand, and grows the private set — bigger, with more annotators and more cultures. Thank you for listening — I'm happy to take questions."

---

## Slide 17 — Thank you / Q&A + BACKUP SLIDES

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
- The opening problem (slide 3) and the closing slides 15–16 are your highest-attention moments — land them cleanly and slowly. The five ⭐ DWELL slides (7, 9, 12–14) are where you slow down; everything else moves briskly.
