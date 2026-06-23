# Defense Presentation — Slide Design & Speaking Script

**Thesis:** Student's engagement detection in online classes
**Author:** Phan Minh Hòa
**Audience:** technical committee · **Language:** English · **Target length:** ~12 min (12:30 budgeted)

---

## How to use this file

- Each slide has: **Visual** (what to put on screen, which figure/table, layout), a **Provenance** tag for any chart/table, and **Script** (what to say).
- The deck **follows the thesis chapter order** (Intro → Literature → Methodology → Baseline → Hybrid → Conclusions). Express slides move fast; the **three contributions are marked ⭐ DWELL** — slow down, make eye contact, let the result land.
- The script is written to be *rewritten in your own speaking style* — keep the numbers and the logical beats, change the words.

### Chart/table provenance legend (use these consistently)

| Tag | Meaning |
|-----|---------|
| 🟦 **BASELINE** | Produced from the 5 monolithic baseline models (Chapter 4). |
| 🟩 **HYBRID** | Produced from the proposed semantic-group hybrid, 243/486 configs (Chapter 5). |
| 🟪 **BASELINE vs HYBRID** | Direct comparison of best baseline against best hybrid. |
| ⬜ **DATA / PROTOCOL** | Dataset statistics or evaluation protocol — model-independent (Chapter 3). |
| 🟨 **ARCHITECTURE** | A design/architecture diagram, not a result. |

> Say the provenance out loud the first time a chart appears, e.g. *"this plot summarises the five baseline models"* vs *"this is the proposed hybrid."* The committee must never be unsure which model a number describes.

---

## Timing budget (target 12:30)

| # | Slide | Chapter | Time | Running | Mark |
|---|-------|---------|------|---------|------|
| 1 | Title | — | 0:25 | 0:25 | |
| 2 | The problem | Ch.1 | 0:55 | 1:20 | |
| 3 | Why it's hard (3 properties + culture) | Ch.1 | 1:00 | 2:20 | |
| 4 | The gap + 3 contributions roadmap | Ch.1 | 0:45 | 3:05 | |
| 5 | Building blocks: features & encoders | Ch.2 | 0:55 | 4:00 | express |
| 6 | Data, imbalance & the private set | Ch.3 | 1:25 | 5:25 | ⭐ DWELL (private set) |
| 7 | Core idea: monolithic → decompose | Ch.3 | 0:55 | 6:20 | |
| 8 | The semantic-group hybrid | Ch.3 | 1:15 | 7:35 | ⭐ DWELL (architecture) |
| 9 | Evaluation discipline: QWK not accuracy | Ch.3/4 | 0:50 | 8:25 | |
| 10 | Baselines in-domain + complementarity | Ch.4 | 0:55 | 9:20 | |
| 11 | Result A — architecture ablation | Ch.5 | 1:00 | 10:20 | ⭐ DWELL (architecture) |
| 12 | Result B — generalization collapses | Ch.4/5 | 0:55 | 11:15 | ⭐ DWELL (generalization) |
| 13 | Result C — private set, the payoff | Ch.5 | 1:05 | 12:20 | ⭐ DWELL (private set) |
| 14 | Conclusions & contributions | Ch.6 | 0:35 | 12:55 | |
| 15 | Limitations & future work | Ch.6 | 0:30 | 13:25 | |
| 16 | Thank you / Q&A + backup | — | — | — | |

If you must hit a hard 10:00: cut slide 5 to 20s, merge 10 into 11, and shorten 15. The four ⭐ DWELL slides are non-negotiable — they are your contributions.

---

# CHAPTER 1 — INTRODUCTION

## Slide 1 — Title (0:25)

**Visual:** Title, your name, supervisor, HUST logo. Optional faint background: a webcam frame with OpenFace landmark overlay.

**Script:**
> "Good morning. My thesis is on automatic detection of student engagement in online classes — taking a short webcam clip of one learner and estimating how engaged they are. I'll show that the interesting challenge is not raw accuracy but doing this *honestly*, and I make three contributions: a new architecture, a new self-collected test dataset, and the first honest measurement of how these models generalize."

---

## Slide 2 — The problem (0:55)

**Visual:** Left: instructor facing a grid of ~40 video tiles, one dimmed. Right: the ordinal scale **HD → DE → EG → HE**. Built from shapes, no thesis figure.

**Script:**
> "Online learning made the webcam the only window onto the learner. A teacher facing forty live tiles cannot notice that one student quietly disengaged ten minutes ago. Automatic engagement recognition closes that gap. The input is a ten-second clip of one learner; the output is one of four *ordered* levels, from Highly-Disengaged up to Highly-Engaged. The order matters, and the value of the whole system lives in the tails — catching the rare moments a student slips away."

---

## Slide 3 — Why it's hard: three properties (1:00)

**Visual:** Three stacked rows:
1. **Ordinal** — the HD→HE line; "off-by-one ≈ fine, opposite end = serious error."
2. **Imbalanced** — mini bar "CMOSE: 69% Engage, <3% Highly-Disengaged."
3. **Subjective + corpus-specific + cultural** — two webcams, different people/lighting; small icon of diverse faces.

**Script:**
> "Three properties drive every decision in the thesis. First, the labels are *ordinal*: predicting Highly-Engaged for a Highly-Disengaged student is a serious error, while being off by one is almost right — plain accuracy ignores this. Second, they are *severely imbalanced*: in CMOSE, 69% of clips are 'Engaged' and under 3% are 'Highly-Disengaged', so a model can score high accuracy by always guessing the majority and never detecting disengagement — the one thing we built it for.
> Third — and this motivates my dataset — engagement is *subjective and population-specific*. It's an annotator judgement under a particular rubric and population. And crucially, **learners from different cultural and ethnic groups outwardly display the same inner engagement differently** — gaze norms, head-movement habits, and baseline facial dynamics all vary. So a model tuned on one population's expressive style can systematically misread another's. That's why a model must be tested on a population that looks like its deployment, not just on the corpus it trained on."

---

## Slide 4 — The gap and the three contributions (0:45)

**Visual:** Top: two gaps in red — "(1) 709 facial features encoded *monolithically*" and "(2) cross-dataset transfer almost never measured." Bottom: three numbered contribution cards:
1. 🏗️ **Architecture** — semantic-group hybrid: *divide the face, then fuse*.
2. 📹 **Private dataset** — 366 self-collected, hand-labelled clips, *test only*.
3. 🌍 **Generalization** — 3×3 cross-dataset study; QWK, not accuracy.

**Script:**
> "Two gaps in the literature define the thesis. One: the 709-dimensional OpenFace facial descriptor is almost always crushed into a single encoder, ignoring its structure. Two: cross-dataset generalization — the only thing that matters for deployment — is almost never measured. My three contributions answer these: a new architecture that respects the descriptor's structure; a new private dataset I collected and labelled myself, used only for testing; and an honest cross-dataset study. I'll take you through them in the thesis's own order."

---

# CHAPTER 2 — LITERATURE REVIEW (kept deliberately short)

## Slide 5 — Building blocks: features and encoders (0:55) — *express*

**Visual:** Two columns.
- **Features:** "OpenFace → 709-d per frame (gaze, eye/face landmarks, head pose, action units). I3D → 1024-d per clip (motion/appearance)." Small note: *"CMOSE ships these; DAiSEE & private extracted by us from raw video."*
- **Encoders:** three tiny icons — **TCN** (`openface_tcn_block.png`), **LSTM** (`openface_lstm_cell.png`), **Transformer** (`openface_transformer_encoder_layer.png`).

**Provenance:** 🟨 ARCHITECTURE (building-block diagrams, not results).

**Script:**
> "Two feature families. OpenFace gives a 709-dimensional descriptor per *frame* — gaze, eye and face landmarks, head pose, and action units. I3D gives one 1024-dimensional vector per *clip*, summarising motion and appearance. One clarification I'll come back to: **only CMOSE ships these features precomputed; for DAiSEE and my private set I extracted both myself from the raw video**, with the same toolkits, so all three are format-compatible.
> To model the OpenFace sequence over time I use three standard encoder families — Temporal Convolutional Networks, LSTMs, and Transformers. These are the interchangeable building blocks of everything that follows."

---

# CHAPTER 3 — METHODOLOGY

## Slide 6 — ⭐ DWELL — Data, imbalance, and the private set (1:25)

**Visual:** Left: **`dataset_class_distribution_overall.png`**. Right: a highlighted **private-set card** — "366 clips · 10s · single learner · OpenFace+I3D extracted by us · single annotator · **TEST ONLY** · 58% EG / 31% HE / 9% DE / 3% HD." Optional: screenshot of your labelling interface. Optionally show **Table T3_1** as a backup.

**Provenance:** ⬜ DATA / PROTOCOL (Figure = class distribution, model-independent; Table T3_1 = dataset statistics).

**Script (slow down here — this is contribution #2):**
> "Now the data. I train on two public benchmarks — CMOSE and DAiSEE — and their union. This figure shows the central difficulty: all three are heavily imbalanced, but *in different directions*. CMOSE is dominated by 'Engaged'; DAiSEE is split between 'Engaged' and 'Highly-Engaged'. A model fit to one prior meets a very different one at test time — which is exactly the transfer problem.
> And here is my second contribution — **the private set**. These are 366 webcam clips from real online sessions, cut to ten seconds, one learner each, filtered for tracking quality, and **hand-labelled by me** through a purpose-built interface. I ran them through the *same* OpenFace and I3D extraction I applied to DAiSEE — identical in format to what CMOSE ships. It is used strictly for testing: never in training, never in model selection.
> Why build this at all? Because, as I said, engagement is *displayed* differently across cultural and ethnic groups. A model validated only on a foreign-population benchmark gives no guarantee on a local population. A small, locally collected, hand-labelled test set is the most direct way to check a model is fit for its real users *before* deployment. I'll be honest about its one weakness — a single annotator — in the limitations."

*(Pre-empting the single-annotator question here is deliberate.)*

---

## Slide 7 — The core idea: monolithic → decomposition (0:55)

**Visual:** Two-panel contrast.
- Left "Standard practice": "709 features → ONE encoder → class" (use `openface_tcn.png` as icon). 🟨 BASELINE architecture.
- Right "Proposed": the 709 vector fanning into 5 colored streams — **gaze · eye landmarks · face landmarks · head pose · action units**.

**Provenance:** 🟨 ARCHITECTURE.

**Script:**
> "Here is the idea behind the architecture. One OpenFace frame bundles signals of completely different physical natures — a few gaze angles, hundreds of landmark coordinates, head pose, action-unit intensities. They move on different time scales: eyes flick fast and noisily, head turns are slow and smooth, action units fire in bursts. The standard recipe, on the left, concatenates all 709 numbers into one encoder and forces a single temporal assumption on all of them. My proposal, on the right, is 'divide the face, then fuse': split the descriptor into five behaviourally meaningful groups, and give each its own encoder."

---

## Slide 8 — ⭐ DWELL — The semantic-group hybrid architecture (1:15)

**Visual:** **`hybrid.png`** full width. Corner legend with the three group-encoder options (`hybrid_tcn_encoder.png`, `_transformer_`, `_lstm_`).

**Provenance:** 🟨 ARCHITECTURE (this is the proposed model — contribution #1).

**Script (slow down — contribution #1):**
> "This is the full proposed architecture. Each of the five groups is encoded independently into a 64-dimensional embedding by one of three encoders — a TCN, a Transformer, or an LSTM, chosen per group. The 1024-dimensional I3D vector can be added as a sixth, motion stream. The embeddings are concatenated and classified — and, importantly, **every stream also gets its own auxiliary head**, so each behavioural channel is forced to be discriminative on its own, not just useful in the mix.
> Because each of five groups can take any of three encoders, there are three-to-the-fifth — 243 — possible configurations, and I evaluate *all* of them, with and without I3D — 486 models in total. That exhaustive sweep is what lets me make claims about the *family* of models, not just one lucky network."

---

## Slide 9 — Evaluation discipline: QWK, not accuracy (0:50)

**Visual:** **`metric_correlation_base.png`** (six-metric Kendall-τ agreement). Caption strip: "Accuracy can be gamed by majority-guessing → primary metrics: **QWK · macro-accuracy · macro-MAE**."

**Provenance:** 🟦 BASELINE (the metric-disagreement evidence is computed over the 15 baseline configurations — say so).

**Script:**
> "Before comparing any models I fixed the evaluation, using the baselines. This heatmap — computed across the baseline configurations — shows the six candidate metrics *disagree* about which model is best, so the choice of metric is itself a decision. Because accuracy can be gamed by majority-guessing on imbalanced data, I commit to three order-aware, balanced primary metrics: quadratic-weighted kappa — QWK, which rewards ordinal agreement — plus macro-accuracy and macro-MAE. QWK is my headline number. This discipline, fixed on the baselines, is exactly what later exposes a failure that accuracy hides."

---

# CHAPTER 4 — BASELINE MODELS

## Slide 10 — Baselines in-domain & why I fuse I3D (0:55)

**Visual:** Left: **`base_models_all_metrics.png`** (the 5 baselines × 3 losses, in-domain CMOSE) — best baseline QWK **0.537**. Right (small): **`agreement_base_models.png`** with callout "OpenFace and I3D make *complementary* errors → motivates fusion; oracle reaches 97.6% vs 77.3% best single."

**Provenance:** 🟦 BASELINE (both plots are the five baseline models; left = leaderboard, right = pairwise prediction agreement).

**Script:**
> "Chapter 4 reads the baseline leaderboard in-domain on CMOSE. The strongest monolithic baseline reaches a QWK of 0.537 — that's the bar everything must beat. The second plot is the justification for adding I3D: it's a prediction-agreement analysis of the baselines, and it shows the best OpenFace model and the I3D model *disagree more with each other than the OpenFace models do among themselves* — they solve different clips. That measured complementarity — not intuition — is why I fuse the two families in the hybrid. The same analysis shows an oracle would reach 97.6% of clips versus 77.3% for the best single model: real headroom, which I flag as future work."

---

# CHAPTER 5 — SEMANTIC-GROUP HYBRID

## Slide 11 — ⭐ DWELL — Result A: architecture ablation (1:00)

**Visual:** **`hybrid_ablation_all_metrics.png`** (243 configs, with/without I3D). Overlay headline numbers: "Best baseline **0.537** → hybrid median **0.553**, best **0.605** (+0.068) · 82% of configs beat baseline." Sub-note: "Only **head pose** prefers an encoder → **TCN**." Optionally **Table T5_3** (top-k configs) as backup.

**Provenance:** 🟩 HYBRID (the 243-configuration population; this is the proposed model's result).

**Script (slow down — architecture result):**
> "Result A is the architecture, in-domain. This is the whole population of 243 hybrid configurations. As a *family*, the I3D-fused hybrid sits above the baseline: median QWK 0.553 against the baseline's 0.537, with 82% of all configurations clearing the bar, and the best reaching 0.605. I want to be honest: that best-case gain, plus-0.068, is *modest*. The more durable finding is the design rule — four of the five groups don't care which encoder you give them; only *head pose* shows a clear preference, for the TCN, because head motion is temporally local. So the message is not a magic network; it's that *decomposing-and-fusing* is what helps, and a TCN is a safe default. That family-level honesty is the point."

---

## Slide 12 — ⭐ DWELL — Result B: generalization collapses (0:55)

**Visual:** **`crossdomain_base.png`** (3×3 matrix, best baseline per cell) on the left; **`crossdomain_hybrid.png`** (best hybrid per cell) on the right, or just the baseline matrix with a callout. Big text: "Off-diagonal QWK ≈ 0 (CMOSE→DAiSEE **0.02**) — yet accuracy looks fine." 

**Provenance:** 🟦 BASELINE matrix (left) + 🟩 HYBRID matrix (right). Label both clearly.

**Script (slow down — generalization, contribution #3):**
> "Result B is the critical, negative finding — and it's the third contribution. When I move *any* model off the corpus it trained on, performance collapses. Look at the off-diagonal cells of this 3-by-3 matrix: cross-dataset QWK falls to near zero — CMOSE to DAiSEE is 0.02. These models do *not* transfer off-the-shelf. And here's the trap that justifies my whole metric discipline: on those same cells, *accuracy still looks acceptable*, because majority-guessing scores well. Had I reported accuracy like most of the field, I would have hidden a total failure. The single most effective simple fix is to *pool* the source corpora for training — which sets up the final result."

---

## Slide 13 — ⭐ DWELL — Result C: the private set, where it all compounds (1:05)

**Visual:** **`private_confusion_combined.png`** (best baseline vs best hybrid confusion on the private set, side by side). Big callout: "Private set, combined training — Hybrid QWK **0.379** vs baseline **0.285** (**+0.094**, wider than in-domain) · Hybrid wins **all 9** cross-dataset cells." Optional inset: **`crossdomain_delta.png`** or **Table T5_4** (private by source).

**Provenance:** 🟪 BASELINE vs HYBRID (this is the head-to-head on the contribution dataset — say it explicitly).

**Script (the climax — slow down most; tie all three contributions together):**
> "Result C brings the three contributions together on the private set — the unseen, real-population probe. This is the best baseline against the best hybrid, on my own data. Two things. First, the hybrid beats the best baseline in *all nine* train-by-test cells of the matrix — the advantage is not one lucky comparison. Second, because the private set is test-only, the only lever I have is the training source: training on the *combined* corpus with the *hybrid* gives the best result in the whole thesis — QWK 0.379 versus 0.285 for the best baseline. That margin, plus-0.094, is *wider* than the plus-0.068 I got in-domain.
> So the architecture earns its keep precisely where it's hardest — on a real, unseen population that displays engagement differently from either training corpus. The architecture, the pooling insight, and the private set all pay off *together*. That is the practical heart of the thesis."

---

# CHAPTER 6 — CONCLUSIONS

## Slide 14 — Conclusions & contributions (0:35)

**Visual:** Three checkmark cards mirroring slide 4:
1. ✅ **Architecture** — decompose-and-fuse; +0.068 in-domain; interpretable (head pose → TCN).
2. ✅ **Private dataset** — 366 clips, the decisive deployment probe across a different population.
3. ✅ **Generalization** — transfer collapses; QWK exposes it; pooling + hybrid generalize best (+0.094 on private).

**Script:**
> "To conclude. The architecture is a sound, interpretable model that degrades *gracefully* under distribution shift. The private dataset gave me the cleanest test of deployment on a different population. And the generalization study reframes the task — from 'beat the benchmark' to 'report ordinal agreement and instrument rare-class failure.' The three contributions compound exactly where it counts: on unseen, real footage."

---

## Slide 15 — Limitations & future work (0:30)

**Visual:** Two columns. **Limitations:** single annotator (private) · clip-level (not temporal) I3D · modest in-domain gain · weak DAiSEE signal. **Future:** target rare HD class (ordinal loss / SMOTE) · domain adaptation · learned (not enumerated) group encoders · per-window motion + audio · larger, multi-annotator, multi-culture private set.

**Script:**
> "Honest limitations: the private set has a single annotator and is small; the I3D feature is clip-level, not temporal; and the in-domain gain is modest. The biggest open problem is the rarest class, Highly-Disengaged — two-to-three percent of the data, and zero recall under shift, even though it's the most decision-relevant. Future work targets it directly, adds domain adaptation, learns the group encoders instead of enumerating them, and — connecting back to the motivation — builds a larger, multi-annotator, multi-culture private set. Thank you — I'm happy to take questions."

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

1. **"Why is the in-domain gain so small (+0.068)?"** → The design space is mostly flat; the value is *off-domain* robustness (+0.094 on the private set) and an interpretable framework, not a new in-domain SOTA. *(Backup: `indomain_vs_generalization_hybrid.png`.)*
2. **"Single annotator — isn't the private set unreliable?"** → Yes, it's the main limitation; model suggestions during labelling were non-binding; conclusions rest on QWK trends, not single clips; future work adds multiple annotators. Note it's still the only test on a *different population*.
3. **"Why not ensemble the complementary models?"** → Oracle shows 97.6% headroom vs 77.3%; deliberately left as future work to keep the contribution focused. *(Backup: `agreement_base_models.png`.)*
4. **"Why develop on CMOSE, not DAiSEE?"** → DAiSEE's in-domain QWK is 0.166 — too weak to distinguish architectures from label noise.
5. **"Why does the I3D stream help in-domain but not under shift?"** → It encodes useful but *corpus-specific* appearance cues (+0.060 in-domain, −0.021 cross-public). Recommended, but claimed only for the regime where it holds. *(Backup: `T5_2_i3d_fusion_effect`.)*
6. **"Did you compute the features yourselves?"** → CMOSE ships OpenFace + I3D; I extracted both from raw video for DAiSEE and the private set with the same toolkits, identical in format.

---

## Delivery notes

- Say **"QWK"** as **"ordinal agreement"** the first two times — bridges any non-CV panelist and reinforces the thesis's identity.
- The arc to land: **modest in-domain → collapse on transfer → compounds on the private set.** Repeat the words *honest* and *compound*.
- On every chart, state provenance first (🟦 baseline / 🟩 hybrid / 🟪 head-to-head / ⬜ data).
- Pre-empt the single-annotator question on slide 6; don't wait for it.
- The three ⭐ DWELL contribution slides (6, 8, 11–13) are where you slow down; everything else can move briskly.
