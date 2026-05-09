# Thesis Direction: Cross-Domain Engagement Recognition Without Target Labels

## Core Research Question

> **Can an engagement recognition model trained on a labeled source domain (CMOSE) generalize to a completely unlabeled target domain (private online meeting dataset) — and can pseudo-labeling strategies recover some of the lost performance?**

This is fundamentally a study of **unsupervised domain adaptation (UDA)** applied to student/participant engagement recognition. You are not claiming to solve domain adaptation — you are *diagnosing* what breaks, *quantifying* the gap, and *experimenting* with lightweight remedies (pseudo-labeling), then reflecting honestly on what is possible without any target labels.

---

## Thesis Narrative Arc

The thesis is structured as a progression from "what does the baseline know?" to "what does it fail to know, and can we help it?":

```
Naive models (11 weeks)
    ↓ (compare)
CMOSE baseline (video-only) — reproduced
    ↓ (apply to)
Private unlabeled dataset — domain shift analysis
    ↓ (attempt to adapt)
Pseudo-labeling strategies — do they help?
    ↓ (reflect)
Discussion: what transfers, what breaks, what is fundamentally limited
    ↓ (optional)
Proposed new pipeline — motivated by the failure analysis
```

This arc is honest: the naive models are not throwaway work, they are your **ablation history**. The lack of target labels is not a weakness — it is the **research condition** your thesis interrogates.

---

## Background & Positioning

### The CMOSE Baseline (Video-Only Branch)

The CMOSE paper proposes a multi-modal engagement recognition system. The paper-baseline implementation has been removed from this repository; the remaining code focuses on retained OpenFace, I3D, and fusion comparison models.

- **OpenFace features** (gaze, head pose, action units) processed through a **TCN** with chunk-wise min/max/variance aggregation
- **I3D features** (spatiotemporal, pretrained on Kinetics-400) used as an **attention guide** over TCN outputs and then **concatenated** with the attended TCN output
- A **normalized FC layer (MLP3)** that outputs a scalar score ∈ [−1, 1], thresholded at (−0.5, 0, 0.5) into the four classes: HD / DE / EG / HE
- Paper-baseline training code is not part of the current repo.

The video-only result reported in the paper is **78.14% accuracy / 55.74% average accuracy**. Average accuracy is the more meaningful metric given severe class imbalance (HD: 346, DE: 2208, EG: 8469, HE: 1170 clips).

### Your Naive Models (Prior Work in This Thesis)

| Model | Features | Notes |
|---|---|---|
| MLP | OpenFace | No temporal modeling |
| LSTM | OpenFace | Temporal but simple |
| TCN | OpenFace | Matches part of the baseline |
| Transformer | OpenFace | Over-parameterized for this scale |
| MLP | I3D only | No high-level features |
| MLP | TCN(OpenFace) + I3D | Concatenation without attention |

These models are compared against each other to answer which retained feature/model combinations transfer best.

### The Domain Gap

| Property | CMOSE | Private Dataset |
|---|---|---|
| Setting | Online presentation training class | Online meeting / video call |
| Subjects | 102 participants, coached environment | Unknown participants, uncontrolled |
| Clip count | 12,197 | 808 raw clips (analysis uses accepted subset) |
| Labels | 4-class, psychology-guided | **None** |
| Resolution | 412×234, 25 fps | Unknown |
| Segment length | ~13.72s average | Unknown |
| Behavior context | Instructed / coached | Natural meeting behavior |

Before any model analysis, the private clips should be filtered through an OpenFace quality gate and recorded in `data/private/accepted.csv` (`is_accepted=1` only). All feature-space comparisons, domain-shift plots, and pseudo-labeling metrics are computed on that accepted subset.

The domain gap is likely driven by: different **engagement cues** (a meeting participant's "engaged" behavior may not look like a student's), different **recording conditions** (virtual backgrounds, varied lighting, different webcam quality), and **context mismatch** (no coach, no instructed task).

---

## Research Questions

### RQ1 — Retained Model Comparison
Which retained CMOSE-trained model performs best under the train/unlabel/test split?

### RQ2 — Dataset Comparison (Feature-Space)
How different are the CMOSE and private datasets at the feature level (OpenFace + I3D distributions), *independent of any model*?

### RQ3 — Domain Shift Analysis
When the CMOSE-trained model is applied to the private dataset, what prediction distribution does it produce? Which classes dominate? Which classes are likely unstable under domain shift?

### RQ4 — Feature Transferability
Which features transfer better across domains: OpenFace high-level features (gaze, AUs, head pose) or I3D spatiotemporal features? Does the attention mechanism still produce meaningful weights in the target domain?

### RQ5 — Pseudo-Labeling
Can pseudo-labeling strategies improve model confidence or consistency on the target domain? Which strategy is most stable?

### RQ6 — Model Comparison
How do the retained OpenFace, I3D, and fusion models compare on CMOSE and on the private dataset?

### RQ7 — Limitations
What are the fundamental limits of adapting without target labels in this specific setting?

---

## Proposed New Pipeline — Brainstorm

Since the thesis will expose failure modes of direct transfer, the proposed pipeline should be **motivated by those failures**. Here are directions worth considering — you will choose after completing Phase 3-4:

### Option A — Domain-Adversarial Feature Alignment (DANN-style)
Train a domain classifier alongside the engagement classifier. The engagement encoder is penalized for producing domain-discriminative features. This explicitly minimizes the feature-space gap without requiring target labels.
- **Motivated by**: if RQ2 shows large feature distribution shift (e.g., AU distributions very different), this directly attacks the cause.
- **Cost**: moderate — requires adding a gradient reversal layer on top of the existing encoder.

### Option B — Confidence-Weighted Self-Training with Ordinal Constraint
Instead of hard pseudo-labels, use **soft ordinal pseudo-labels**: only accept high-confidence predictions AND enforce that pseudo-labels respect the ordinal structure (no isolated HD predictions surrounded by HE, for instance).
- **Motivated by**: if vanilla pseudo-labeling (RQ5) shows erratic class assignments, the ordinal constraint may stabilize it.
- **Cost**: low — can be layered on top of the retained training loop.

### Option C — Prototype-Based Feature Normalization
Compute class prototypes in the embedding space from CMOSE. Normalize target-domain embeddings toward the nearest prototype before scoring. This is a feature-space adaptation without retraining.
- **Motivated by**: if RQ4 shows that the embedding space is structurally similar but shifted in scale/mean, a simple normalization may close the gap.
- **Cost**: very low — inference-time only, no retraining needed.

### Option D — Temporal Consistency Regularization
Enforce that adjacent clips from the same meeting participant receive similar engagement scores (engagement is locally smooth). Use this as a self-supervised signal to fine-tune on the target domain.
- **Motivated by**: the private dataset has temporal structure (it is a meeting) that CMOSE also has but which is not exploited at test time.
- **Cost**: moderate — requires knowing clip order (participant-level metadata).

**Recommendation for thesis scope**: Implement Option C as a low-cost first proposal (always reportable), then attempt Option A or B if time allows. This gives you a guaranteed result to discuss even if the more complex option does not converge.

---

## What This Thesis Contributes

1. **First reproduction** of the CMOSE video-only baseline with ablations on naive variants
2. **Feature-space analysis** comparing CMOSE and a real-world online meeting dataset
3. **Domain shift diagnosis** — which classes and features are most vulnerable
4. **Empirical evaluation** of pseudo-labeling strategies under zero target-label conditions
5. **Honest limitations analysis** — a rare but valued contribution in applied ML theses
6. **(Optional)** A motivated pipeline proposal addressing one identified failure mode

---

## Thesis Claim (Refined)

*"The CMOSE-trained engagement recognition model demonstrates measurable domain shift when applied to an uncontrolled online meeting dataset, with minority classes (HD, HE) being most unstable. OpenFace high-level features transfer more reliably than I3D spatiotemporal features under this shift. Pseudo-labeling provides marginal but inconsistent improvement without target-label supervision. These findings motivate [proposed pipeline], which addresses [specific failure mode identified in RQ2-RQ4]."*

Fill in the bracketed parts after completing Phases 3–4.
