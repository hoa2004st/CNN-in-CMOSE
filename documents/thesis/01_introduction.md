# Chapter 1. Introduction

## 1.1 Problem Statement

The rapid growth of online and hybrid education has made *automatic engagement recognition* a
practically important problem: if a system can tell when learners are engaged or disengaged
from a webcam video, instructors and adaptive platforms can intervene, personalize pacing, and
measure learning experience at scale. Concretely, the task is to map a short video clip of a
single learner to one of four ordinal engagement levels — **Highly-Disengage (HD),
Disengage (DE), Engage (EG), Highly-Engage (HE)**.

This is harder than a typical image classification problem for three reasons that recur
throughout this thesis:

1. **The labels are ordinal, not nominal.** Confusing HD with HE is far worse than confusing
   EG with HE, so accuracy — which treats all errors equally — is the wrong objective and the
   wrong metric.
2. **The labels are severely imbalanced.** In CMOSE, ~69% of clips are "Engage" and under 3%
   are "Highly-Disengage" (Table&nbsp;T1). A model can reach high accuracy by ignoring exactly
   the rare, decision-relevant disengaged classes.
3. **Engagement is subjective and corpus-specific.** Different datasets are annotated under
   different protocols and recording conditions, so a model trained on one may not transfer to
   another.

## 1.2 Background and Problems of Research

Most prior engagement-recognition systems take one of two routes: (a) hand-crafted or toolkit
facial descriptors (e.g. OpenFace action units, head pose, gaze) fed to a classical classifier
or an RNN [@openface_bilstm; @classical_ml_engagement], or (b) end-to-end deep video models
(CNNs, EfficientNet+LSTM) over raw frames [@efficientnet_lstm_engagement]. The CMOSE
dataset [@cmose] provides a large, expert-labeled multimodal benchmark with both OpenFace and
I3D features, and DAiSEE [@daisee] is a widely used public counterpart.

Two problems motivate this work. **First**, facial-behavior descriptors are *heterogeneous*: a
709-dimensional OpenFace vector mixes gaze directions, hundreds of eye/face landmark
coordinates, head-pose angles, and action-unit intensities. Feeding this heterogeneous bundle
into a single sequence encoder forces one temporal inductive bias onto signals with very
different dynamics. **Second**, the engagement literature reports in-domain numbers almost
exclusively; how well models *generalize across datasets* is rarely quantified, even though
deployment always means an unseen target distribution.

## 1.3 Research Objectives and Conceptual Framework

This thesis pursues two objectives, framed as four research questions:

- **RQ1.** Does decomposing OpenFace features into semantic groups, each with its own temporal
  encoder, improve engagement classification over monolithic temporal baselines?
- **RQ2.** Which group→encoder (TCN vs. Transformer) assignments actually matter?
- **RQ3.** Does adding an I3D motion stream to the hybrid help?
- **RQ4.** How well do these models generalize across datasets, and how do loss functions
  trade off accuracy against minority-class / ordinal performance?

**Conceptual framework (Figure 3.1).** We treat a clip as a multi-stream temporal signal. Each
OpenFace semantic group and the I3D stream is encoded independently into a fixed embedding;
the embeddings are concatenated and classified, with an auxiliary head per stream. This
"divide the face, then fuse" design is the central idea evaluated against standard baselines
and across datasets and losses.

> TODO: insert the conceptual-framework diagram (see Figure 3.1 in Chapter 3) and 1 paragraph
> walking the reader through it.

## 1.4 Contributions

1. **A semantic-group hybrid temporal architecture** for engagement classification that
   decomposes OpenFace features into five behavioral groups with per-group TCN/Transformer
   encoders and optional I3D fusion (Section 3.4).
2. **A systematic 32-config ablation** of per-group encoder choice, ± I3D, showing the
   I3D-fused hybrid is robustly above the best baseline (QWK 0.574 vs 0.537) and isolating
   head-pose as the one group that benefits from a TCN (Section 4.4).
3. **A new, self-collected private evaluation set** of 366 clips, manually labeled by the
   author and processed through the same OpenFace/I3D pipeline, used strictly as a held-out,
   test-only probe of real-world generalization (Section 3.2). On this set the two contributions
   above *compound*: the combined-trained semantic-group hybrid is the best model (QWK 0.365 vs
   0.285 for the best baseline — a larger margin than in-domain), establishing the practical
   value of both the architecture and combined-corpus training (Section 4.6).
4. **A 3×3 cross-dataset generalization study** (CMOSE / DAiSEE / Combined × CMOSE-test /
   DAiSEE-test / Private) demonstrating that engagement models do not transfer off-the-shelf,
   that accuracy masks this while QWK exposes it, and that combined-corpus training best
   mitigates it (Section 4.5).
5. **A reproducible analysis pipeline** turning ~237 trained runs into the thesis figures and
   tables (`src/analysis`, `src/visualization`).

## 1.5 Organization of Thesis

Chapter 2 reviews engagement-detection literature and the foundational building blocks
(feature extractors, temporal architectures, losses, ordinal metrics). Chapter 3 details the
data, the semantic-group hybrid architecture, the baselines, and the training/evaluation
protocol. Chapter 4 presents the numerical results: dataset analysis, baseline and loss
comparison, the hybrid ablation, the cross-domain study, and error analysis. Chapter 5
summarizes findings and outlines future work.
