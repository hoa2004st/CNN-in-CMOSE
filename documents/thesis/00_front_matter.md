# Acknowledgement

> TODO: Personal acknowledgements — advisor, lab, family, dataset providers (HKUST/LifeHikes
> for CMOSE [@cmose], the DAiSEE authors [@daisee]).

# Abstract

Automatically recognizing student engagement from video is a key enabler for online and
hybrid learning analytics, yet it is hard: engagement labels are *ordinal* (Highly-Disengage,
Disengage, Engage, Highly-Engage), severely *imbalanced*, and subjective. This thesis studies
engagement classification from per-frame facial-behavior descriptors (OpenFace) and video
motion features (I3D), and makes two contributions.

First, we propose a **semantic-group hybrid temporal architecture**: instead of feeding all
709 OpenFace features into one monolithic sequence model, we decompose them into five
behaviorally meaningful groups — gaze, eye landmarks, face landmarks, head pose, and action
units — and assign each group its own temporal encoder (a Temporal Convolutional Network or a
Transformer), optionally fused with an I3D motion stream. Through a systematic ablation over
all 32 per-group encoder assignments (with and without I3D), we show that the I3D-fused hybrid
family is robustly stronger than the best monolithic baseline on the CMOSE test set
(quadratic-weighted kappa, QWK, of 0.574 vs. 0.537), and that the only group with a clear
encoder preference is **head pose, which favors a TCN**.

Second, we conduct a **3×3 cross-dataset generalization study** (training on CMOSE, DAiSEE, or
their union; testing on each test set plus a **self-collected, hand-labeled private set**) and
treat the private set as the decisive real-world probe. We find that engagement classifiers do
**not** transfer across corpora — off-diagonal QWK collapses to near zero even when raw accuracy
appears acceptable — and that the two contributions of this thesis *compound on the unseen
private set*: training on the combined corpus with the semantic-group hybrid yields the best
private-set result (QWK 0.365 vs 0.285 for the best baseline, a wider margin than in-domain).
We argue that ordinal agreement metrics (QWK, Cohen's kappa), not accuracy, must be the primary
yardstick for imbalanced engagement data.

> TODO: add 1–2 sentences of quantitative headline + keywords.
>
> **Keywords:** student engagement, affective computing, OpenFace, I3D, temporal convolutional
> networks, ordinal classification, cross-dataset generalization.
