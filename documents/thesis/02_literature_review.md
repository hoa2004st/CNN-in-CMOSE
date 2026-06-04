# Chapter 2. Literature Review

## 2.1 Scope of Research

This review is scoped to **vision-based automatic engagement recognition** — classifying a
learner's engagement level from facial video — and to the technical components this thesis
builds on: facial-behavior and motion feature extraction, temporal sequence modeling, loss
functions for imbalanced ordinal targets, and the metrics used to evaluate them. We do not
cover physiological-sensor or clickstream-based engagement estimation, nor general
facial-expression recognition except where it informs feature design.

## 2.2 Related Work

**Engagement datasets and benchmarks.** DAiSEE [@daisee] is the most widely used in-the-wild
benchmark, with four ordinal engagement levels. CMOSE [@cmose] is a more recent, larger,
expert-labeled multimodal dataset that ships precomputed OpenFace and I3D features and reports
strong inter-rater reliability; it is the primary dataset of this thesis. A systematic review
of computer-vision engagement detection [@review_engagement_cv] catalogs the common pipelines
and notes the field's recurring imbalance and subjectivity problems.

**Feature-based approaches.** A large body of work feeds OpenFace descriptors (action units,
gaze, head pose) into sequence models — e.g. OpenFace + BiLSTM [@openface_bilstm] — or applies
dimensionality reduction and resampling (PCA/SVD, SMOTE) before a CNN/MLP
classifier [@openface_pca_smote]. Classical pipelines compare LR/SVM/MLP/KNN/XGBoost on
aggregated features [@classical_ml_engagement]. These establish that OpenFace features carry
real engagement signal but typically treat the 709-dim vector as a single homogeneous input.

**End-to-end video approaches.** Other systems learn directly from frames, e.g.
EfficientNetV2 + LSTM [@efficientnet_lstm_engagement], or borrow context/relationship modeling
from affect recognition [@context_aware_emotion_3d]. These capture appearance and motion but
are heavier and less interpretable than feature-based models.

**Gap.** Across both camps, two gaps persist that this thesis targets: (i) heterogeneous
facial descriptors are encoded monolithically rather than per behavioral subsystem, and
(ii) cross-dataset generalization is rarely measured. Our semantic-group hybrid addresses
(i); our 3×3 study addresses (ii).

## 2.3 Foundational Knowledge: Feature Extraction

**OpenFace 2.0** [@openface2] is an open-source toolkit that, per frame, estimates facial
landmarks, head pose, eye-gaze vectors, and facial action-unit (AU) intensities. In CMOSE the
per-frame descriptor is **709-dimensional**, which this thesis partitions into five semantic
groups (Section 3.3): **gaze (8)**, **eye landmarks (280)**, **face landmarks (340)**,
**head pose (46)**, and **action units (35)**.

**I3D** [@i3d] (Inflated 3D ConvNet) inflates 2D ImageNet filters into 3D and is pretrained on
Kinetics; it produces a 1024-dimensional spatiotemporal feature per temporal window, capturing
appearance and motion that landmark-based descriptors miss. CMOSE provides precomputed I3D
features used here as a complementary motion stream.

## 2.4 Foundational Knowledge: Temporal Architectures, Losses, and Metrics

**Temporal architectures.** Three families model the time axis of a clip:
- **Temporal Convolutional Networks (TCN)** [@tcn] apply dilated causal 1-D convolutions; they
  have a fixed, local-to-global receptive field and strong inductive bias for short temporal
  motifs.
- **LSTM** [@lstm] recurrently integrates the sequence, suited to long dependencies.
- **Transformers** [@transformer] use self-attention to relate all frames directly, flexible
  but data-hungry.
This thesis uses all three as baselines and as the *per-group* encoders inside the hybrid.

**Loss functions for imbalance and ordinality.** We compare:
- **Cross-entropy (CE)** — the standard baseline.
- **Weighted CE** — inverse-frequency class weights to counter imbalance.
- **Focal loss** [@focal] — down-weights easy majority examples.
- **Ordinal (EMD-style) loss** — penalizes predictions by squared CDF distance so that
  far-apart ordinal errors cost more, matching the label structure.
Class imbalance is also classically addressed by resampling such as SMOTE [@smote]; we instead
study loss-level remedies. Optimization uses Adam [@adam].

**Evaluation metrics.** Because the labels are ordinal and imbalanced, we report six metrics
but treat two as primary:
- **Quadratic-Weighted Kappa (QWK)** — agreement corrected for chance, weighting errors by
  squared class distance; the natural ordinal metric.
- **Macro-accuracy** — mean per-class recall, exposing minority collapse.
Secondary: **accuracy**, **Cohen's kappa** [@cohenkappa], **MAE**, and **macro-MAE**.

## 2.5 Others

> TODO: optionally add a short subsection on auxiliary-loss / multi-task learning (the hybrid
> uses per-stream auxiliary heads), and on multimodal fusion strategies, to round out the
> foundations referenced in Chapter 3.
