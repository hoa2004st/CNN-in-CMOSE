# Thesis Direction

## Recommended Thesis Title

**Reassessing PCA/SVD-Based CNN Engagement Classification on CMOSE Through Direct Temporal Modeling Under a Strict No-Leakage Protocol**

Alternative title:

**Engagement Classification from OpenFace Sequences on CMOSE: Revisiting Reduced-Matrix CNN and Direct Temporal Models**

## Final Thesis Statement

This thesis reassesses engagement classification from OpenFace-based facial behavior features on the CMOSE dataset by examining whether the reproduced `PCA/SVD -> reduced matrix -> CNN` pipeline is necessary under a strict no-leakage evaluation setting. Specifically, it compares the baseline reduced-matrix CNN with direct temporal architectures, including temporal CNN, rectangular-filter CNN, LSTM, and Transformer models, to determine whether preserving the original frame-feature sequence improves classification performance, robustness, and methodological validity while reducing preprocessing complexity.

## Research Problem

The reproduced baseline follows a pipeline in which raw frame-level OpenFace features are transformed through PCA or SVD into a reduced square matrix before CNN classification. Although this design is faithful to the source paper, it imposes two assumptions that are not yet well justified for CMOSE:

- dimensionality reduction is necessary before effective learning can occur
- converting a temporal feature sequence into an image-like matrix is a suitable representation for engagement classification

As a result, the key research problem is not merely which model achieves the highest score, but whether the baseline representation itself is appropriate and necessary when evaluated under a stricter and more defensible protocol.

## Research Objectives

1. Reproduce the baseline PCA-CNN and SVD-CNN engagement-classification pipeline on CMOSE using the original train/test split and a strict no-leakage setup.
2. Design and implement direct temporal comparison models that operate on the original frame-feature sequence without PCA or SVD, including temporal CNN, rectangular-filter CNN, LSTM, and Transformer architectures.
3. Compare the baseline and direct temporal models using accuracy, macro-F1, per-class performance, robustness across runs, and preprocessing complexity.
4. Assess whether preserving the original temporal structure of OpenFace features provides a more methodologically valid and practically efficient approach to engagement classification on CMOSE.
5. Identify which architecture family offers the best tradeoff between predictive performance, stability, and implementation cost for thesis-level engagement-recognition research.

## Research Questions

1. Is the reproduced PCA/SVD-to-CNN pipeline necessary for effective engagement classification on CMOSE, or can direct temporal models achieve comparable or better results without dimensionality reduction?
2. Does preserving the original frame-feature sequence improve classification robustness and methodological validity compared with transforming the data into a reduced square matrix?
3. Among temporal CNN, rectangular-filter CNN, LSTM, and Transformer models, which architecture is most suitable for engagement classification from OpenFace sequences under the strict CMOSE protocol?

## Hypotheses

1. Direct temporal models will match or outperform the reproduced PCA/SVD-CNN baseline because they preserve the original temporal structure of the OpenFace features and avoid information distortion introduced by square-matrix conversion.
2. Temporal CNN and rectangular-filter CNN models will provide the best overall tradeoff between performance and stability because they exploit temporal patterns directly while remaining simpler than recurrent and attention-based architectures.
3. Transformer-based models will not necessarily produce the strongest results on CMOSE because the dataset size and task setting may favor lighter architectures with lower risk of overfitting.

## Model Scope

The experimental comparison will cover the following model families:

- baseline PCA-CNN and SVD-CNN
- raw-sequence temporal CNN
- rectangular-filter CNN
- single-direction LSTM
- small Transformer encoder

This scope is sufficient for a thesis because it tests both representation choices and architecture families without becoming too broad to evaluate rigorously.

## Short Description of the Comparison Models

### Temporal CNN

The temporal CNN consumes the original `T x F` OpenFace sequence directly and applies convolution primarily along the time axis. Its purpose is to capture short-term and mid-range temporal patterns in facial behavior without imposing handcrafted feature reduction.

### Rectangular-Filter CNN

The rectangular-filter CNN treats the frame-feature matrix as an asymmetric 2D structure and applies non-square kernels so that temporal and feature dimensions are modeled differently. This architecture tests whether 2D convolution remains useful when the kernel design reflects the real structure of the data rather than a forced square representation.

### LSTM

The LSTM processes the frame-level feature vectors as an ordered sequence and learns temporal dependencies through recurrent memory. It serves as a strong sequential baseline for testing whether longer-range temporal modeling is more appropriate than reduced-matrix CNN classification.

### Transformer

The Transformer encoder models interactions across the full sequence through self-attention. Its role in this thesis is to test whether global temporal dependency modeling offers measurable benefits over simpler direct temporal architectures on CMOSE.

## Expected Contribution

The expected contribution of the thesis is a methodologically stricter and empirically clearer reassessment of engagement classification from OpenFace sequences on CMOSE. Rather than proposing a model only for novelty, the study is intended to show whether the field’s existing reduced-matrix CNN formulation is actually necessary, and whether direct temporal modeling provides a more faithful, simpler, and more defensible alternative.
