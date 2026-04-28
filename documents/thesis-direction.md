# Thesis Direction

## Recommended Thesis Title

**Engagement Classification on CMOSE with OpenFace and I3D Features Under a Train + Unlabeled Selection Protocol**

Alternative title:

**A Narrowed Comparison of OpenFace and I3D Baselines for CMOSE Engagement Classification**

## Final Thesis Statement

This thesis studies engagement classification on the CMOSE dataset with a deliberately narrowed model scope. It compares OpenFace-only baselines (`openface_mlp`, `temporal_cnn`, `lstm`, `transformer`), an I3D-only baseline (`i3d_mlp`), and a multimodal fusion model (`openface_tcn_i3d_fusion`) under a train + unlabeled selection protocol, where the dataset's source split key `test` is used for early stopping and checkpoint selection.

## Research Problem

The core question is no longer whether paper-style dimensionality reduction or spectral feature engineering is necessary. The narrowed problem is to determine how much predictive value comes from:

- simple OpenFace baselines
- temporal modeling over OpenFace sequences
- I3D-only appearance-motion features
- multimodal fusion of OpenFace and I3D

## Research Objectives

1. Establish lightweight sanity-check baselines with `openface_mlp` and `i3d_mlp`.
2. Compare OpenFace temporal models (`temporal_cnn`, `lstm`, `transformer`) against the OpenFace MLP baseline.
3. Evaluate whether combining OpenFace and I3D in `openface_tcn_i3d_fusion` improves over single-modality models.
4. Report accuracy, macro-F1, weighted F1, and per-class behavior under the same train + unlabeled selection protocol.

## Research Questions

1. How far can simple OpenFace and I3D MLP baselines go on CMOSE?
2. Do temporal OpenFace models outperform the OpenFace MLP baseline?
3. Does `openface_tcn_i3d_fusion` improve over the strongest single-modality model?

## Hypotheses

1. `temporal_cnn`, `lstm`, and `transformer` will outperform `openface_mlp` because they preserve temporal structure.
2. `openface_tcn_i3d_fusion` will outperform both `openface_mlp` and `i3d_mlp` if the two modalities contribute complementary information.
3. `i3d_mlp` will act as a useful sanity check but will likely be weaker than the stronger OpenFace temporal baselines unless the I3D features are especially informative.

## Model Scope

The experimental comparison is restricted to:

- `openface_mlp`
- `temporal_cnn`
- `lstm`
- `transformer`
- `i3d_mlp`
- `openface_tcn_i3d_fusion`

Anything outside this list is out of scope for the current thesis direction.
