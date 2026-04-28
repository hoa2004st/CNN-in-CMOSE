# OpenFace TCN + I3D Fusion Pipeline

## Goal

Design a multimodal engagement-classification pipeline with:

- one input stream from OpenFace frame-level features
- one input stream from I3D video features
- temporal fusion before the final classifier

The design keeps the current repo direction of sequence modeling while extending it with an appearance-motion stream from I3D.

## High-Level Architecture

```text
Video / person-track
    ->
time alignment and sample ID matching
    ->
    +---------------- OpenFace stream ----------------+
    | OpenFace CSV -> T x F_openface                 |
    | clean / normalize / resample                   |
    | Temporal CNN encoder                           |
    | -> T' x D                                      |
    +------------------------------------------------+
    +------------------ I3D stream ------------------+
    | RGB or precomputed I3D features -> T_i3d x F_i3d |
    | normalize / temporal resample                  |
    | linear projection or lightweight temporal block|
    | -> T' x D                                      |
    +------------------------------------------------+
    ->
temporal fusion
    ->
global pooling
    ->
MLP classifier
    ->
4 engagement classes
```

## Input Definitions

### 1. OpenFace branch

- Source: per-sample OpenFace CSV already used in this repo
- Raw shape: `300 x 709` after frame resampling to 300 frames
- Content: landmarks, pose, gaze, AU, and related frame-level descriptors
- Preprocessing:
  - keep the best detected face per frame
  - fill or interpolate short missing spans
  - min-max or z-score normalization using training-set statistics only
  - output shape: `T x F_openface`, with `T = 300` and `F_openface = 709`

### 2. I3D branch

- Source: precomputed I3D features for the same sample IDs
- Recommended feature type: clip-level embeddings from RGB I3D, or RGB+flow if available
- Typical raw shape:
  - `T_i3d x 1024` for a single RGB stream
  - `T_i3d x 2048` if RGB and flow are concatenated
- Preprocessing:
  - extract features from fixed-length overlapping clips
  - assign each clip a center timestamp
  - normalize with training-set statistics only
  - resample to the common fusion length `T'`

## Recommended Fusion Strategy

Use **late temporal fusion** with a shared sequence length.

Reasoning:

- OpenFace and I3D have different feature semantics and raw dimensionalities.
- Early concatenation at the input level is brittle and can let the larger modality dominate.
- Late fusion lets each branch learn modality-specific patterns before combination.

## Detailed Pipeline

## Stage 1. Sample Synchronization

- Use the CMOSE sample key, for example `video10_100_person0`, as the join key.
- Ensure both modalities map to the same person-track and label.
- Drop samples missing one modality, or use a masking strategy if missing data must be tolerated.

## Stage 2. Temporal Standardization

Convert both streams to a shared sequence length `T'`.

Recommended setting:

- OpenFace input length: `300`
- I3D input length: depends on clip extraction
- Fusion length: `T' = 75`

Why `75`:

- it preserves temporal order
- it reduces memory compared with keeping all 300 OpenFace steps
- it matches clip-based I3D features more naturally than `300`

Implementation:

- OpenFace branch: temporal downsampling from `300 -> 75`
- I3D branch: interpolation or nearest resampling from `T_i3d -> 75`

## Stage 3. OpenFace Temporal CNN Encoder

Process OpenFace as a sequence, consistent with the repo's direct temporal modeling direction.

Recommended encoder:

```text
Input: B x 300 x 709
transpose -> B x 709 x 300
Conv1d(709, 256, kernel=5, padding=2)
ReLU
MaxPool1d(2)
Conv1d(256, 128, kernel=5, padding=2)
ReLU
MaxPool1d(2)
Conv1d(128, 128, kernel=3, padding=1)
ReLU
Conv1d(128, D, kernel=3, padding=1)
transpose -> B x T' x D
```

Recommended hidden size:

- `D = 128`

This branch learns fine-grained facial dynamics from OpenFace without forcing the sequence into a square matrix.

## Stage 4. I3D Feature Encoder

I3D features already encode spatiotemporal video content, so this branch should stay lighter than the OpenFace branch.

Recommended encoder:

```text
Input: B x T_i3d x F_i3d
Linear(F_i3d, 256)
ReLU
Dropout(0.3)
Linear(256, D)
optional temporal Conv1d or Transformer block
resample / align to B x T' x D
```

Recommended default:

- use two linear layers plus one lightweight temporal Conv1d block
- keep output width equal to the OpenFace branch width, `D = 128`

## Stage 5. Temporal Fusion

Fuse the two aligned sequences after both are projected to `B x T' x D`.

Recommended primary option:

```text
F_openface: B x T' x D
F_i3d:      B x T' x D
Fuse:       concat along feature dimension -> B x T' x 2D
Gate:       Linear(2D, D) + sigmoid
Mixed:      gate * F_openface + (1 - gate) * F_i3d
```

Why gating is preferred over plain concatenation:

- it lets the model shift trust between facial behavior and broader video motion cues
- it is still simple enough for this dataset size

Fallback option:

- concatenate `F_openface` and `F_i3d`, then project back to `D` with `Linear(2D, D)`

## Stage 6. Sequence Aggregation and Classification

After fusion:

```text
B x T' x D
    ->
temporal average pooling or attentive pooling
    ->
B x D
    ->
Linear(D, 128)
ReLU
Dropout(0.3)
Linear(128, 4)
```

Output classes:

- `0 = very low`
- `1 = low`
- `2 = high`
- `3 = very high`

## Training Design

### Loss

Recommended order:

1. weighted cross-entropy
2. focal loss if class imbalance remains difficult

### Split protocol

- keep the repo's train/evaluation/test protocol
- compute normalization statistics on train only
- fit any temporal alignment parameters on train only

### Regularization

- dropout: `0.3`
- early stopping on validation macro-F1
- weight decay: `1e-4`

### Optimizer

- AdamW
- starting learning rate: `1e-4`

## Tensor Summary

Recommended default tensor flow:

```text
OpenFace:
B x 300 x 709
    ->
B x 75 x 128

I3D:
B x T_i3d x 1024
    ->
B x 75 x 128

Fusion:
(B x 75 x 128, B x 75 x 128)
    ->
B x 75 x 128
    ->
B x 128
    ->
B x 4
```

## Why This Design Fits This Repo

- It preserves the existing OpenFace-based data pipeline.
- It reuses the temporal CNN direction already present in `src/paper_repro_model.py`.
- It adds I3D as a second modality without replacing the current strict evaluation setup.
- It avoids forcing multimodal data into a square 2D matrix, which is harder to justify than direct temporal fusion.

## Suggested Implementation Modules

If this design is later implemented, the cleanest additions would be:

- `src/paper_repro_data.py`
  - add loading and alignment for precomputed I3D features
- `src/paper_repro_model.py`
  - add a `DualStreamOpenFaceI3DModel`
- `src/paper_repro_train.py`
  - add training and evaluation support for multimodal batches

## Recommended Model Name

`openface_tcn_i3d_fusion`

## Minimal Pseudocode

```python
openface_seq = openface_encoder(openface_x)   # B x 75 x 128
i3d_seq = i3d_encoder(i3d_x)                  # B x 75 x 128

gate = torch.sigmoid(gate_layer(torch.cat([openface_seq, i3d_seq], dim=-1)))
fused = gate * openface_seq + (1.0 - gate) * i3d_seq

pooled = fused.mean(dim=1)
logits = classifier(pooled)
```

## Final Recommendation

The best starting design for this project is a **dual-stream late-fusion model**:

- OpenFace -> temporal CNN encoder
- I3D -> lightweight projection encoder
- shared temporal alignment
- gated fusion
- pooled classifier head

This gives a clean multimodal extension with low implementation risk and strong compatibility with the current CMOSE pipeline.
