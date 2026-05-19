# Technical Summary for Thesis and Slide Writing

Generated from repository state on 2026-05-19.

This document is intentionally written as a handoff brief for another writing agent. It contains the project narrative, data protocol, model and training details, results, caveats, and the artifact map needed to write a thesis and presentation slides without re-reading the whole codebase.

## 1. Project in One Paragraph

This thesis project evaluates computer-vision-based student engagement recognition on the CMOSE dataset using pre-extracted facial behavior features and video-action embeddings. The final codebase trains and compares six neural classifiers across three losses under a fixed CMOSE split protocol: fit on `train`, use `unlabel` as the validation/evaluation split for checkpointing and early stopping, and reserve `test` for final reporting only. The compared models are an OpenFace MLP, an OpenFace temporal CNN, an OpenFace LSTM, an OpenFace Transformer, an I3D MLP, and an OpenFace plus I3D fusion model. The strongest held-out CMOSE result is `i3d_mlp/ce`, with accuracy 0.7723, macro F1 0.5960, weighted F1 0.7549, MAE 0.2473, and MSE 0.2883 on 1,221 test clips. A private manually labeled dataset is also used as an out-of-domain stress test; it shows strong domain shift despite similar label proportions, especially in OpenFace geometric features and I3D embedding centroids.

## 2. Thesis Framing

Recommended thesis title idea:

`Student Engagement Recognition from Facial and Video Features: A CMOSE-Based Comparison of OpenFace, I3D, and Multimodal Temporal Models`

Core research problem:

Automatic recognition of learner engagement from video is useful for online learning analytics, but real-world deployment is hard because engagement labels are imbalanced, facial/video signals are noisy, temporal behavior matters, and models may not transfer cleanly from benchmark datasets to private or local recordings.

Main research question:

How do facial behavior features from OpenFace, video-action embeddings from I3D, and their fusion compare for four-level student engagement classification on CMOSE, and how robust are the learned models when evaluated on a private manually labeled dataset?

Useful sub-questions:

1. Does temporal modeling of OpenFace sequences improve over a simple flattened OpenFace MLP?
2. Do I3D embeddings outperform OpenFace features for this CMOSE split?
3. Do class-weighted or ordinal losses improve minority-class and ordinal-label behavior compared with standard cross entropy?
4. Does OpenFace plus I3D fusion improve over single-modality models?
5. How much does performance change when CMOSE-trained models are applied to accepted private clips?

Main contributions to claim:

1. A reproducible CMOSE train/evaluation/test pipeline using OpenFace and I3D features.
2. A controlled comparison of six model families across CE, weighted CE, and ordinal losses.
3. A multimodal OpenFace TCN plus I3D fusion model with auxiliary reconstruction regularization.
4. A private/manual-label evaluation workflow for out-of-domain assessment.
5. Feature-space and prediction-space analysis showing that the private set has meaningful domain shift even when label proportions are similar.

## 3. Repository Map

Important source files:

| Path | Role |
|---|---|
| `main.py` | Main experiment entry point. Loads CMOSE metadata/features, preprocesses tensors, builds model, trains, evaluates, and writes metrics. |
| `src/feature_extraction/extract_openface.py` | CMOSE/OpenFace metadata alignment, label mapping, OpenFace CSV loading, highest-confidence face row selection, and temporal resampling. |
| `src/feature_extraction/extract_i3d.py` | I3D feature path resolution, loading `.npy`/`.npz`/`.pt`, shape coercion, resampling, and materialization from JSON embeddings. |
| `src/models/models.py` | All model architectures: MLP, TCN, LSTM, Transformer, and OpenFace-I3D fusion. |
| `src/training/train.py` | Normalization, losses, dataloaders, training loop, early stopping, checkpointing, prediction, and metric helpers. |
| `src/evaluation/metrics.py` | Accuracy, balanced/macro accuracy, macro/weighted F1, confusion matrices, MAE, and MSE. |
| `src/output_paths.py` | Canonical output paths and model/loss folder naming. |
| `src/feature_analysis/run_domain_shift_analysis.py` | Generates raw CMOSE-test and private-set predictions for all trained checkpoints. |
| `src/feature_analysis/domain_analysis.py` | Wasserstein and centroid-distance helpers for domain-shift scoring. |
| `src/manual_label_ui/app.py` | Manual labeling UI for private clips. |
| `scripts/compare_naive_models.py` | Runs the full six-model by three-loss experiment suite in one process. |
| `scripts/visualize_models_outputs.py` | Creates per-run curves/reports and CMOSE cross-run charts/tables. |
| `scripts/visualize_dataset_analysis.py` | Creates dataset distribution charts and summaries. |
| `scripts/feature_space_dataset_comparison.py` | Computes feature-space distance reports between CMOSE and accepted private samples. |
| `scripts/visualize_feature_space_domains.py` | Creates t-SNE feature-space visualizations for CMOSE vs private data. |
| `scripts/visualize_model_assessment.py` | Creates CMOSE/private prediction assessment charts and performance-drop charts. |

Important generated artifact roots:

| Path | Content |
|---|---|
| `outputs/training_log/<model>/<loss>/` | Per-run checkpoints, metrics, preprocessing summary, class counts, curves, and Markdown report. |
| `outputs/model_assessment/cmose_testset/` | CMOSE test prediction tables, summaries, heatmaps, confusion matrices, and model-comparison charts. |
| `outputs/model_assessment/private/` | Private manual-label metrics and prediction assessment charts. |
| `outputs/model_assessment/comparison/` | CMOSE vs private performance-drop charts and raw predictions. |
| `outputs/dataset_analysis/` | Dataset class distributions, domain-difference reports, and feature-space visualizations. |
| `documents/references/` | PDF references already collected for thesis literature review. |
| `document/` | This generated thesis handoff summary. |

## 4. Dataset Details

### 4.1 Label Space

The task is four-class ordinal engagement classification. Class IDs are fixed in `src/feature_extraction/extract_openface.py`.

| Class ID | Label |
|---:|---|
| 0 | Highly Disengage |
| 1 | Disengage |
| 2 | Engage |
| 3 | Highly Engage |

The ordering is meaningful: mistakes between adjacent classes are less severe than mistakes across the full range. This is why the project also reports MAE/MSE over class IDs and includes an ordinal loss.

### 4.2 CMOSE Dataset

Main files:

| Path | Meaning |
|---|---|
| `data/CMOSE/final_data_1.json` | Main CMOSE metadata and embedding source. Contains 12,197 records. |
| `data/CMOSE/labels.csv` | Compact labels with columns `clip_id,label,split`. |
| `data/CMOSE/secondFeature/` | Local OpenFace CSV directory, 12,197 CSV files. |
| `data/CMOSE/features/i3d/` | Expected materialized I3D directory in the pipeline. If absent, I3D features can be materialized from `final_data_1.json`. |

`final_data_1.json` record structure:

| Key | Meaning |
|---|---|
| `split` | One of `train`, `unlabel`, `test`. |
| `label` | Human-readable class name. |
| `agreement` | Label agreement score. |
| `embeds` | 1024-dimensional I3D-style embedding vector. |

CMOSE split counts:

| Split | Samples | Proportion |
|---|---:|---:|
| train | 8,783 | 0.72 |
| unlabel, used as evaluation/validation | 2,193 | 0.18 |
| test | 1,221 | 0.10 |
| total | 12,197 | 1.00 |

CMOSE total class distribution:

| Class | Count | Proportion |
|---|---:|---:|
| Highly Disengage | 347 | 0.0284 |
| Disengage | 2,209 | 0.1811 |
| Engage | 8,470 | 0.6944 |
| Highly Engage | 1,171 | 0.0960 |

CMOSE split-by-class counts used in the final experiment:

| Split | Highly Disengage | Disengage | Engage | Highly Engage | Total |
|---|---:|---:|---:|---:|---:|
| train | 250 | 1,591 | 6,099 | 843 | 8,783 |
| evaluation (`unlabel`) | 62 | 397 | 1,524 | 210 | 2,193 |
| test | 35 | 221 | 847 | 118 | 1,221 |

Important imbalance point:

The dataset is dominated by `Engage` at about 69 percent overall. This makes raw accuracy insufficient by itself. Macro accuracy, macro F1, per-class recall, and label-distance metrics are necessary for a fair interpretation.

### 4.3 Private Dataset

Main files:

| Path | Meaning |
|---|---|
| `data/private/clips/` | 428 accepted private `.mp4` clips. |
| `data/private/accepted.csv` | Manifest with accepted/private clip metadata. |
| `data/private/features/openface/` | 808 OpenFace CSV files plus 808 text files. |
| `data/private/features/i3d/` | 428 I3D `.npy` feature files. |
| `outputs/dataset_analysis/private/manual_labels/private_manual_labels.csv` | Manual labels for accepted private clips. |

Private set coverage:

| Item | Count |
|---|---:|
| Candidate private entries in manifest | 808 |
| Accepted private clips | 428 |
| Manually labeled private clips used for supervised metrics | 368 |

Private manual-label class distribution:

| Class | Count | Proportion |
|---|---:|---:|
| Highly Disengage | 11 | 0.0299 |
| Disengage | 52 | 0.1413 |
| Engage | 261 | 0.7092 |
| Highly Engage | 44 | 0.1196 |

The private label distribution is broadly similar to CMOSE: both are heavily dominated by `Engage`. However, feature-space analysis shows the private data is not simply a smaller sample from the same feature distribution.

## 5. Feature Engineering and Preprocessing

### 5.1 OpenFace Features

OpenFace input format:

Each sample is one CSV file keyed by CMOSE person-track ID, for example `video10_100_person0.csv`. CSV metadata columns are:

`frame`, `face_id`, `timestamp`, `confidence`, `success`

All other columns are treated as OpenFace features. The expected feature count is 709.

OpenFace loading procedure:

1. Read CSV with pandas.
2. Strip whitespace from column names.
3. Verify metadata columns exist.
4. For frames with multiple detections, sort by `frame` and descending `confidence`, then keep the highest-confidence row per frame.
5. Drop metadata columns and keep the 709 feature columns.
6. Convert to `float32`.
7. Resample each variable-length temporal sequence to exactly 300 frames using linear interpolation.

OpenFace tensor shape for OpenFace-only models:

`N x 300 x 709`

### 5.2 I3D Features

I3D feature loading supports `.npy`, `.npz`, and `.pt`. The loader coerces features into a stable `time x features` matrix.

Key behavior:

1. If an I3D array is 1-D, it is reshaped to `1 x feature_dim`.
2. If a matrix appears transposed, the loader transposes it.
3. If `target_frames` is provided, the matrix is resampled using the same interpolation helper as OpenFace.
4. For materialized CMOSE I3D features from `final_data_1.json`, the embedding dimension is 1024.
5. Empty CMOSE embedding lists are replaced by zero vectors during materialization.

I3D materialization summary from completed runs:

| Field | Value |
|---|---:|
| Source JSON | `data/CMOSE/final_data_1.json` |
| Embedding dimension | 1024 |
| Written feature files | 12,197 |
| Empty embeddings replaced by zeros | 295 |
| Invalid records skipped | 0 |

I3D tensor shape used by I3D-based models:

`N x 75 x 1024`

### 5.3 Fusion Temporal Alignment

The fusion model uses both OpenFace and I3D streams at a shared temporal length of 75 frames.

OpenFace path for fusion:

1. Load and resample raw OpenFace to 300 frames.
2. Normalize using train-fitted OpenFace mean/std.
3. Resample normalized OpenFace from 300 to 75 frames for temporal alignment with I3D.

Fusion input shapes:

| Stream | Shape |
|---|---|
| OpenFace | `N x 75 x 709` |
| I3D | `N x 75 x 1024` |

### 5.4 Normalization

All feature normalization is train-only fitted z-score normalization.

For each feature dimension:

`x_norm = (x - mean_train) / std_train`

The mean and standard deviation are computed over training samples and frames only, using axes `(sample, frame)`. Evaluation and test data are normalized using the training statistics. Standard deviations less than or equal to zero are replaced with 1.0.

This is important for thesis methodology:

There is no normalization leakage from evaluation or test samples because statistics are fitted only on the CMOSE `train` split.

### 5.5 SMOTE and Class Balancing

SMOTE is disabled in the final pipeline. The generated `smote_summary.json` files record `after_smote: null`.

Class imbalance is addressed only through alternative losses:

1. Standard cross entropy.
2. Weighted cross entropy with inverse-frequency weights.
3. Ordinal EMD loss with inverse-frequency weights.

Training-set inverse-frequency class weights, normalized to mean 1:

| Class | Train count | Weight |
|---|---:|---:|
| Highly Disengage | 250 | 2.6762 |
| Disengage | 1,591 | 0.4205 |
| Engage | 6,099 | 0.1097 |
| Highly Engage | 843 | 0.7936 |

## 6. Train/Evaluation/Test Protocol

The final protocol is fixed in `main.py`:

| Usage | CMOSE source split key | Purpose |
|---|---|---|
| Train | `train` | Model fitting and feature normalizer fitting. |
| Evaluation | `unlabel` | Early stopping and best-checkpoint selection. |
| Test | `test` | Final held-out reporting only. |

Important wording:

The source split key is named `unlabel`, but the project uses it as a labeled evaluation/validation split because records have labels in `final_data_1.json`. In the thesis, describe it as "the CMOSE split named `unlabel`, used here as the evaluation split for checkpoint selection."

Training configuration used by the full comparison suite:

| Parameter | Value |
|---|---:|
| Maximum epochs | 400 |
| Batch size | 64 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Early-stopping patience | 10 epochs |
| Selection metric | Evaluation loss |
| Seed | 42 |
| Device in completed runs | CUDA |
| AMP | False |
| DataLoader workers | 4 |
| OpenFace target frames | 300 |
| I3D/fusion target frames | 75 |

Training loop details:

1. Build model from `src/models/models.py`.
2. Build loss from `src/training/train.py`.
3. Train mini-batches with Adam.
4. Evaluate every epoch on the evaluation split.
5. Save `best_model.pth` whenever evaluation loss improves.
6. Stop after 10 stale epochs.
7. Reload the best checkpoint.
8. Predict the held-out CMOSE test split.
9. Write `metrics.json`, `preprocessing_summary.json`, `selection_summary.json`, and `smote_summary.json`.

## 7. Model Architectures

### 7.1 OpenFace MLP: `openface_mlp`

Purpose:

Baseline model that ignores explicit temporal structure by flattening the full OpenFace frame-feature tensor.

Input:

`N x 300 x 709`

Architecture:

1. Flatten input.
2. Lazy linear layer to 256 hidden units.
3. ReLU.
4. Dropout 0.3.
5. Linear 256 -> 128.
6. ReLU.
7. Dropout 0.3.
8. Linear 128 -> 4 logits.

Thesis interpretation:

This baseline tests whether simple global feature aggregation is enough, without explicit sequence modeling.

### 7.2 OpenFace Temporal CNN / TCN: `temporal_cnn`, output folder `tcn`

Purpose:

Temporal model over frame-level OpenFace features using causal dilated residual convolutions.

Input:

`N x 300 x 709`

Architecture:

1. Transpose to channels-first `N x 709 x 300`.
2. TCN encoder with residual causal blocks:
   - Block 1: 709 -> 256 channels, kernel 3, dilation 1.
   - Block 2: 256 -> 128 channels, kernel 3, dilation 2.
   - Block 3: 128 -> 128 channels, kernel 3, dilation 4.
3. Each block uses weight-normalized 1-D convolutions, Chomp1d to remove right padding, ReLU, and dropout 0.2.
4. Adaptive average pooling over time.
5. Classifier: dropout 0.3, linear 128 -> 128, ReLU, dropout 0.3, linear 128 -> 4.

Thesis interpretation:

This model tests whether local temporal dynamics and longer-range dilated temporal context improve over flattened OpenFace features.

### 7.3 OpenFace LSTM: `lstm`

Purpose:

Sequential recurrent baseline over OpenFace frame features.

Input:

`N x 300 x 709`

Architecture:

1. Two-layer LSTM.
2. Input size 709.
3. Hidden size 256.
4. Dropout 0.3 between LSTM layers.
5. Use final hidden state from the last LSTM layer.
6. Classifier: dropout 0.3, linear 256 -> 128, ReLU, dropout 0.3, linear 128 -> 4.

Thesis interpretation:

This model tests recurrent temporal memory against convolutional and Transformer temporal models.

### 7.4 OpenFace Transformer: `transformer`

Purpose:

Self-attention model over frame-level OpenFace features.

Input:

`N x 300 x 709`

Architecture:

1. Linear projection 709 -> 128.
2. Sinusoidal positional encoding.
3. Transformer encoder:
   - 2 layers.
   - 4 attention heads.
   - Feed-forward dimension 256.
   - Dropout 0.2.
   - GELU activation.
   - Batch-first tensors.
4. Mean pooling across time.
5. LayerNorm.
6. Classifier: dropout 0.3, linear 128 -> 128, ReLU, dropout 0.3, linear 128 -> 4.

Thesis interpretation:

This model tests whether self-attention can capture informative temporal relationships in OpenFace sequences.

### 7.5 I3D MLP: `i3d_mlp`

Purpose:

Single-modality model using I3D video-action embeddings rather than facial behavior features.

Input:

`N x 75 x 1024`

Architecture:

Same `FlattenMLP` pattern as OpenFace MLP:

1. Flatten input.
2. Lazy linear to 256.
3. ReLU.
4. Dropout 0.3.
5. Linear 256 -> 128.
6. ReLU.
7. Dropout 0.3.
8. Linear 128 -> 4.

Thesis interpretation:

This model tests whether video/action embeddings encode stronger engagement cues than OpenFace-only facial features.

### 7.6 OpenFace TCN + I3D Fusion: `openface_tcn_i3d_fusion`

Purpose:

Multimodal sequence model combining OpenFace facial behavior and I3D video embeddings.

Inputs:

| Stream | Shape |
|---|---|
| OpenFace | `N x 75 x 709` |
| I3D | `N x 75 x 1024` |

Architecture:

OpenFace branch:

1. Transpose to `N x 709 x 75`.
2. TCN encoder with channels `[256, 128, 128]`.
3. Output transposed back to `N x 75 x 128`.

I3D branch:

1. Per-frame projection 1024 -> 256.
2. ReLU.
3. Dropout 0.3.
4. Linear 256 -> 128.
5. ReLU.
6. TCN encoder with channels `[128, 128]`.
7. Output `N x 75 x 128`.

Fusion:

1. Concatenate OpenFace and I3D encoded streams at each time step: `N x 75 x 256`.
2. Shared fusion MLP: linear 256 -> 128, ReLU, dropout 0.3, linear 128 -> 128, ReLU.
3. Auxiliary reconstruction heads:
   - Linear 128 -> 128 to reconstruct OpenFace encoded stream.
   - Linear 128 -> 128 to reconstruct I3D encoded stream.
4. Auxiliary loss is MSE reconstruction loss multiplied by 0.1.
5. Mean pool fused sequence over time.
6. Classifier: dropout 0.3, linear 128 -> 128, ReLU, dropout 0.3, linear 128 -> 4.

Thesis interpretation:

The fusion model tests whether combining facial behavior and video-action embeddings improves recognition. In final results, fusion did not outperform the best single-modality I3D model and did not outperform the best OpenFace temporal models under CE on macro F1. This is a key negative finding, likely affected by domain shift, modality alignment, class imbalance, and the simplicity of the fusion mechanism.

## 8. Loss Functions

### 8.1 Cross Entropy: `cross_entropy`, folder `ce`

Standard multi-class cross entropy. It optimizes overall classification likelihood and tends to favor the majority class when labels are imbalanced.

### 8.2 Weighted Cross Entropy: `weighted_cross_entropy`, folder `weighted_ce`

Cross entropy with inverse-frequency class weights from the CMOSE training split. It gives much more weight to `Highly Disengage` and less weight to `Engage`.

Expected behavior:

It often improves balanced metrics or minority-class behavior but can reduce overall accuracy because it discourages simply predicting the dominant `Engage` class.

### 8.3 Ordinal EMD Loss: `ordinal`

Ordinal loss based on squared distance between predicted and target cumulative class distributions. The implementation:

1. Applies softmax to logits.
2. Computes predicted cumulative distribution function over classes.
3. Converts targets to one-hot vectors.
4. Computes target cumulative distribution.
5. Averages squared CDF distance per sample.
6. Applies inverse-frequency class weights.

Expected behavior:

This loss treats the class order as meaningful. It should penalize far-away mistakes more than adjacent mistakes. It often improves macro accuracy but can reduce raw accuracy and weighted F1.

### 8.4 Focal Loss

`FocalLoss` exists in the code and can be selected by `main.py`, but it was not part of the final 18-run comparison suite saved in `outputs/training_log`.

## 9. Evaluation Metrics

Reported metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Fraction of exactly correct predictions. Sensitive to the majority class. |
| Macro accuracy | Balanced accuracy, average recall across classes. Better for imbalance. |
| Macro F1 | Unweighted average F1 across the four classes. Main fair-summary metric. |
| Weighted F1 | F1 averaged by support. Tracks majority-class performance more strongly. |
| MAE | Mean absolute class-ID error; lower is better. Uses ordinal label distance. |
| MSE | Mean squared class-ID error; lower is better and penalizes far mistakes more. |
| Confusion matrix | Rows are true labels, columns are predictions. |
| Classification report | Precision, recall, F1, and support per class. |

Recommended primary result metric for the thesis:

Use macro F1 as the main model comparison metric because of class imbalance, while also reporting accuracy and MAE/MSE to show practical correctness and ordinal severity. Accuracy alone would overstate models that over-predict `Engage`.

## 10. CMOSE Test Results

All rows are held-out CMOSE test results on 1,221 samples.

| Run | Accuracy | Macro Acc | Macro F1 | Weighted F1 | MAE | MSE | Best Epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| `i3d_mlp/ce` | 0.7723 | 0.5392 | 0.5960 | 0.7549 | 0.2473 | 0.2883 | 11 |
| `i3d_mlp/weighted_ce` | 0.6880 | 0.6176 | 0.5686 | 0.7043 | 0.3505 | 0.4357 | 16 |
| `tcn/ce` | 0.7576 | 0.4990 | 0.5428 | 0.7368 | 0.2604 | 0.2981 | 26 |
| `transformer/ce` | 0.7518 | 0.4931 | 0.5411 | 0.7325 | 0.2744 | 0.3301 | 25 |
| `openface_tcn_i3d_fusion/weighted_ce` | 0.6486 | 0.6139 | 0.5268 | 0.6732 | 0.4021 | 0.5119 | 6 |
| `tcn/ordinal` | 0.6167 | 0.6214 | 0.5063 | 0.6451 | 0.4373 | 0.5569 | 41 |
| `i3d_mlp/ordinal` | 0.6077 | 0.6173 | 0.5044 | 0.6355 | 0.4472 | 0.5684 | 11 |
| `tcn/weighted_ce` | 0.6282 | 0.6124 | 0.4989 | 0.6521 | 0.4333 | 0.5659 | 22 |
| `transformer/weighted_ce` | 0.5823 | 0.5831 | 0.4781 | 0.6118 | 0.4767 | 0.6110 | 37 |
| `openface_tcn_i3d_fusion/ordinal` | 0.5586 | 0.5707 | 0.4776 | 0.5873 | 0.4816 | 0.5651 | 5 |
| `lstm/weighted_ce` | 0.5971 | 0.5507 | 0.4727 | 0.6228 | 0.4578 | 0.5741 | 32 |
| `transformer/ordinal` | 0.5266 | 0.5879 | 0.4456 | 0.5588 | 0.5405 | 0.6863 | 26 |
| `openface_tcn_i3d_fusion/ce` | 0.7453 | 0.4318 | 0.4396 | 0.7200 | 0.2981 | 0.3980 | 4 |
| `openface_mlp/ce` | 0.7248 | 0.3851 | 0.4241 | 0.6760 | 0.2973 | 0.3415 | 35 |
| `openface_mlp/weighted_ce` | 0.4996 | 0.5113 | 0.4100 | 0.5320 | 0.5708 | 0.7183 | 29 |
| `lstm/ce` | 0.7011 | 0.3788 | 0.4092 | 0.6611 | 0.3227 | 0.3718 | 10 |
| `lstm/ordinal` | 0.4373 | 0.5247 | 0.3805 | 0.4675 | 0.6413 | 0.8116 | 15 |
| `openface_mlp/ordinal` | 0.5471 | 0.3388 | 0.3389 | 0.5556 | 0.4840 | 0.5479 | 35 |

Best run per model by macro F1:

| Model Family | Best Run | Accuracy | Macro Acc | Macro F1 | Weighted F1 | MAE |
|---|---|---:|---:|---:|---:|---:|
| I3D MLP | `i3d_mlp/ce` | 0.7723 | 0.5392 | 0.5960 | 0.7549 | 0.2473 |
| OpenFace TCN | `tcn/ce` | 0.7576 | 0.4990 | 0.5428 | 0.7368 | 0.2604 |
| OpenFace Transformer | `transformer/ce` | 0.7518 | 0.4931 | 0.5411 | 0.7325 | 0.2744 |
| Fusion | `openface_tcn_i3d_fusion/weighted_ce` | 0.6486 | 0.6139 | 0.5268 | 0.6732 | 0.4021 |
| OpenFace LSTM | `lstm/weighted_ce` | 0.5971 | 0.5507 | 0.4727 | 0.6228 | 0.4578 |
| OpenFace MLP | `openface_mlp/ce` | 0.7248 | 0.3851 | 0.4241 | 0.6760 | 0.2973 |

Best overall run:

`i3d_mlp/ce` is strongest by accuracy, macro F1, weighted F1, MAE, and MSE. It is not strongest by macro accuracy; the highest macro accuracy is `tcn/ordinal` at 0.6214, narrowly above `i3d_mlp/weighted_ce` at 0.6176 and `i3d_mlp/ordinal` at 0.6173.

Confusion matrix for best CMOSE run, `i3d_mlp/ce`:

Rows are true labels; columns are predicted labels in class order `[HD, DE, EG, HE]`.

| True \ Pred | HD | DE | EG | HE | Support |
|---|---:|---:|---:|---:|---:|
| Highly Disengage | 12 | 9 | 13 | 1 | 35 |
| Disengage | 3 | 96 | 119 | 3 | 221 |
| Engage | 4 | 50 | 781 | 12 | 847 |
| Highly Engage | 0 | 2 | 62 | 54 | 118 |

Per-class report for `i3d_mlp/ce`:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Highly Disengage | 0.63 | 0.34 | 0.44 | 35 |
| Disengage | 0.61 | 0.43 | 0.51 | 221 |
| Engage | 0.80 | 0.92 | 0.86 | 847 |
| Highly Engage | 0.77 | 0.46 | 0.57 | 118 |

Interpretation:

The best model is very strong on the majority `Engage` class and reasonably precise on the two extremes, but recall for minority classes remains limited. Many `Disengage` and `Highly Engage` errors are pulled toward `Engage`, which is expected given the class imbalance.

## 11. Main Experimental Findings

### 11.1 I3D Is the Strongest Single Modality

The best overall result is `i3d_mlp/ce`, with macro F1 0.5960 and accuracy 0.7723. This suggests that the 1024-dimensional I3D embeddings contain strong engagement-relevant information, likely including body motion, activity, and visual context beyond face geometry.

Thesis claim:

I3D embeddings provide the most effective representation in this CMOSE comparison, outperforming all OpenFace-only and fusion runs on macro F1 and accuracy.

### 11.2 Temporal OpenFace Models Improve Over the OpenFace MLP

Under CE:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| OpenFace MLP | 0.7248 | 0.4241 |
| OpenFace TCN | 0.7576 | 0.5428 |
| OpenFace Transformer | 0.7518 | 0.5411 |
| OpenFace LSTM | 0.7011 | 0.4092 |

The TCN and Transformer substantially improve macro F1 over the flattened OpenFace MLP, indicating that temporal structure matters. The LSTM underperforms TCN and Transformer in this configuration.

Thesis claim:

Explicit temporal modeling is beneficial for OpenFace features, especially with convolutional and attention-based sequence encoders.

### 11.3 Weighted and Ordinal Losses Trade Accuracy for Balanced Behavior

Examples:

| Model | CE Macro Acc | Weighted CE Macro Acc | Ordinal Macro Acc |
|---|---:|---:|---:|
| OpenFace MLP | 0.3851 | 0.5113 | 0.3388 |
| TCN | 0.4990 | 0.6124 | 0.6214 |
| Transformer | 0.4931 | 0.5831 | 0.5879 |
| I3D MLP | 0.5392 | 0.6176 | 0.6173 |
| Fusion | 0.4318 | 0.6139 | 0.5707 |

Weighted/ordinal losses often improve macro accuracy because they force attention to minority classes. However, they frequently reduce overall accuracy and weighted F1. For example, `tcn/ordinal` has the highest macro accuracy (0.6214) but lower macro F1 (0.5063) and lower accuracy (0.6167) than `tcn/ce`.

Thesis claim:

Loss choice controls a tradeoff between majority-class accuracy and balanced class recall. CE is best for top-line accuracy and macro F1 in this setup, while weighted/ordinal losses are useful when minority-class recall is prioritized.

### 11.4 Fusion Did Not Beat the Best Single-Modality Models

Best fusion run:

`openface_tcn_i3d_fusion/weighted_ce`: accuracy 0.6486, macro F1 0.5268.

Fusion CE has higher accuracy, 0.7453, but much lower macro F1, 0.4396, due to poorer balanced class behavior.

Interpretation:

The implemented fusion strategy is not enough to outperform I3D alone or the best OpenFace temporal models. Possible reasons:

1. I3D already captures much of the useful signal.
2. The CMOSE I3D embeddings are per-sample vectors materialized from JSON and expanded/resampled to 75 frames, limiting true temporal complementarity.
3. Fusion adds parameters and optimization complexity.
4. OpenFace and I3D streams may have different noise and domain-shift profiles.
5. Weighted loss helps fusion macro metrics but hurts accuracy.

Thesis claim:

Multimodal fusion is not automatically beneficial; careful temporal alignment and modality-specific robustness are important.

## 12. Private Dataset and Domain-Shift Results

The private evaluation is not the main benchmark result. It is an out-of-domain stress test using 368 manually labeled accepted private clips.

Private supervised metrics:

| Run | Accuracy | Macro Acc | Macro F1 | Weighted F1 | MAE | MSE |
|---|---:|---:|---:|---:|---:|---:|
| `openface_mlp/ce` | 0.7065 | 0.2632 | 0.2360 | 0.6002 | 0.3234 | 0.3832 |
| `lstm/ce` | 0.6712 | 0.2490 | 0.2259 | 0.5849 | 0.3641 | 0.4348 |
| `transformer/ce` | 0.6495 | 0.3166 | 0.3150 | 0.6001 | 0.3967 | 0.4891 |
| `tcn/ordinal` | 0.6522 | 0.2812 | 0.2721 | 0.5983 | 0.4212 | 0.5679 |
| `tcn/weighted_ce` | 0.6332 | 0.3067 | 0.3055 | 0.6065 | 0.4565 | 0.6522 |
| `tcn/ce` | 0.5842 | 0.3099 | 0.3219 | 0.5766 | 0.4484 | 0.5136 |
| `i3d_mlp/ce` | 0.4728 | 0.2972 | 0.2432 | 0.4962 | 0.6658 | 0.9973 |
| `openface_tcn_i3d_fusion/weighted_ce` | 0.3424 | 0.2808 | 0.2156 | 0.3832 | 0.7826 | 1.0543 |
| `transformer/weighted_ce` | 0.3587 | 0.3886 | 0.2811 | 0.4020 | 0.8098 | 1.1630 |
| `openface_mlp/weighted_ce` | 0.2147 | 0.3079 | 0.1510 | 0.1783 | 0.9293 | 1.2283 |
| `i3d_mlp/weighted_ce` | 0.2120 | 0.2761 | 0.1310 | 0.1993 | 0.9701 | 1.3995 |
| `lstm/weighted_ce` | 0.3342 | 0.2403 | 0.2055 | 0.3851 | 0.7880 | 1.0543 |
| `openface_tcn_i3d_fusion/ordinal` | 0.3859 | 0.2592 | 0.1961 | 0.4186 | 0.7391 | 0.9891 |
| `openface_mlp/ordinal` | 0.3967 | 0.2245 | 0.1864 | 0.4248 | 0.7065 | 0.9130 |
| `lstm/ordinal` | 0.2310 | 0.2740 | 0.1813 | 0.2368 | 0.9049 | 1.1984 |
| `transformer/ordinal` | 0.1712 | 0.3466 | 0.1890 | 0.1225 | 0.9701 | 1.2582 |
| `openface_tcn_i3d_fusion/ce` | 0.1712 | 0.2383 | 0.0973 | 0.1225 | 0.9511 | 1.1957 |
| `i3d_mlp/ordinal` | 0.1304 | 0.2491 | 0.0628 | 0.0521 | 1.0761 | 1.5435 |

CE-run private-minus-CMOSE performance drops:

| CE Run | Accuracy Drop | Macro Acc Drop | Macro F1 Drop | Weighted F1 Drop | MAE Increase |
|---|---:|---:|---:|---:|---:|
| `openface_mlp/ce` | -0.0183 | -0.1219 | -0.1881 | -0.0758 | +0.0261 |
| `tcn/ce` | -0.1733 | -0.1890 | -0.2209 | -0.1602 | +0.1879 |
| `lstm/ce` | -0.0299 | -0.1298 | -0.1833 | -0.0762 | +0.0414 |
| `transformer/ce` | -0.1024 | -0.1765 | -0.2261 | -0.1324 | +0.1224 |
| `i3d_mlp/ce` | -0.2995 | -0.2420 | -0.3528 | -0.2587 | +0.4184 |
| `openface_tcn_i3d_fusion/ce` | -0.5741 | -0.1935 | -0.3423 | -0.5974 | +0.6530 |

Private prediction distribution examples:

1. `openface_mlp/ce` predicts `Engage` for 98.4 percent of accepted private clips. This explains why its private accuracy is high while macro F1 is weak.
2. `lstm/ce` predicts `Engage` for 93.2 percent of private clips.
3. `tcn/ce` predicts a more diverse distribution: 1.2 percent HD, 23.1 percent DE, 72.4 percent EG, 3.3 percent HE.
4. `i3d_mlp/ce` shifts toward `Highly Engage` on private clips: 3.3 percent HD, 2.6 percent DE, 54.2 percent EG, 40.0 percent HE.
5. Fusion CE collapses mostly to `Disengage` on private clips: 87.6 percent DE.

Interpretation:

The private set exposes calibration and domain-transfer weaknesses. High private accuracy for some models can be misleading because the private labels are dominated by `Engage`, and some models collapse toward majority-class predictions. Macro F1 and prediction distribution are essential in the thesis discussion.

## 13. Feature-Space Domain Shift

Feature-space comparison uses only extracted features, not labels or model outputs. Private data is filtered by `data/private/accepted.csv`.

Coverage:

| Item | Count |
|---|---:|
| Accepted private clips from manifest | 428 of 808 |
| Private accepted OpenFace CSV files | 428 |
| Private sampled OpenFace rows | 54,784 |
| CMOSE OpenFace CSV files | 12,197 |
| CMOSE sampled OpenFace rows | 60,000 |
| Common OpenFace feature columns | 709 |
| Private I3D vectors | 428 |
| CMOSE I3D vectors | 12,197 |
| I3D dimension | 1024 |

Overall feature-space answer:

1. OpenFace distributions are strongly shifted when using all extractor rows: mean standardized Wasserstein 0.955 and mean KS 0.495.
2. I3D distributions show comparable marginal shift and clear centroid movement: centroid cosine distance 0.204, centroid RMS shift 1.248, and mean standardized Wasserstein 0.971.
3. The private set is not simply a smaller sample of the same CMOSE feature distribution.

OpenFace group distances, all extractor rows:

| Group | Features | Mean Wasserstein z | Mean KS | Mean Abs Mean Shift z |
|---|---:|---:|---:|---:|
| face_landmark_2d | 136 | 1.678 | 0.882 | 1.664 |
| eye_landmark_2d | 112 | 1.663 | 0.869 | 1.651 |
| pdm | 40 | 0.771 | 0.381 | 0.361 |
| eye_landmark_3d | 168 | 0.615 | 0.308 | 0.478 |
| face_landmark_3d | 204 | 0.571 | 0.299 | 0.391 |
| gaze | 8 | 0.422 | 0.248 | 0.342 |
| head_pose | 6 | 0.238 | 0.206 | 0.114 |
| au_presence | 18 | 0.231 | 0.083 | 0.231 |
| au_intensity | 17 | 0.169 | 0.071 | 0.157 |

OpenFace extraction quality:

| Dataset | Mean Confidence | Median Confidence | Confidence >= 0.80 | Confidence >= 0.95 | Success Rate |
|---|---:|---:|---:|---:|---:|
| Private | 0.834 | 0.980 | 0.856 | 0.640 | 0.863 |
| CMOSE | 0.923 | 0.980 | 0.960 | 0.703 | 0.963 |

High-confidence sensitivity:

After filtering to `success == 1` and `confidence >= 0.80`, the OpenFace shift remains strong. Valid-only mean standardized Wasserstein is 0.979 and mean KS is 0.497. The largest shifted groups remain 2-D face landmarks and 2-D eye landmarks.

I3D distances:

| Metric | Value |
|---|---:|
| Raw centroid cosine distance | 0.204 |
| Raw centroid L2 distance | 2.925 |
| Pooled-z centroid L2 distance | 39.951 |
| Pooled-z centroid RMS shift per dimension | 1.248 |
| Mean standardized Wasserstein | 0.971 |
| Median standardized Wasserstein | 0.712 |
| Mean KS statistic | 0.402 |
| Median KS statistic | 0.377 |

Interpretation for thesis:

Private and CMOSE labels look similar, but features do not. The largest OpenFace differences are in geometric/pose-like groups, especially 2-D landmarks. These may reflect camera framing, resolution, face scale, cropping, and capture conditions rather than only behavior. This explains why models trained on CMOSE can degrade or collapse on private recordings.

## 14. Visual Assets and Tables for Thesis

Use these artifacts directly when writing the thesis and slides:

### Dataset Figures

| Artifact | Use |
|---|---|
| `outputs/dataset_analysis/cmose/class_distribution_barchart.png` | Show CMOSE imbalance. |
| `outputs/dataset_analysis/cmose/data_split_diagram.png` | Show train/evaluation/test split. |
| `outputs/dataset_analysis/private/class_distribution_barchart.png` | Show private manual-label distribution. |
| `outputs/dataset_analysis/comparison/class_distribution_stacked_barchart.png` | Compare CMOSE and private class distributions. |

### Feature-Space Figures

| Artifact | Use |
|---|---|
| `outputs/dataset_analysis/comparison/domain_difference/feature_space_dataset_comparison.md` | Textual domain-shift report. |
| `outputs/dataset_analysis/comparison/domain_difference/feature_space_openface_groups.csv` | OpenFace group distance table. |
| `outputs/dataset_analysis/comparison/domain_difference/feature_space_visualizations/tsne_openface.png` | OpenFace CMOSE/private t-SNE. |
| `outputs/dataset_analysis/comparison/domain_difference/feature_space_visualizations/tsne_i3d.png` | I3D CMOSE/private t-SNE. |
| `outputs/dataset_analysis/comparison/domain_difference/feature_space_visualizations/tsne_openface_plus_i3d.png` | Fused feature t-SNE. |

### Training and Model Assessment Figures

| Artifact | Use |
|---|---|
| `outputs/model_assessment/cmose_testset/main_metrics_all_models_losses.png` | Main CMOSE result comparison across models/losses. |
| `outputs/model_assessment/cmose_testset/metric_heatmap_ce_models.png` | Clean CE model comparison heatmap. |
| `outputs/model_assessment/cmose_testset/confusion_matrices_all_runs.png` | All CMOSE confusion matrices. |
| `outputs/model_assessment/cmose_testset/summary_table_all_runs.csv` | Exact metrics for all 18 CMOSE runs. |
| `outputs/model_assessment/cmose_testset/summary_table_best_per_model.csv` | Best run per model family. |
| `outputs/model_assessment/private/main_metrics_all_models_losses.png` | Private metrics across runs. |
| `outputs/model_assessment/comparison/performance_drop_chart_ce_models.png` | CE model performance drop from CMOSE to private. |
| `outputs/model_assessment/comparison/ce_models_cmose_private_metric_lines.png` | Line comparison of CMOSE vs private CE metrics. |

### Per-Run Artifacts

For any run, use:

`outputs/training_log/<model>/<loss>/`

Important files:

| File | Use |
|---|---|
| `best_model.pth` | Best saved checkpoint. |
| `metrics.json` | Full config, history, metrics, confusion matrix, and classification report. |
| `preprocessing_summary.json` | Tensor shapes and normalization details. |
| `selection_summary.json` | Split usage and assumptions. |
| `smote_summary.json` | Class counts and SMOTE disabled record. |
| `training_curves.png` | Loss and evaluation curves. |
| `evaluation_metrics.png` | Evaluation metric curves. |
| `report.md` | Short per-run summary. |

## 15. Reproduction Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a single model:

```bash
python main.py --model temporal_cnn
python main.py --model openface_mlp
python main.py --model lstm
python main.py --model transformer
python main.py --model i3d_mlp
python main.py --model openface_tcn_i3d_fusion
```

Run the final full comparison suite:

```bash
python scripts/compare_naive_models.py
```

Useful comparison-suite defaults:

```text
--epochs 400
--batch_size 64
--lr 1e-4
--patience 10
--target_frames 300
--fusion_frames 75
--naive_losses cross_entropy weighted_cross_entropy ordinal
```

Regenerate training-log visualizations and CMOSE assessment charts:

```bash
python scripts/visualize_models_outputs.py
```

Regenerate dataset analysis:

```bash
python scripts/visualize_dataset_analysis.py
python scripts/feature_space_dataset_comparison.py
python scripts/visualize_feature_space_domains.py
```

Generate raw CMOSE/private predictions from saved checkpoints:

```bash
python -m src.feature_analysis.run_domain_shift_analysis
```

Regenerate private/manual-label and CMOSE-vs-private assessment charts:

```bash
python scripts/visualize_model_assessment.py
```

## 16. Suggested Thesis Structure

### Chapter 1: Introduction

Suggested content:

1. Motivation: online learning, engagement as an important but hard-to-measure state.
2. Problem: video-based engagement recognition has class imbalance, temporal dynamics, feature noise, and deployment/domain-shift challenges.
3. Goal: compare OpenFace facial features, I3D video embeddings, temporal models, losses, and multimodal fusion on CMOSE.
4. Contributions: controlled pipeline, six-model comparison, three-loss comparison, private dataset stress test, domain-shift analysis.
5. Research questions from Section 2.

### Chapter 2: Background and Related Work

Use collected PDFs in `documents/references/`. Likely reference themes:

1. CMOSE dataset and engagement labels: `documents/CMOSE-dataset.pdf`.
2. DAiSEE and engagement detection literature: `documents/references/DAiSEE.pdf`.
3. OpenFace plus sequence models: `documents/references/OpenFace+BiLSTM.pdf`.
4. CNN/OpenFace/PCA/SVD/SMOTE work: `documents/references/CNN OpenFace PCA SVD SMOTE.pdf`.
5. EfficientNetV2 plus LSTM for engagement: `documents/references/EfficientNetV2+LSTM.pdf`.
6. Classical baselines such as LR, SVM, MLP, KNN, XGB: `documents/references/LR, SVM, MLP, KNN, XGB.pdf`.
7. Survey context: `documents/references/Students_Engagement_Detection_Based_on_Computer_Vision_A_Systematic_Literature_Review.pdf`.

Suggested discussion points:

1. Engagement labels are ordinal.
2. Face-based methods use landmarks, gaze, pose, and action units.
3. Video-based methods use spatial-temporal embeddings.
4. Sequence models include CNN/TCN, LSTM, and Transformer encoders.
5. Imbalance handling includes class weighting, sampling, and specialized losses.
6. Cross-domain generalization is a major limitation.

### Chapter 3: Methodology

Suggested sections:

1. Dataset and label mapping.
2. CMOSE split protocol.
3. OpenFace feature extraction and resampling.
4. I3D embedding preparation.
5. Train-only normalization.
6. Model architectures.
7. Loss functions.
8. Training procedure and early stopping.
9. Evaluation metrics.
10. Private dataset and domain-shift analysis.

Important methodology wording:

"All models use the same CMOSE train/evaluation/test split. The train split is used for model fitting and normalization-statistic fitting. The split named `unlabel` in the source metadata is used as the evaluation split for early stopping and checkpoint selection. The test split is held out until final reporting."

### Chapter 4: Experiments

Suggested content:

1. Hardware/software: PyTorch, CUDA runs, Python dependencies from `requirements.txt`.
2. Experiment matrix: six models x three losses = 18 runs.
3. Hyperparameters: 400 epochs, batch 64, lr 1e-4, patience 10, seed 42.
4. Output artifacts and reproducibility commands.
5. Private dataset evaluation as out-of-domain test.

### Chapter 5: Results and Discussion

Suggested result order:

1. Dataset imbalance table.
2. Main 18-run CMOSE result table.
3. Best per model table.
4. Confusion matrix for `i3d_mlp/ce`.
5. OpenFace temporal model comparison.
6. Loss-function tradeoff analysis.
7. Fusion model discussion.
8. Private dataset performance.
9. Feature-space domain-shift analysis.

Key discussion statements:

1. I3D MLP with CE is the best overall CMOSE model.
2. TCN and Transformer are the strongest OpenFace-only models.
3. Weighted and ordinal losses improve balanced recall-style metrics but often reduce accuracy.
4. Fusion does not guarantee improvement; in this implementation it underperforms I3D alone.
5. Private evaluation reveals domain shift and prediction collapse for some runs.
6. Macro F1 and macro accuracy should be emphasized because class imbalance makes accuracy alone misleading.

### Chapter 6: Conclusion

Suggested conclusion:

This project shows that precomputed I3D video embeddings outperform OpenFace-only facial features for four-level engagement classification on the selected CMOSE protocol, while temporal OpenFace models improve substantially over a flattened OpenFace MLP. Class-weighted and ordinal losses improve balanced metrics but reduce top-line accuracy. A simple OpenFace-I3D fusion architecture does not outperform the best single-modality model. Private data evaluation demonstrates that similar label distributions do not imply similar feature distributions, and domain shift remains a major obstacle for deployment.

Future work:

1. Calibrated domain adaptation between CMOSE and private recordings.
2. Stronger multimodal fusion, such as cross-attention or late-fusion ensembles.
3. Better temporal I3D sequences instead of per-sample embeddings expanded to fixed frames.
4. Larger and more balanced private labels.
5. Subject-independent or video-independent split analysis if metadata supports it.
6. Calibration analysis and thresholding for practical deployment.

## 17. Suggested Slide Deck Story

Recommended 12-slide structure:

1. Title: student engagement recognition from CMOSE video features.
2. Motivation: engagement matters, manual observation does not scale.
3. Problem: imbalanced ordinal labels, noisy features, temporal behavior, domain shift.
4. Dataset: CMOSE counts and label distribution; private dataset as out-of-domain check.
5. Pipeline: OpenFace/I3D features -> resampling -> train-only normalization -> model -> metrics.
6. Models: MLP, TCN, LSTM, Transformer, I3D MLP, fusion diagram.
7. Training protocol: train/eval/test, losses, hyperparameters.
8. Main CMOSE result table: emphasize `i3d_mlp/ce`, TCN, Transformer.
9. Confusion matrix: best model still struggles on minority classes.
10. Loss tradeoff: weighted/ordinal improves macro accuracy but lowers accuracy.
11. Private/domain-shift result: feature shift and performance drop.
12. Conclusion: I3D best, temporal OpenFace helps, fusion not automatically better, domain shift is the next challenge.

Best figures for slides:

1. `outputs/dataset_analysis/cmose/class_distribution_barchart.png`
2. `outputs/model_assessment/cmose_testset/metric_heatmap_ce_models.png`
3. `outputs/model_assessment/cmose_testset/main_metrics_all_models_losses.png`
4. `outputs/model_assessment/cmose_testset/confusion_matrices/i3d_mlp/ce/confusion_matrix.png`
5. `outputs/model_assessment/comparison/performance_drop_chart_ce_models.png`
6. `outputs/dataset_analysis/comparison/domain_difference/feature_space_visualizations/tsne_openface_plus_i3d.png`

## 18. Caveats and Limitations

Use these honestly in the thesis:

1. The dataset is highly imbalanced, with `Engage` around 69 percent of CMOSE and 71 percent of labeled private data.
2. The CMOSE source split named `unlabel` is used as evaluation because labels are available in the metadata. This should be explained clearly.
3. Private supervised metrics use only 368 labeled samples, not all 428 accepted clips.
4. Private labels are manual and may differ in labeling criteria from CMOSE.
5. Some models achieve high private accuracy by predicting mostly `Engage`; macro F1 reveals weaker balanced performance.
6. OpenFace 2-D landmark shifts may reflect camera/framing differences, not necessarily engagement behavior.
7. I3D features materialized from CMOSE JSON are 1024-dimensional per-sample embeddings; if they are expanded to 75 frames, temporal variation may be limited.
8. Fusion was simple and may not represent the best possible multimodal approach.
9. The repository contains no active source test files under `tests/`, only cached bytecode, so validation is primarily through generated experiment artifacts rather than a maintained automated test suite.

## 19. Exact Claims Safe to Make

Safe claims based on saved outputs:

1. The final comparison evaluates 18 CMOSE runs: six models across CE, weighted CE, and ordinal losses.
2. `i3d_mlp/ce` is the best held-out CMOSE run by accuracy, macro F1, weighted F1, MAE, and MSE.
3. `tcn/ordinal` has the highest CMOSE macro accuracy.
4. TCN and Transformer substantially outperform OpenFace MLP on OpenFace-only macro F1 under CE.
5. The implemented OpenFace-I3D fusion model does not outperform the best I3D-only model.
6. Private evaluation shows meaningful performance drops for CE runs, especially I3D and fusion.
7. Feature-space analysis shows measurable CMOSE/private domain shift in both OpenFace and I3D representations.
8. Weighted and ordinal losses often improve balanced accuracy but do not consistently improve macro F1 or overall accuracy.

Claims to avoid or qualify:

1. Do not claim the model is deployment-ready; private-domain results show transfer issues.
2. Do not claim fusion is generally worse than single modality; only this implemented fusion under this protocol underperforms.
3. Do not claim OpenFace features are unhelpful; TCN and Transformer OpenFace models are strong.
4. Do not claim private data has the same distribution as CMOSE just because label proportions are similar.
5. Do not treat accuracy as the sole result because the class distribution is imbalanced.

## 20. Final Thesis Abstract Draft

This thesis studies automatic student engagement recognition from video using the CMOSE dataset. Engagement is modeled as a four-class ordinal classification problem: Highly Disengage, Disengage, Engage, and Highly Engage. The project builds a reproducible training and evaluation pipeline using OpenFace facial behavior features and I3D video embeddings. Six neural classifiers are compared: a flattened OpenFace MLP, an OpenFace temporal convolutional network, an OpenFace LSTM, an OpenFace Transformer, an I3D MLP, and an OpenFace-I3D fusion model. Each model is trained with cross entropy, weighted cross entropy, and an ordinal Earth-Mover-style loss. All experiments use the same protocol: train on the CMOSE train split, select checkpoints on the split named `unlabel`, and report final performance on the held-out test split. Results show that the I3D MLP trained with cross entropy performs best overall, reaching 0.7723 accuracy and 0.5960 macro F1 on the CMOSE test set. Among OpenFace-only models, temporal convolution and Transformer encoders substantially outperform the flattened MLP, confirming the value of temporal modeling. Weighted and ordinal losses improve balanced recall-style behavior in several cases but often reduce overall accuracy. The implemented multimodal fusion model does not outperform the best single-modality I3D model. Additional evaluation on a manually labeled private dataset and feature-space distance analysis reveal significant domain shift, showing that benchmark performance does not directly guarantee robust transfer to private recordings. The study concludes that I3D embeddings and temporal OpenFace models are effective for CMOSE engagement recognition, while class imbalance and domain shift remain central challenges for practical deployment.

