# Implementation Specification
## Cross-Domain Engagement Recognition — CMOSE Baseline + Domain Adaptation

**Stack**: Python 3.10+, PyTorch 2.x, vast.ai GPU (recommend ≥16GB VRAM, e.g. RTX 3090/4090)  
**Libraries**: `openface` (pre-built binary), `mmaction2` or direct I3D weights, `timm`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `umap-learn`, `tqdm`

---

## Phase 0 — Environment & Data Setup

### 0.1 Directory Structure
```
project/
|-- data/
|   |-- CMOSE/
|   |   |-- openface-features/
|   |   |   `-- secondFeature/   # OpenFace .csv per clip
|   |   |-- labels.csv            # clip_id, label (0=HD,1=DE,2=EG,3=HE), split
|   |   `-- final_data_1.json     # split/label metadata + I3D embeds
|   `-- private/
|       |-- clips/                # raw private .mp4 clips
|       `-- features/
|           |-- openface/         # generated private OpenFace .csv
|           `-- i3d/              # generated private I3D .npy
|-- src/
|   |-- features/
|   |   |-- extract_openface.py
|   |   `-- extract_i3d.py
|   |-- models/
|   |   |-- backbone.py           # TCN, MLP blocks
|   |   |-- cmose_baseline.py     # full CMOSE video-only model
|   |   `-- naive_models.py       # all your prior models
|   |-- training/
|   |   |-- mocorank.py           # MocoRank loss + momentum encoder
|   |   `-- train.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   `-- domain_analysis.py
|   `-- pseudo_label/
|       |-- strategy_confidence.py
|       |-- strategy_teacher_student.py
|       `-- strategy_knn_propagation.py
|-- configs/
|   `-- baseline.yaml
|-- notebooks/
|   `-- analysis.ipynb
`-- checkpoints/
```
### 0.1A Corrected Dataset Assumptions (Use This as Ground Truth)

**CMOSE (in this repository)**
- Do **not** assume raw CMOSE clips are present.
- Assume OpenFace CSVs are already available at `data/CMOSE/openface-features/secondFeature/*.csv`.
- Assume `data/CMOSE/final_data_1.json` is available and contains split/label metadata plus I3D embeds.
- Materialize `data/CMOSE/features/i3d/*.npy` from `final_data_1.json` when needed.
- Verified from current data:
  - `labels.csv` rows: 12,197
  - OpenFace CSV count: 12,197
  - `final_data_1.json` entries: 12,197

**Private dataset (in this repository)**
- Assume raw private clips are in `data/private/clips/*.mp4`.
- Do **not** assume private OpenFace/I3D features already exist.
- Maintain `data/private/accepted.csv` as the clip-acceptance registry used by all later phases.

### 0.1B Private Dataset Preprocessing Gate (Mandatory)

1. Run OpenFace extraction on **all** `data/private/clips/*.mp4`.
2. For each clip, run QA checks from OpenFace output (parse success, non-empty rows, minimum successful frames).
3. Write/update `data/private/accepted.csv` with at least:
   - `clip_id`
   - `clip_path`
   - `openface_csv`
   - `is_accepted` (0/1)
   - `reject_reason` (empty for accepted clips)
4. Run I3D extraction **only** on rows where `is_accepted=1`.
5. All downstream analysis, inference, and pseudo-labeling must use only accepted clips (`is_accepted=1`).

### 0.2 Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mmaction2  # or use a standalone I3D repo
pip install timm scikit-learn umap-learn pandas seaborn tqdm pyyaml
```

### 0.3 OpenFace Setup (on vast.ai Linux)
```bash
# OpenFace requires pre-built binary on Ubuntu
# Follow: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation
# After build, the key binary is: OpenFace/build/bin/FeatureExtraction
```

---

## Phase 1 — Reproduce CMOSE Baseline (Video-Only)

### 1.1 Feature Extraction — OpenFace

**Script**: `src/features/extract_openface.py`

**Repository-specific note**:
- CMOSE OpenFace CSVs are pre-extracted and already stored under `data/CMOSE/openface-features/secondFeature/`.
- OpenFace extraction in this thesis workflow is mandatory for private clips, followed by acceptance filtering via `data/private/accepted.csv`.

**Actual OpenFace dimensions in this repository**:
- Raw OpenFace CSV columns per frame: **714**
- Model-selected columns per frame (used by CMOSE baseline): **49**
  - Gaze: 8
  - Head pose: 6
  - AU intensities (`*_r`): 17
  - AU presence (`*_c`): 18

For each clip, run:
```bash
./FeatureExtraction -f <clip.mp4> -out_dir <output_dir> -aus -gaze -pose -2Dfp
```

**Output per clip**: a `.csv` with one row per frame containing:
- Gaze: `gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y` (8 values)
- Head pose: `pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz` (6 values)
- AU intensities: `AU01_r` through `AU45_r` (17 values)
- AU presence: `AU01_c` through `AU45_c` (18 values)

**Total OpenFace feature dim D = 49** (gaze 8 + head 6 + AU_r 17 + AU_c 18)

**Chunking** (following paper exactly):
```python
def chunk_openface(frames: np.ndarray, T: int = 10) -> np.ndarray:
    """
    frames: (N_frames, D=49)
    Returns: (3*D, T) = (147, 10) — min/max/var per chunk
    """
    # Repeat short clips until >= 250 frames
    while len(frames) < 250:
        frames = np.tile(frames, (2, 1))[:250*2]
    frames = frames[:250]  # standardize to 250 frames
    
    chunks = np.array_split(frames, T, axis=0)
    stats = []
    for chunk in chunks:
        stats.append(np.concatenate([
            chunk.min(axis=0),
            chunk.max(axis=0),
            chunk.var(axis=0)
        ]))  # (3D,)
    return np.stack(stats, axis=1)  # (3D, T)
```

**Save**: per clip as `.npy` of shape `(147, 10)`.

### 1.2 Feature Extraction — I3D

**Script**: `src/features/extract_i3d.py`

**Repository-specific note**:
- For CMOSE, `data/CMOSE/final_data_1.json` provides per-clip I3D embeds; materialize to `data/CMOSE/features/i3d/*.npy` if files are missing.
- For private data, run I3D extraction only for clips marked `is_accepted=1` in `data/private/accepted.csv`.
- In `final_data_1.json`, the `embeds` field is typically **1024-dim** per clip (float vector), but some clips have empty embeds and should be filtered out.

Use I3D pretrained on Kinetics-400 (RGB stream only, since no audio). Extract the **feature vector before the final classification layer** (typically 1024-dim from the average-pooled output).

```python
# Using mmaction2 or a standalone I3D:
# Input: clip resized to 224×224, 64 frames sampled uniformly
# Output: (1024,) feature vector per clip
# Save as .npy
```

**Note**: If the clip is shorter than 64 frames, loop it. If longer, sample uniformly.

### 1.3 Model Architecture

**File**: `src/models/cmose_baseline.py`

#### MLP1 — Attention Weight Generator
```
Input:  X_I3D  ∈ R^{1024}
FC(1024 → C) → ReLU → Dropout(0.5) → FC(C → T)
Output: logits ∈ R^T  (then Softmax to get X_attn ∈ R^{1×T})
```

#### TCN — Temporal Convolutional Network
```
Input:  ps ∈ R^{3D×T} = R^{147×10}
TCN(in=147, hidden=C, layers=4, kernel_size=3, dropout=0.2)
Output: X_TCN ∈ R^{C×T}
```

Use the TCN implementation from `https://github.com/locuslab/TCN` (directly copyable, MIT license).

#### Attention + HL feature
```python
X_attn = Softmax(MLP1(X_I3D))          # (1, T)
X_HL   = X_TCN @ X_attn.T              # (C, 1) → squeeze → (C,)
```

#### MLP2 — I3D projection
```
Input:  X_I3D ∈ R^{1024}
FC(1024 → C) → ReLU → FC(C → C)
Output: ∈ R^C
```

#### Concatenation
```python
X_vis = torch.cat([MLP2(X_I3D), X_HL], dim=-1)  # (2C,)
```

#### MLP3 — Normalized FC (score head)
```python
# Normalize both weight and input — no bias
s = F.linear(F.normalize(X_vis), F.normalize(weight))  # scalar ∈ [-1, 1]
```

#### Threshold to class
```python
def score_to_class(s):
    if s < -0.5: return 0  # HD
    elif s < 0:  return 1  # DE
    elif s < 0.5: return 2  # EG
    else:        return 3  # HE
```

**Hyperparameter C = 128** (start here; ablate with 64 and 256).

### 1.4 MocoRank

**File**: `src/training/mocorank.py`

#### Momentum Encoder
- Same architecture as the main model, initialized with the same weights
- Updated each iteration: `θ_mom = 0.999 * θ_mom + 0.001 * θ_model`
- **NOT** updated by gradient — only by the momentum rule

#### Score Pool
- FIFO queue of length `|P| = 2048`
- Each entry: `(label l2, score s2, embedding e2)`
- Initialized with one forward pass over a shuffled balanced sample of all 4 classes
- Updated each iteration: pop oldest `|B|` entries, push new entries from momentum encoder

#### Multi-Margin Loss
```python
def multi_margin_loss(scores_B, labels_B, pool):
    """
    scores_B: (|B|,)  — current model scores
    labels_B: (|B|,)  — ground truth labels (0-3)
    pool: list of (l2, s2, e2) tuples
    """
    loss = 0.0
    count = 0
    for i, (s1, l1, e1) in enumerate(zip(scores_B, labels_B, embeddings_B)):
        for (l2, s2, e2) in pool:
            cos_sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
            sim_scaled = (cos_sim + 1) / 2  # map to [0, 1]
            
            diff = abs(int(l1) - int(l2))
            if diff == 0:
                # Same class: penalize large score difference
                f = F.l1_loss(scores_B[i], s2, reduction='none')
            elif diff == 1:
                M = 0.5 * sim_scaled
                f = M - (s1 - s2) if l1 > l2 else M - (s2 - s1)
            elif diff == 2:
                M = 0.5 + 0.5 * sim_scaled
                f = M - (s1 - s2) if l1 > l2 else M - (s2 - s1)
            else:  # diff == 3
                M = 1.0 + 0.5 * sim_scaled
                f = M - (s1 - s2) if l1 > l2 else M - (s2 - s1)
            
            loss += max(f, 0)
            count += 1
    return loss / count
```

**Optional Center Loss** (for MocoRank + Center Loss variant):
```python
center_loss = CenterLoss(num_classes=4, feat_dim=2*C)
total_loss = mocorank_loss + 0.2 * center_loss
```

### 1.5 Training Configuration

```yaml
# configs/baseline.yaml
optimizer: AdamW
weight_decay: 1e-3
batch_size: 256
score_pool_size: 2048
epochs: 1200
lr_init: 5e-4
lr_final: 5e-7
scheduler: CosineAnnealing
momentum_update: 0.999
C: 128  # hidden dim
T: 10   # temporal chunks
```

**Training split**: 70% train / 20% val / 10% test (use CMOSE's released splits).

### 1.6 Evaluation Metrics

```python
def evaluate(preds, labels):
    acc = (preds == labels).mean()  # overall accuracy
    avg_acc = np.mean([
        (preds[labels == c] == c).mean()
        for c in range(4)
    ])  # per-class accuracy averaged — primary metric for imbalance
    
    # Also report:
    # - Confusion matrix (normalized by true class)
    # - Per-class precision, recall, F1
    # - MAE on ordinal labels (treat HD=0, DE=1, EG=2, HE=3)
    return acc, avg_acc
```

**Target to match**: Acc ≥ 75%, AvgAcc ≥ 53% (MocoRank without Center Loss, as Center Loss adds complexity).

---

## Phase 2 — Dataset Comparison (Feature-Space Analysis)

*This phase is independent of model performance — it compares the two datasets at the raw feature level.*

### 2.1 OpenFace Distribution Comparison

For each feature dimension (D=49), compute mean and std across all frames in CMOSE vs. private dataset. Report:
- **Feature-wise KL divergence** (using KDE or histogram approximation)
- **Wasserstein distance** per feature group (gaze, head pose, AU intensities, AU presence)
- **Box plots** side by side for the most discriminative AUs (AU01, AU04, AU06, AU12, AU45 — typically linked to engagement)

### 2.2 I3D Distribution Comparison

- Use I3D features for all CMOSE clips and all **accepted** private clips (`is_accepted=1`)
- **UMAP projection** (2D) of all clips, colored by dataset — visualize the gap
- **PCA variance explained** per component — do both datasets occupy similar subspaces?
- **Centroid distance** between CMOSE class centroids and private dataset centroid

### 2.3 Temporal Pattern Comparison

- Distribution of **clip lengths** (private vs CMOSE)
- **AU activation rate** over time — do engagement-related AUs activate at similar rates?
- **Gaze direction variance** — online meetings may show more variable gaze (phone checking, multiple monitors)

### 2.4 Key Report: Domain Gap Score

Report a simple composite score summarizing the gap:
```
Domain Gap = mean(Wasserstein_OpenFace_per_group) + cosine_distance(I3D_centroids)
```
This gives a single interpretable number to reference throughout the thesis discussion.

---

## Phase 3 — Domain Shift Analysis (Model on Private Dataset)

*Apply the CMOSE-trained model to accepted private clips (`data/private/accepted.csv`, `is_accepted=1`) without any adaptation.*

### 3.1 Prediction Distribution

- Run inference on all accepted private clips (`is_accepted=1`)
- Report: predicted class distribution (what % are labeled HD/DE/EG/HE)
- Compare to: CMOSE test distribution and expected real-world meeting distribution
- Flag: if >60% of clips are predicted as a single class, the model has collapsed on the target domain

### 3.2 Score Distribution Analysis

- Plot histogram of raw scalar scores s ∈ [−1, 1] for private clips vs CMOSE test clips
- Check if scores are concentrated near thresholds (uncertain) or near extremes (overconfident)
- **Entropy of prediction**: `H = -Σ p_c * log(p_c)` — high entropy = uncertain model

### 3.3 Per-Feature Attention Analysis

- Extract `X_attn` weights for private clips — are temporal attention weights uniform (no discrimination) or structured?
- Compare attention weight patterns between CMOSE test clips and private clips
- A flat/uniform attention pattern in target domain suggests I3D features are not guiding the model meaningfully

### 3.4 Embedding Space Visualization

- Extract `X_vis` (2C-dim embeddings) for:
  - CMOSE test clips (colored by true label)
  - Private clips (colored by predicted label)
- UMAP projection — do private clips cluster within CMOSE class clusters, or form separate regions?
- This is the **core domain shift visualization** for the thesis

### 3.5 Class Instability Assessment

Classes are "unstable" under domain shift if:
1. They are predicted at rates very different from CMOSE base rates
2. Their embeddings in the private dataset fall far from CMOSE class centroids
3. High prediction entropy concentrated around their threshold boundaries

Expected finding: **HD and HE** (minority classes) will be most unstable, because MocoRank was trained to push these apart from EG — but in a new domain, the boundary features may not transfer.

---

## Phase 4 — Pseudo-Labeling Strategies

*Three strategies, applied to accepted private clips (`is_accepted=1`). All use the CMOSE-trained model as starting point.*

### Strategy 1 — Confidence Threshold Self-Training

**Concept**: Only assign pseudo-labels to clips where the model is highly confident (score far from thresholds). Retrain on CMOSE + these confident pseudo-labeled clips. Repeat.

**Implementation**:
```python
CONFIDENCE_MARGIN = 0.3  # score must be > 0.3 away from nearest threshold

def is_confident(score):
    thresholds = [-0.5, 0.0, 0.5]
    min_dist = min(abs(score - t) for t in thresholds)
    return min_dist > CONFIDENCE_MARGIN

# Iteration:
# 1. Run model on private clips → get scores
# 2. Select clips where is_confident(score) == True
# 3. Assign pseudo-label via score_to_class(score)
# 4. Fine-tune model on CMOSE_train + pseudo_labeled_private (250 epochs)
# 5. Re-evaluate on CMOSE test set (check for catastrophic forgetting)
# 6. Repeat up to 3 iterations
```

**Report**: how many clips get pseudo-labels per iteration, CMOSE test accuracy before/after, prediction distribution shift on private set.

### Strategy 2 — Teacher-Student with Exponential Moving Average (EMA)

**Concept**: The "teacher" is an EMA copy of the "student" model. The student is trained on CMOSE labels + teacher's soft predictions on private clips. The teacher never sees gradients — it is a smoothed version of the student. This prevents the model from overfitting to its own wrong predictions.

**Implementation**:
```python
# Teacher = EMA of student (same momentum as MocoRank's momentum encoder)
# For private clips: teacher produces soft scores, convert to soft labels
# Student loss = CMOSE_MocoRank_loss + λ * consistency_loss(student_score, teacher_score)
# λ = 0.1 (start small to avoid dominating the CMOSE signal)
```

**This is the most principled strategy** because it avoids confirmation bias (the main failure mode of Strategy 1).

**Report**: student vs teacher score agreement over training epochs, entropy reduction on private clips, CMOSE accuracy retention.

### Strategy 3 — k-NN Label Propagation in Embedding Space

**Concept**: No retraining needed. Use the embedding space (X_vis, 2C-dim) from the CMOSE model. For each private clip, find its k nearest neighbors among CMOSE training clips (by cosine distance). Assign the majority label of those neighbors as the pseudo-label.

**Implementation**:
```python
from sklearn.neighbors import KNeighborsClassifier

# Extract X_vis embeddings for all CMOSE train clips + private clips
# Fit kNN on CMOSE embeddings with true labels
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(cmose_embeddings, cmose_labels)

# Predict on private clips
pseudo_labels = knn.predict(private_embeddings)
pseudo_probs  = knn.predict_proba(private_embeddings)
pseudo_confidence = pseudo_probs.max(axis=1)
```

**Advantage**: No retraining, fully interpretable. **Disadvantage**: Relies on embedding space being well-structured for the target domain (which Phase 3 will test).

**Report**: kNN confidence distribution, pseudo-label distribution, comparison to Strategy 1 labels (agreement rate between strategies = proxy for stability).

### Pseudo-Labeling Evaluation Protocol

Since there are no ground-truth labels for private clips, evaluate indirectly via:
0. **Population definition**: all metrics are computed on the accepted subset only (`data/private/accepted.csv`, `is_accepted=1`)
1. **Label stability**: run each strategy 3 times with different random seeds — what % of clips get the same label every time?
2. **Strategy agreement**: what % of clips get the same label from all 3 strategies? These are the most reliable pseudo-labels.
3. **CMOSE accuracy retention**: after any retraining, does CMOSE test accuracy drop? (Catastrophic forgetting indicator)
4. **Score entropy before/after**: does pseudo-labeling reduce model uncertainty on private clips?

---

## Phase 5 — Comparison: Naive Models vs. CMOSE Baseline

### On CMOSE Test Set
Report a clean comparison table:

| Model | Features | Acc (%) | AvgAcc (%) | HD Recall | DE Recall | EG Recall | HE Recall |
|---|---|---|---|---|---|---|---|
| MLP | OpenFace | | | | | | |
| LSTM | OpenFace | | | | | | |
| TCN | OpenFace | | | | | | |
| Transformer | OpenFace | | | | | | |
| MLP | I3D | | | | | | |
| MLP | TCN(OF)+I3D | | | | | | |
| **CMOSE Baseline** | **OF+I3D+MocoRank** | | | | | | |

### On Private Dataset (Pseudo-Labels as Proxy)
Run the same comparison using pseudo-labels from the most stable strategy (Strategy 3 as baseline, override with Strategy 1/2 if they show higher agreement). Report prediction distribution consistency, not accuracy.

### Discussion Points
- "Which features transfer better?" → Compare OpenFace-only vs I3D-only model domain shift severity (RQ4)
- "Which classes are unstable?" → Per-class embedding centroid drift (RQ3)
- "Does pseudo-labeling help?" → Score entropy and label stability metrics (RQ5)
- "Limitations of no target labels" → Be explicit: without a validation set on the target domain, there is no way to know if pseudo-labeling is improving real performance or drifting

---

## Phase 6 — (Optional) Proposed New Pipeline

*To be specified after Phase 3–4 analysis. The direction will be chosen from the Thesis Direction doc based on which failure mode is most clearly identified.*

**Placeholder architecture choice**: Prototype-Based Feature Normalization (Option C from thesis direction) — implementable in one week with no additional training:

```python
# Compute CMOSE class prototypes in embedding space
prototypes = {c: cmose_embeddings[cmose_labels == c].mean(axis=0) for c in range(4)}

# For each private clip embedding e:
# 1. Find closest prototype
nearest_class = min(prototypes, key=lambda c: cosine_distance(e, prototypes[c]))
# 2. Shift e toward that prototype
e_adapted = e + alpha * (prototypes[nearest_class] - e)  # alpha in [0.1, 0.5]
# 3. Re-score with MLP3
s_adapted = MLP3(normalize(e_adapted))
```

**Ablation**: vary alpha ∈ {0.1, 0.2, 0.3, 0.5}, report label stability and entropy.

---

## Deliverables Per Phase

| Phase | Key Output |
|---|---|
| 0 | Feature files for CMOSE + private, verified shapes |
| 1 | Trained model checkpoint, CMOSE test metrics matching paper |
| 2 | Feature distribution plots, domain gap score |
| 3 | Embedding UMAP plot, class instability report |
| 4 | Pseudo-label files, strategy comparison table |
| 5 | Full model comparison table, discussion draft |
| 6 | Adapted model checkpoint, delta metrics |


