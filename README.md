# CNN-in-CMOSE

Evaluate the **SVD/PCA + CNN** student-engagement detection pipeline from  
*"Convolutional Neural Network Model based Students' Engagement Detection in  
Imbalanced DAiSEE Dataset"* on the **CMOSE** dataset.

## Background

The original paper tackles the DAiSEE engagement dataset, which is highly
imbalanced. To work around this the authors created their own sub-sampled
dataset, making fair comparison with other methods difficult. The
[CMOSE](https://doi.org/10.1145/3581783.3612471) (**C**omprehensive
**M**ultimodal **O**pen **S**tudent **E**ngagement) dataset offers a better
class balance **and** ships with pre-extracted
[OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) features, removing
the need for per-machine face analysis.

This repository:

1. Loads the per-frame OpenFace features from CMOSE.
2. Aggregates them into clip-level feature vectors.
3. Reduces dimensionality with **PCA** (or Truncated SVD).
4. Reshapes each vector into a 2-D feature map and trains a **CNN**.
5. Evaluates on the held-out test split (accuracy, macro/weighted F1,
   confusion matrix).

## Dataset layout

Download CMOSE and arrange it as follows:

```
<cmose_root>/
    labels/
        train_labels.csv   # columns: clip_id, label
        val_labels.csv
        test_labels.csv
    openface/
        <clip_id>.csv      # per-frame OpenFace features (one file per clip)
```

`label` must be an integer engagement level: **0** (not engaged), **1**
(barely engaged), **2** (engaged), **3** (highly engaged).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.10.

## Usage

```bash
python main.py --cmose_root /path/to/CMOSE
```

Full option list:

```
--cmose_root    Root directory of the CMOSE dataset            (required)
--n_components  Number of PCA components                       (default: 64)
--use_svd       Use TruncatedSVD instead of PCA
--aggregation   Clip aggregation: mean | std | mean_std | max  (default: mean_std)
--epochs        Maximum training epochs                        (default: 50)
--batch_size    Mini-batch size                                (default: 32)
--lr            Learning rate                                  (default: 1e-3)
--weight_decay  L2 regularisation                              (default: 1e-4)
--patience      Early-stopping patience (epochs)              (default: 10)
--no_class_weights  Disable inverse-frequency class weighting
--output_dir    Where to save artefacts                        (default: outputs/)
--seed          Random seed                                    (default: 42)
```

Outputs saved under `--output_dir`:

| File | Description |
|---|---|
| `preprocessor.pkl` | Fitted scaler + PCA for re-use / inference |
| `best_model.pth` | Best model checkpoint (lowest validation loss) |
| `confusion_matrix.png` | Test-set confusion matrix heatmap |

## Project structure

```
CNN-in-CMOSE/
├── main.py              – CLI entry point (full pipeline)
├── requirements.txt
├── src/
│   ├── data_loader.py   – Load CMOSE OpenFace CSVs and labels
│   ├── preprocess.py    – StandardScaler + PCA/SVD + CNN reshape
│   ├── model.py         – EngagementCNN (2-block Conv + FC head)
│   ├── train.py         – Training loop with class weights & early stopping
│   └── evaluate.py      – Accuracy, F1, confusion matrix, plot
└── tests/               – Pytest test suite (42 tests)
```

## Running tests

```bash
pytest tests/ -v
```

## Pipeline summary

```
CMOSE OpenFace CSVs
      │  per-frame features (AU, pose, gaze)
      ▼
data_loader.py  →  clip-level feature vectors  (mean + std over frames)
      │
      ▼
preprocess.py   →  StandardScaler  →  PCA (64 components)
      │                                     reshape → (1, 8, 8)
      ▼
model.py        →  Conv→BN→ReLU→Pool ×2  →  Dropout→FC(128)→FC(4)
      │
      ▼
train.py        →  Adam + class-weighted CE + ReduceLROnPlateau + early stop
      │
      ▼
evaluate.py     →  accuracy / F1 / confusion matrix
```