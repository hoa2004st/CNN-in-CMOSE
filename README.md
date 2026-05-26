# CNN-in-CMOSE

Train the narrowed CMOSE engagement-classification comparison using OpenFace and I3D features.

The repo now focuses on these models:

- `openface_mlp`
- `temporal_cnn`
- `lstm`
- `transformer`
- `i3d_mlp`
- `openface_tcn_i3d_fusion`

The dataset uses three source split keys: `train`, `unlabel`, and `test`. The pipeline fits on `train`, uses `unlabel` for checkpointing and early stopping, and uses `test` only for final reporting.

## Dataset layout

```text
data/CMOSE/
    final_data_1.json
    labels.csv
    features/
        openface/
            <sample_id>.csv
        i3d/
            <sample_id>.npy
```

`sample_id` is a CMOSE person-track key such as `video10_100_person0`.

## Usage

Run a single model:

```bash
python main.py --model temporal_cnn
python main.py --model openface_mlp
python main.py --model lstm
python main.py --model transformer
python main.py --model i3d_mlp
python main.py --model openface_tcn_i3d_fusion
```

Run the comparison suite:

```bash
python scripts/compare_naive_models.py
```

By default, the batch script writes model/loss runs under `outputs/training_log/`,
then writes cross-run CMOSE test-set charts under `outputs/model_assessment/cmose_testset/`.

Key options:

```text
--model                  openface_mlp | temporal_cnn | lstm | transformer | i3d_mlp | openface_tcn_i3d_fusion
--target_frames          Frames per OpenFace sample after resampling         (default: 300)
--fusion_frames          Frames per I3D/fusion sample after resampling       (default: 75)
--epochs                 Maximum training epochs                             (default: 800)
--batch_size             Mini-batch size                                     (default: 128)
--lr                     Learning rate                                       (default: 1e-4)
--output_dir             Where to save artefacts                             (default: outputs/training_log/<model>/<loss>)
--seed                   Random seed                                         (default: 42)
```

`openface_mlp`, `temporal_cnn`, `lstm`, and `transformer` use normalized OpenFace tensors only. `i3d_mlp` uses normalized I3D tensors only. `openface_tcn_i3d_fusion` uses both modalities.

## Outputs

The generated artifact tree is organized by question:

```text
outputs/
    dataset_analysis/
        cmose/
        private/
        comparison/
            domain_difference/
    training_log/
        openface_mlp/
        tcn/
        lstm/
        transformer/
        i3d_mlp/
        openface_tcn_i3d_fusion/
            ce/
            weighted_ce/
            ordinal/
    model_assessment/
        cmose_testset/
        private/
        comparison/
```

Training files written under each `outputs/training_log/<model>/<loss>/` run:

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint |
| `metrics.json` | Final metrics and run config |
| `selection_summary.json` | Split usage and run assumptions |
| `preprocessing_summary.json` | Normalization and tensor-shape summary |
| `smote_summary.json` | Train/evaluation/test class counts; SMOTE is disabled |

Training-log curves can be regenerated from completed runs:

```bash
python scripts/visualize_models_outputs.py
```

That command writes `training_curves.png` and `report.md` into each run folder,
plus CMOSE test-set metric bars, heatmaps, confusion matrices, and summary CSVs
under `outputs/model_assessment/cmose_testset/`.

Dataset analysis charts can be regenerated with:

```bash
python scripts/visualize_dataset_analysis.py
python scripts/feature_space_dataset_comparison.py
python scripts/visualize_feature_space_domains.py
```

Raw per-model predictions for the CMOSE test split and accepted private clips can be generated with:

```bash
python -m src.feature_analysis.run_domain_shift_analysis
```

Key prediction files:

| File | Description |
|---|---|
| `outputs/model_assessment/cmose_testset/predictions.csv` | Long-format CMOSE test predictions, one row per run and clip |
| `outputs/model_assessment/cmose_testset/predictions_by_clip.csv` | CMOSE test predictions with every run side by side per clip |
| `outputs/model_assessment/private/predictions.csv` | Long-format accepted-private predictions, one row per run and clip |
| `outputs/model_assessment/private/predictions_by_clip.csv` | Accepted-private predictions with every run side by side per clip |

Private/manual-label and CMOSE-vs-private assessment charts can be regenerated after prediction CSVs exist:

```bash
python scripts/visualize_model_assessment.py
```

## DAiSEE dataset

The same OpenFace + I3D extraction can be run on the public
[DAiSEE](https://www.kaggle.com/datasets/olgaparfenova/daisee) engagement
dataset and used in place of CMOSE. `scripts/daisee_extract_vast.sh` drives the
whole thing on a vast.ai Ubuntu GPU box: it downloads DAiSEE from Kaggle,
extracts features in the CMOSE format, builds the engagement labels, commits the
small label files here, and uploads the large feature files to a private Kaggle
dataset.

```bash
export GITHUB_TOKEN=... GIT_USER_EMAIL=you@example.com GIT_USER_NAME="You"
export KAGGLE_USERNAME=... KAGGLE_KEY=...
bash scripts/daisee_extract_vast.sh                 # full run
STAGES="openface i3d" bash scripts/daisee_extract_vast.sh   # rerun a subset
```

Mappings used: engagement `0/1/2/3` -> `Highly Disengage/Disengage/Engage/Highly
Engage`; split `Train/Validation/Test` -> `train/unlabel/test`. Each clip id gets
a `_person0` suffix so the CMOSE loader works unchanged. Produced files:

```text
data/DaiSEE/
    final_data_1.json   # committed to git
    labels.csv          # committed to git
    features/
        openface/<clipid>_person0.csv   # 709 features, pushed to Kaggle
        i3d/<clipid>_person0.npy        # 1024-dim,    pushed to Kaggle
```

Train on DAiSEE by pointing the existing pipeline at the new paths:

```bash
python main.py --model openface_tcn_i3d_fusion \
  --labels_json     data/DaiSEE/final_data_1.json \
  --feature_dir     data/DaiSEE/features/openface \
  --i3d_feature_dir data/DaiSEE/features/i3d
```

## Pipeline summary

```text
CMOSE OpenFace CSVs + final_data_1.json
    ->
src/features/extract_openface.py + src/features/extract_i3d.py
    ->
OpenFace tensors (target_frames x 709)
and/or
I3D tensors (fusion_frames x i3d_dim)
    ->
train split + evaluation split from `unlabel` + final test split
    ->
src/training/train.py (normalization helpers)
    ->
selected model
    ->
src/training/train.py
    ->
metrics.json
```
