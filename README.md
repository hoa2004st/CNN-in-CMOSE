# CNN-in-CMOSE

Train the narrowed CMOSE engagement-classification comparison using OpenFace and I3D features.

The repo now focuses on six models only:

- `cmose_baseline_paper`
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
python main.py --model cmose_baseline_paper
```

Run the comparison suite:

```bash
python scripts/compare_naive_models.py --run_root outputs/comparison_public_cmose
```

By default, the batch script writes one folder per kept model directly under `outputs/`, plus `outputs/logs/`.

Key options:

```text
--model                  cmose_baseline_paper | openface_mlp | temporal_cnn | lstm | transformer | i3d_mlp | openface_tcn_i3d_fusion
--target_frames          Frames per OpenFace sample after resampling         (default: 300)
--fusion_frames          Frames per I3D/fusion sample after resampling       (default: 75)
--baseline_chunk_count   Number of temporal chunks for paper baseline        (default: 10)
--score_pool_size        MocoRank score queue length for baseline            (default: 2048)
--momentum_update        Momentum encoder update coefficient                  (default: 0.999)
--epochs                 Maximum training epochs                             (default: 800)
--batch_size             Mini-batch size                                     (default: 128)
--lr                     Learning rate                                       (default: 1e-4)
--output_dir             Where to save artefacts                             (default: outputs/<model>)
--seed                   Random seed                                         (default: 42)
```

`cmose_baseline_paper` uses paper-aligned OpenFace chunk features (49 selected dimensions -> min/max/var -> `(147,10)`) plus I3D vectors and MocoRank training. `openface_mlp`, `temporal_cnn`, `lstm`, and `transformer` use normalized OpenFace tensors only. `i3d_mlp` uses normalized I3D tensors only. `openface_tcn_i3d_fusion` uses both modalities.

## Outputs

Files written under `--output_dir`:

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint |
| `metrics.json` | Final metrics and run config |
| `selection_summary.json` | Split usage and run assumptions |
| `preprocessing_summary.json` | Normalization and tensor-shape summary |
| `smote_summary.json` | Train/evaluation/test class counts; SMOTE is disabled |

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
