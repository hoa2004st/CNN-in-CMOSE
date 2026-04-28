# CNN-in-CMOSE

Train the narrowed CMOSE engagement-classification comparison using OpenFace and I3D features.

The repo now focuses on six models only:

- `openface_mlp`
- `temporal_cnn`
- `lstm`
- `transformer`
- `i3d_mlp`
- `openface_tcn_i3d_fusion`

The dataset's source split key `test` is treated as the CMOSE unlabeled/evaluation split for checkpointing and early stopping, not as a separate held-out benchmark.

## Dataset layout

```text
data/CMOSE/
    final_data_1.json
    secondFeature/
        secondFeature/
            <sample_id>.csv
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
bash scripts/run_comparison_models.sh
```

By default, the batch script writes one folder per kept model directly under `outputs/`, plus `outputs/logs/`.

Key options:

```text
--model                  openface_mlp | temporal_cnn | lstm | transformer | i3d_mlp | openface_tcn_i3d_fusion
--target_frames          Frames per OpenFace sample after resampling         (default: 300)
--fusion_frames          Frames per I3D/fusion sample after resampling       (default: 75)
--epochs                 Maximum training epochs                             (default: 800)
--batch_size             Mini-batch size                                     (default: 128)
--lr                     Learning rate                                       (default: 1e-4)
--output_dir             Where to save artefacts                             (default: outputs/<model>)
--seed                   Random seed                                         (default: 42)
```

`openface_mlp`, `temporal_cnn`, `lstm`, and `transformer` use normalized OpenFace tensors only. `i3d_mlp` uses normalized I3D tensors only. `openface_tcn_i3d_fusion` uses both modalities.

## Outputs

Files written under `--output_dir`:

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint |
| `metrics.json` | Final metrics and run config |
| `selection_summary.json` | Split usage and run assumptions |
| `preprocessing_summary.json` | Normalization and tensor-shape summary |
| `smote_summary.json` | Train/unlabeled class counts; SMOTE is disabled |

## Pipeline summary

```text
CMOSE secondFeature CSVs + final_data_1.json
    ->
paper_repro_data.py
    ->
OpenFace tensors (target_frames x 709)
and/or
I3D tensors (fusion_frames x i3d_dim)
    ->
train split + CMOSE unlabeled/evaluation split
    ->
paper_repro_preprocess.py
    ->
selected model
    ->
paper_repro_train.py
    ->
metrics.json
```
