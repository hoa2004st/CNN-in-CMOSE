# CNN-in-CMOSE

Reproduce the paper-inspired PCA/SVD + CNN engagement-classification pipeline on the CMOSE dataset using the provided OpenFace feature CSVs.

The repo now uses the strict CMOSE train/test protocol by default. The older paper-style subset-selection plus pre-split SMOTE path has been removed to avoid leakage.

## Dataset layout

The current pipeline expects:

```text
data/CMOSE/
    final_data_1.json
    secondFeature/
        secondFeature/
            <sample_id>.csv
```

`sample_id` is a CMOSE person-track key such as `video10_100_person0`.

## Usage

Default baseline run:

```bash
python main.py
```

Run one of the raw-sequence comparison models:

```bash
python main.py --model temporal_cnn
python main.py --model rectangular_cnn
python main.py --model lstm
python main.py --model transformer
```

Run all four comparison models sequentially with separate output folders and logs:

```bash
bash scripts/run_comparison_models.sh
```

Optional custom run root:

```bash
bash scripts/run_comparison_models.sh outputs/comparison_runs/my_server_run
```

Key options:

```text
--model                  paper_cnn | temporal_cnn | rectangular_cnn | lstm | transformer
--method                 pca | svd                              (default: svd)
--n_components           Reduced features per frame             (default: 300)
--target_frames          Frames per sample after resampling     (default: 300)
--epochs                 Maximum training epochs                (default: 1600)
--batch_size             Mini-batch size                        (default: 8)
--lr                     Learning rate                          (default: 1e-4)
--output_dir             Where to save artefacts                (default: outputs/)
--seed                   Random seed                            (default: 42)
```

`--method` and `--n_components` are used only by `paper_cnn`. The raw-sequence models operate directly on the normalized `frames x features` matrices.

The batch size, learning rate, and epochs inside `scripts/run_comparison_models.sh` can be adjusted in the `COMMON_ARGS` array.

## Outputs

Files written under `--output_dir`:

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint |
| `metrics.json` | Final metrics and run config |
| `selection_summary.json` | Selection mode and assumptions |
| `preprocessing_summary.json` | Reduction or normalization summary |
| `smote_summary.json` | Train/test class counts; SMOTE is disabled in strict mode |

## Project structure

```text
CNN-in-CMOSE/
|-- main.py
|-- requirements.txt
|-- src/
|   |-- paper_repro_data.py
|   |-- paper_repro_model.py
|   |-- paper_repro_preprocess.py
|   `-- paper_repro_train.py
`-- tests/
    `-- test_paper_repro_pipeline.py
```

## Pipeline summary

```text
CMOSE secondFeature CSVs + final_data_1.json
    ->
paper_repro_data.py
    ->
fixed-size 300 x 709 per-sample matrix
    ->
strict train/test split
    ->
paper_repro_preprocess.py
    ->
paper_cnn: PCA/SVD -> 300 x 300 -> square CNN
or
raw models: min-max normalized 300 x 709 -> temporal/rectangular/LSTM/Transformer
    ->
paper_repro_train.py
    ->
metrics.json
```
