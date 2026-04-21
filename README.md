# CNN-in-CMOSE

Reproduce the paper-inspired PCA/SVD + CNN engagement-classification pipeline on the CMOSE dataset using the provided OpenFace feature CSVs.

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

Default run:

```bash
python main.py
```

Strict CMOSE split mode:

```bash
python main.py --strict_paper_protocol
```

Key options:

```text
--method                 pca | svd                              (default: svd)
--n_components           Reduced features per frame             (default: 300)
--target_frames          Frames per sample after resampling     (default: 300)
--strict_paper_protocol  Respect original CMOSE train/test split
--include_unlabel        Also include entries marked unlabel
--epochs                 Maximum training epochs                (default: 1600)
--batch_size             Mini-batch size                        (default: 8)
--lr                     Learning rate                          (default: 1e-4)
--output_dir             Where to save artefacts                (default: outputs/)
--seed                   Random seed                            (default: 42)
```

## Outputs

Files written under `--output_dir`:

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint |
| `metrics.json` | Final metrics and run config |
| `selection_summary.json` | Selection mode and assumptions |
| `preprocessing_summary.json` | Explained-variance summary |
| `smote_summary.json` | Class counts before/after SMOTE |

## Project structure

```text
CNN-in-CMOSE/
├── main.py
├── requirements.txt
├── src/
│   ├── paper_repro_data.py
│   ├── paper_repro_model.py
│   ├── paper_repro_preprocess.py
│   └── paper_repro_train.py
└── tests/
    └── test_paper_repro_pipeline.py
```

## Pipeline summary

```text
CMOSE secondFeature CSVs + final_data_1.json
    ↓
paper_repro_data.py
    ↓
fixed-size 300 x 709 per-sample matrix
    ↓
paper_repro_preprocess.py
    ↓
PCA or SVD → 300 x 300
    ↓
paper_repro_model.py
    ↓
paper_repro_train.py
    ↓
metrics.json
```
