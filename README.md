# CNN-in-CMOSE

Engagement classification on the CMOSE dataset, comparing several models built on
OpenFace and I3D features. Models: `openface_mlp`, `temporal_cnn`, `lstm`,
`transformer`, `i3d_mlp`, and the `openface_tcn_i3d_fusion` hybrid.

Each sample is split into `train` / `unlabel` / `test`: the model fits on `train`,
early-stops on `unlabel`, and is reported on `test`.

## Layout

```text
src/        Python package (run modules with: python -m src.<module>)
    main.py            pipeline entry point
    data_prep/         build datasets and labels (CMOSE + DAiSEE)
    feature_extraction/OpenFace + I3D extraction
    models/            model architectures
    training/          training loop and full comparison sweep
    evaluation/        metrics and evaluation matrices
    analysis/          prediction tables and thesis artifacts
    visualization/     figures and dataset charts
scripts/    PowerShell (.ps1) helpers for setup, extraction, and training
```

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows; on Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs the CPU build of PyTorch. For GPU training, install the matching CUDA wheels:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

Train one model:

```bash
python -m src.main --model temporal_cnn
```

Train every model across all losses in one run:

```bash
python -m src.training.full_training_process
```

On Windows the same sweeps are wrapped as scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\full_training_process.ps1   # training only
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1                 # training + predictions
```

Common options: `--model`, `--epochs`, `--batch_size`, `--lr`, `--output_dir`, `--seed`.
See `python -m src.main --help` for the full list.

## Outputs

Each run writes to `outputs/training_log/<model>/<loss>/`:

| File | Description |
|---|---|
| `best_model.pth` | Best checkpoint |
| `metrics.json` | Final metrics and run config |
| `selection_summary.json` | Split usage and run assumptions |
| `preprocessing_summary.json` | Normalization and tensor-shape summary |

The per-clip CMOSE-test prediction table is generated with:

```bash
python -m src.analysis.prediction_generator
```

## DAiSEE dataset

The pipeline also runs on the public
[DAiSEE](https://www.kaggle.com/datasets/olgaparfenova/daisee) dataset. Prepare it with
the `scripts/*_daisee.ps1` extraction scripts and the `src.data_prep` modules
(`build_daisee_labels`, `daisee_convert_i3d`), then train by pointing at the new paths:

```bash
python -m src.main --model openface_tcn_i3d_fusion \
  --labels_json     data/DaiSEE/final_data_1.json \
  --feature_dir     data/DaiSEE/features/openface \
  --i3d_feature_dir data/DaiSEE/features/i3d
```
