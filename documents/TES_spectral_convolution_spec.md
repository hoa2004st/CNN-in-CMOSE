# TES Spectral Convolution Implementation Checklist

This document turns the TES idea into a concrete implementation plan for the current codebase.

Current pipeline entry points:

- `main.py`: CLI, split handling, preprocessing dispatch, training, metrics export
- `src/paper_repro_data.py`: CMOSE metadata loading, OpenFace CSV loading, frame resampling
- `src/paper_repro_preprocess.py`: normalization and per-sample reduction utilities
- `src/paper_repro_model.py`: model definitions and `build_model(...)`
- `src/paper_repro_train.py`: training loop, loss selection, prediction, metrics
- `scripts/visualize_results.py`: post-run plots and summary reports
- `tests/test_paper_repro_pipeline.py`: smoke tests for data, models, preprocessing, and losses

The checklist below is written to match those files directly.

---

## 1. Phase 0: Lock the TES Scope

- [ ] Keep the first TES target narrow: `head pose + gaze + AU intensities`; do not start with all 709 features.
- [ ] Treat TES as a new raw-sequence branch, not an extension of `paper_cnn`.
- [ ] Define the first TES success target against the current best baseline:
  - beat or match `temporal_cnn` on macro-F1
  - improve class-0 and/or class-3 recall
  - preserve interpretable outputs
- [ ] Use the existing strict CMOSE train/test split exactly as in `main.py`; do not reintroduce paper-style subset selection or SMOTE.

Definition of done:

- [ ] A short TES feature subset list exists in code, not only in prose.
- [ ] The baseline to beat is explicitly `temporal_cnn`, with `rectangular_cnn` as the secondary comparison.

---

## 2. Phase 1: Expose Real OpenFace Feature Names

Problem in the current code:

- `src/paper_repro_data.py` loads feature values but does not expose the ordered OpenFace feature column names anywhere outside `load_openface_matrix(...)`.
- TES cannot reliably select head pose, gaze, and AU features until the feature-name order is accessible.

Implementation tasks:

- [ ] In `src/paper_repro_data.py`, add a helper that returns the ordered OpenFace feature column names from one CSV after removing `OPENFACE_META_COLS`.
- [ ] Reuse the same feature-order logic already used inside `load_openface_matrix(...)` so TES indices and loaded tensors stay aligned.
- [ ] Add a second helper that maps a list of feature names or prefixes to column indices.
- [ ] Decide whether TES feature selection should be:
  - exact-name based
  - prefix-based
  - or a fixed hand-written index list generated once from the real column names

Recommended implementation target:

- [ ] Add `get_openface_feature_columns(csv_path: str | Path) -> list[str]`
- [ ] Add `resolve_feature_indices(feature_columns: list[str], *, exact_names: list[str] | None = None, prefixes: list[str] | None = None) -> list[int]`

Tests:

- [ ] Add a test in `tests/test_paper_repro_pipeline.py` that creates a fake CSV and verifies the returned feature-column order excludes metadata columns and preserves all 709 feature columns.

Definition of done:

- [ ] TES feature groups can be selected from actual column names without guessing column numbers in the thesis text.

---

## 3. Phase 2: Add Spectral Preprocessing Utilities

Primary file:

- `src/paper_repro_preprocess.py`

Dependency gap:

- [ ] Add `scipy` to `requirements.txt` because STFT is not available in the current dependency list.

Implementation tasks:

- [ ] Add a small TES config container or constant block for:
  - `fs`
  - `nperseg`
  - `noverlap`
  - `nfft`
  - feature-group definitions
- [ ] Implement a per-sample STFT function that converts one `(frames, features)` sample into a spectral tensor.
- [ ] Keep output order explicit and stable:
  - input: `(frames, selected_features)`
  - output: `(selected_features, freq_bins, time_windows)`
- [ ] Decide and document the STFT boundary behavior explicitly. The current spec assumes SciPy default padding, which gives `33 x 20` for `300` frames at `30 fps` with `nperseg=32`, `noverlap=16`, `nfft=64`.
- [ ] Convert outputs to `float32`.
- [ ] Add a dataset-level wrapper that transforms all train/test samples using the same TES config.
- [ ] Add optional log scaling only if the raw magnitude range is unstable in practice; do not add it blindly.

Recommended function additions:

- [ ] `extract_spectral_sample(...)`
- [ ] `extract_spectral_dataset(...)`
- [ ] `build_tes_feature_groups(feature_columns: list[str]) -> dict[str, list[int]]`
- [ ] `flatten_feature_groups(feature_groups: dict[str, list[int]]) -> list[int]`

Suggested shape contract:

- [ ] Input to TES preprocessing: `(n_samples, 300, 709)`
- [ ] Output from TES preprocessing: `(n_samples, n_selected_features, 33, 20)` for the initial STFT design

Tests:

- [ ] Add a preprocessing test that feeds a dummy `(2, 300, 8)` tensor into TES preprocessing and checks:
  - dtype is `float32`
  - first dimension stays `2`
  - second dimension equals the selected feature count
  - frequency bins are `33`
  - time windows are `20` if using default SciPy boundary handling
- [ ] Add a test that verifies TES preprocessing is deterministic for a fixed input.

Definition of done:

- [ ] A raw train/test tensor can be converted into a spectral tensor with no manual notebook steps.

---

## 4. Phase 3: Add the Spectral Model

Primary file:

- `src/paper_repro_model.py`

Implementation tasks:

- [ ] Add a new model class, likely `SpectralConvNet`.
- [ ] Add a new `build_model(...)` branch for `model_name == "spectral_cnn"`.
- [ ] Introduce a dedicated `ModelSpec.input_kind` for TES, for example:
  - `spectral_tensor`
- [ ] Do not overload `frame_feature_map`; TES input has semantic channels equal to selected features, not a singleton image channel.
- [ ] Keep the first TES model modest:
  - 2 to 3 conv blocks
  - adaptive pooling
  - small MLP head
- [ ] Pass the selected feature count into the model constructor as `in_channels`.

Recommended model interface:

- [ ] `SpectralConvNet(n_input_features: int, num_classes: int = 4)`
- [ ] Expected forward input: `(batch, n_selected_features, 33, 20)`

Tests:

- [ ] Extend `test_model_factory_output_shapes()` in `tests/test_paper_repro_pipeline.py` with:
  - `build_model("spectral_cnn", input_features=<selected_feature_count>)`
  - dummy input shaped like `(2, selected_feature_count, 33, 20)`
  - output shape `(2, 4)`
  - `input_kind == "spectral_tensor"`

Definition of done:

- [ ] `build_model("spectral_cnn", ...)` works exactly like the existing model registry.

---

## 5. Phase 4: Wire TES into `main.py`

Primary file:

- `main.py`

Implementation tasks:

- [ ] Add `"spectral_cnn"` to the `--model` choices in `build_parser()`.
- [ ] Add TES-specific CLI options only if they are genuinely needed for repeatable experiments. Good candidates:
  - `--spectral_feature_set`
  - `--stft_nperseg`
  - `--stft_noverlap`
  - `--stft_nfft`
- [ ] Keep defaults aligned with the first thesis experiment to avoid excessive CLI complexity.
- [ ] After loading raw train/test matrices, branch preprocessing by `model_spec.input_kind`.
- [ ] Add a TES preprocessing path that:
  - fits train-set z-score normalization using the existing `fit_feature_normalizer(...)`
  - normalizes train and test with `normalize_dataset_per_feature(...)`
  - resolves TES feature indices from real feature names
  - transforms normalized tensors into spectral tensors
- [ ] Preserve the existing `paper_cnn` and raw-model branches without regression.
- [ ] Extend `preprocessing_summary.json` with TES-specific metadata:
  - selected feature count
  - selected feature names or groups
  - STFT parameters
  - resulting tensor shape
  - boundary policy

Important repo-specific note:

- [ ] The current pipeline already normalizes raw models using train-fit per-feature z-score. TES should reuse that path before STFT so comparisons remain fair.

Output-dir decision:

- [ ] Decide whether to use the current default output naming logic from `resolve_output_dir(...)`.
- [ ] If you want the folder name to match current baseline folders like `outputs/temporal_cnn`, run TES with an explicit `--output_dir outputs/spectral_cnn`.
- [ ] If you keep the default loss-specific naming scheme, the default TES folder will be `outputs/spectral_cnn_cross_entropy`.

Tests:

- [ ] Add a test for `resolve_output_dir(...)` covering `spectral_cnn`.
- [ ] Add a smoke test that verifies TES preprocessing metadata can be serialized into `preprocessing_summary.json`.

Definition of done:

- [ ] `python main.py --model spectral_cnn --output_dir outputs/spectral_cnn ...` runs end to end through preprocessing, training, prediction, and metrics export.

---

## 6. Phase 5: Reuse the Existing Training Stack

Primary file:

- `src/paper_repro_train.py`

Good news:

- [ ] No major training-loop rewrite should be needed.
- [ ] TES can reuse the current:
  - `train_model(...)`
  - `predict(...)`
  - `build_loss(...)`
  - `evaluate_predictions(...)`

Implementation tasks:

- [ ] Verify TES tensors work with the existing `TensorDataset` and `DataLoader` path.
- [ ] Start with `cross_entropy`.
- [ ] After the baseline TES run is stable, try `weighted_cross_entropy`.
- [ ] Do not start with focal loss for TES given the current collapse in `temporal_cnn_focal`.

Definition of done:

- [ ] TES trains through the existing trainer with no TES-specific hacks inside `paper_repro_train.py`.

---

## 7. Phase 6: Update Visualization and Reporting

Primary files:

- `scripts/visualize_results.py`
- optionally `documents/TES_spectral_convolution_spec.md`
- optionally `README.md`

Current status:

- `scripts/visualize_results.py` should already include `spectral_cnn` automatically once `metrics.json` exists and `config.model` is set correctly.

Checklist:

- [ ] Confirm that TES appears correctly in:
  - `summary_table_all_runs.csv`
  - `summary_table_best_per_model.csv`
  - `comparison_report.md`
- [ ] Add TES-specific interpretability artifacts if needed outside the generic visualizer, for example:
  - average class spectrograms
  - band-energy summaries
  - feature-group ablations
- [ ] Decide whether to extend `scripts/visualize_results.py` or create a TES-specific analysis script under `scripts/`.

Recommended split:

- [ ] Keep the current `visualize_results.py` unchanged for generic comparison plots.
- [ ] Add a separate TES analysis script if you need spectrogram- or band-specific figures.

Documentation cleanup:

- [ ] Update `README.md` once TES is implemented.
- [ ] Fix stale README statements that no longer match the codebase:
  - raw models are z-score normalized, not min-max normalized
  - current CLI defaults in README are outdated relative to `main.py`

Definition of done:

- [ ] TES appears in the normal comparison tables, and TES-specific interpretability figures can be generated in a repeatable way.

---

## 8. Phase 7: Add Regression Tests

Primary file:

- `tests/test_paper_repro_pipeline.py`

Minimum TES test additions:

- [ ] Feature-column extraction test
- [ ] Feature-index resolution test
- [ ] STFT preprocessing shape test
- [ ] `build_model("spectral_cnn", ...)` output-shape test
- [ ] `resolve_output_dir(...)` test for TES

Nice-to-have tests:

- [ ] Test that TES preprocessing raises a clear error if no requested feature names are found.
- [ ] Test that TES preprocessing handles constant-valued feature channels without NaNs.
- [ ] Test that TES preprocessing metadata includes STFT parameters.

Definition of done:

- [ ] `pytest` covers the new TES branch well enough to catch broken tensor shapes and missing CLI/model wiring.

---

## 9. Phase 8: First Experimental Run

Suggested first run:

```bash
python main.py \
  --model spectral_cnn \
  --output_dir outputs/spectral_cnn \
  --epochs 800 \
  --batch_size 128 \
  --lr 1e-4 \
  --patience 50 \
  --loss cross_entropy \
  --amp
```

Checklist:

- [ ] Run TES once with the same general training regime used by the current best baselines.
- [ ] Inspect:
  - `outputs/spectral_cnn/preprocessing_summary.json`
  - `outputs/spectral_cnn/metrics.json`
  - `outputs/spectral_cnn/selection_summary.json`
- [ ] Verify tensor shape, selected feature count, and STFT parameters were recorded correctly.
- [ ] Run:

```bash
python scripts/visualize_results.py
```

- [ ] Confirm TES appears in the comparison outputs under `outputs/visualizations/`.

Definition of done:

- [ ] TES has one clean baseline run with comparable logging and outputs to the existing models.

---

## 10. Phase 9: Thesis-Useful Experiments

After the first TES run is stable:

- [ ] Run TES with `weighted_cross_entropy`.
- [ ] Run at least one feature-group ablation:
  - head pose only
  - gaze only
  - AU only
  - head pose + gaze + AU
- [ ] Compare TES directly against:
  - `temporal_cnn`
  - `rectangular_cnn`
- [ ] Record per-class recall differences, especially for class 0 and class 3.

Interpretability tasks:

- [ ] Compute average class spectrograms for a few selected channels.
- [ ] Summarize energy over coarse bands such as:
  - `0 to 1 Hz`
  - `1 to 3 Hz`
  - `3 to 5 Hz`
- [ ] Avoid overstating physiology; treat these as empirical analysis bands.

Definition of done:

- [ ] TES results answer the thesis question with evidence on macro-F1, minority-class recall, and interpretability.

---

## 11. File-by-File Change Checklist

### `requirements.txt`

- [ ] Add `scipy`

### `src/paper_repro_data.py`

- [ ] Expose ordered feature-column names
- [ ] Add TES feature-index resolution helpers

### `src/paper_repro_preprocess.py`

- [ ] Add STFT-based TES preprocessing
- [ ] Add TES feature-group selection helpers
- [ ] Ensure deterministic `float32` outputs

### `src/paper_repro_model.py`

- [ ] Add `SpectralConvNet`
- [ ] Add `build_model("spectral_cnn", ...)`
- [ ] Add `input_kind="spectral_tensor"`

### `main.py`

- [ ] Add `spectral_cnn` CLI support
- [ ] Add TES preprocessing branch
- [ ] Save TES preprocessing metadata

### `tests/test_paper_repro_pipeline.py`

- [ ] Add TES feature-name, STFT, model-shape, and output-dir tests

### `scripts/visualize_results.py`

- [ ] Verify TES works unchanged
- [ ] Add TES-specific analysis only if generic plots are not enough

### `README.md`

- [ ] Add TES run example
- [ ] Fix stale normalization/default-option text

---

## 12. Recommended Order of Work

- [ ] Step 1: add `scipy`
- [ ] Step 2: expose feature names in `paper_repro_data.py`
- [ ] Step 3: implement TES preprocessing in `paper_repro_preprocess.py`
- [ ] Step 4: add `SpectralConvNet` and model registration
- [ ] Step 5: wire TES into `main.py`
- [ ] Step 6: add tests
- [ ] Step 7: run one TES baseline experiment
- [ ] Step 8: run visualization and inspect outputs
- [ ] Step 9: run weighted-loss and ablation experiments

This order minimizes integration risk because it solves the feature-selection and tensor-shape problems before touching training or thesis analysis.

---

## 13. Final Acceptance Criteria

TES is integrated into the current repo when all of the following are true:

- [ ] `python main.py --model spectral_cnn ...` runs successfully
- [ ] `metrics.json`, `preprocessing_summary.json`, and `best_model.pth` are produced
- [ ] TES appears in `scripts/visualize_results.py` outputs
- [ ] `pytest` passes with TES coverage added
- [ ] At least one TES run is directly comparable to `temporal_cnn` and `rectangular_cnn`
- [ ] The run produces enough information to discuss macro-F1, minority-class recall, and interpretable spectral patterns in the thesis
