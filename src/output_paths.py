"""Canonical output locations for generated project artifacts."""

from __future__ import annotations

from pathlib import Path

OUTPUT_ROOT = Path("outputs")

DATASET_ANALYSIS_DIR = OUTPUT_ROOT / "dataset_analysis"
TRAINING_LOG_DIR = OUTPUT_ROOT / "training_log"
MODEL_ASSESSMENT_DIR = OUTPUT_ROOT / "model_assessment"

DATASET_CMOSE_DIR = DATASET_ANALYSIS_DIR / "cmose"
DATASET_PRIVATE_DIR = DATASET_ANALYSIS_DIR / "private"
DATASET_COMPARISON_DIR = DATASET_ANALYSIS_DIR / "comparison"
DOMAIN_DIFFERENCE_DIR = DATASET_COMPARISON_DIR / "domain_difference"

CMOSE_TESTSET_ASSESSMENT_DIR = MODEL_ASSESSMENT_DIR / "cmose_testset"
PRIVATE_ASSESSMENT_DIR = MODEL_ASSESSMENT_DIR / "private"
MODEL_COMPARISON_ASSESSMENT_DIR = MODEL_ASSESSMENT_DIR / "comparison"
RAW_PREDICTION_DIR = MODEL_COMPARISON_ASSESSMENT_DIR / "raw_predictions"

MANUAL_LABELS_DIR = DATASET_PRIVATE_DIR / "manual_labels"
MANUAL_LABELS_CSV = MANUAL_LABELS_DIR / "private_manual_labels.csv"

MODEL_OUTPUT_DIR_NAMES = {
    "openface_mlp": "openface_mlp",
    "temporal_cnn": "tcn",
    "lstm": "lstm",
    "transformer": "transformer",
    "i3d_mlp": "i3d_mlp",
    "openface_tcn_i3d_fusion": "openface_tcn_i3d_fusion",
}

MODEL_OUTPUT_ALIASES = {
    **{folder: model for model, folder in MODEL_OUTPUT_DIR_NAMES.items()},
    "temporal_cnn": "temporal_cnn",
    "tcn": "temporal_cnn",
    "id3_mlp": "i3d_mlp",
}

LOSS_NAME_TO_SLUG = {
    "cross_entropy": "ce",
    "weighted_cross_entropy": "weighted_ce",
    "ordinal": "ordinal",
    "focal": "focal",
    "ce": "ce",
    "weighted_ce": "weighted_ce",
}

LOSS_SLUG_TO_NAME = {
    "ce": "cross_entropy",
    "weighted_ce": "weighted_cross_entropy",
    "ordinal": "ordinal",
    "focal": "focal",
}


def model_output_name(model_name: object) -> str:
    """Return the stable folder name for a model key."""
    value = str(model_name)
    model_key = MODEL_OUTPUT_ALIASES.get(value, value)
    return MODEL_OUTPUT_DIR_NAMES.get(model_key, value)


def model_key_from_output_name(folder_name: object) -> str:
    """Return the internal model key from a folder/display name."""
    value = str(folder_name)
    return MODEL_OUTPUT_ALIASES.get(value, value)


def loss_slug(loss_name_or_slug: object) -> str:
    """Return the stable folder name for a loss key."""
    value = str(loss_name_or_slug)
    return LOSS_NAME_TO_SLUG.get(value, value)


def loss_name_from_slug(loss_name_or_slug: object) -> str:
    """Return the internal loss key from a folder/display name."""
    value = str(loss_name_or_slug)
    return LOSS_SLUG_TO_NAME.get(value, value)


def training_run_dir(
    *,
    model_name: object,
    loss_name: object,
    root: str | Path = TRAINING_LOG_DIR,
) -> Path:
    """Return outputs/training_log/<model>/<loss> for a model/loss run."""
    return Path(root) / model_output_name(model_name) / loss_slug(loss_name)


def training_run_key(model_name: object, loss_name: object) -> str:
    """Return the model/loss key used in prediction tables."""
    return f"{model_output_name(model_name)}/{loss_slug(loss_name)}"
