"""Tidy loaders that turn raw experiment artifacts into analysis-ready DataFrames.

Everything the thesis figures/tables need is assembled here so the plotting and table
scripts stay declarative. Sources:

* ``outputs/model_assessment/naive/full_matrix.csv``  — base 5 models x 3 losses x 3x3 matrix.
* ``outputs/model_assessment/hybrid/hybrid_matrix.csv`` — 192-config semantic-group ablation.
* ``outputs/training_log/<dataset>/<model>/<loss>/metrics.json`` — loss curves, confusion, report.
* ``outputs/dataset_analysis/*.csv`` — class distributions.

The hybrid ``arch_key`` (e.g. ``T_TCN_LSTM_T_TCN``) is decoded into one column per OpenFace
semantic group, in ``OPENFACE_GROUP_ORDER`` (gaze, eye, face, head, au), with values
``T`` (Transformer), ``TCN``, or ``LSTM``. See ``src/training/run_hybrid_ablation.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.output_paths import (
    DATASET_ANALYSIS_DIR,
    HYBRID_ASSESSMENT_DIR,
    NAIVE_ASSESSMENT_DIR,
    TRAINING_LOG_DIR,
    model_key_from_output_name,
)
from src.visualization.style import (
    CLASS_LABELS,
    display_model_name,
    loss_display_name,
)

# Group order baked into every arch_key (gaze_eye_face_head_au). Kept local to avoid a hard
# import of the feature-extraction stack (torch) just for a constant.
OPENFACE_GROUP_ORDER = ["gaze", "eye", "face", "head", "au"]
GROUP_DISPLAY = {
    "gaze": "Gaze",
    "eye": "Eye landmarks",
    "face": "Face landmarks",
    "head": "Head pose",
    "au": "Action units",
}
# Architecture tokens that appear in arch_key, in display order. "T" = Transformer,
# "TCN" = Temporal Convolutional Network, "LSTM" = LSTM.
ARCH_TOKENS = ["TCN", "T", "LSTM"]
ARCH_TOKEN_DISPLAY = {"T": "Transformer", "TCN": "TCN", "LSTM": "LSTM"}

# The six metrics every run reports. QWK + macro_accuracy are the thesis-primary pair.
METRIC_COLUMNS = [
    "accuracy",
    "macro_accuracy",
    "mae",
    "macro_mae",
    "cohen_kappa",
    "quadratic_weighted_kappa",
]
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "macro_accuracy": "Macro-Accuracy",
    "mae": "MAE",
    "macro_mae": "Macro-MAE",
    "cohen_kappa": "Cohen κ",
    "quadratic_weighted_kappa": "QWK",
}
# Metrics where a smaller value is better (ordinal-distance errors): selection and
# colour scales must be inverted for these relative to the agreement/recall metrics.
LOWER_BETTER_METRICS = {"mae", "macro_mae"}
TRAIN_GROUPS = ["cmose", "daisee", "combined"]
TEST_SETS = ["cmose_test", "daisee_test", "private"]
TRAIN_GROUP_DISPLAY = {"cmose": "CMOSE", "daisee": "DAiSEE", "combined": "Combined"}
TEST_SET_DISPLAY = {"cmose_test": "CMOSE", "daisee_test": "DAiSEE", "private": "Private"}

# Cells of the 3x3 matrix whose TEST corpus contributed no training data — the genuine
# generalization regime. "combined" pools CMOSE+DAiSEE train splits, so for it only the
# private set is unseen; combined→cmose/daisee_test are seen-target cells.
UNSEEN_TARGET_CELLS = [
    ("cmose", "daisee_test"),
    ("cmose", "private"),
    ("daisee", "cmose_test"),
    ("daisee", "private"),
    ("combined", "private"),
]


def unseen_target_mask(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting the unseen-target (generalization) cells of ``frame``."""
    cells = set(UNSEEN_TARGET_CELLS)
    return pd.Series(
        [pair in cells for pair in zip(frame["train_group"], frame["test_set"])],
        index=frame.index,
    )


# --------------------------------------------------------------------------------------
# Assessment matrices
# --------------------------------------------------------------------------------------
def load_matrix(path: str | Path | None = None) -> pd.DataFrame:
    """Load the naive (base-model) 3x3 assessment matrix as a tidy DataFrame."""
    path = Path(path) if path is not None else NAIVE_ASSESSMENT_DIR / "full_matrix.csv"
    frame = pd.read_csv(path)
    # The matrix CSV stores folder slugs (e.g. "tcn"); canonicalize to internal model keys
    # (e.g. "temporal_cnn") so MODEL_ORDER filtering and display mapping line up.
    frame["model"] = frame["model"].map(model_key_from_output_name)
    frame["model_display"] = frame["model"].map(display_model_name)
    frame["loss_display"] = frame["loss"].map(loss_display_name)
    frame["train_display"] = frame["train_group"].map(TRAIN_GROUP_DISPLAY).fillna(frame["train_group"])
    frame["test_display"] = frame["test_set"].map(TEST_SET_DISPLAY).fillna(frame["test_set"])
    frame["is_in_domain"] = (
        frame["train_group"].map({"cmose": "cmose_test", "daisee": "daisee_test"})
        == frame["test_set"]
    )
    return frame


def load_hybrid_matrix(path: str | Path | None = None) -> pd.DataFrame:
    """Load the hybrid ablation matrix and decode arch_key into per-group columns."""
    path = Path(path) if path is not None else HYBRID_ASSESSMENT_DIR / "hybrid_matrix.csv"
    frame = pd.read_csv(path)

    tokens = frame["arch_key"].str.split("_", expand=True)
    if tokens.shape[1] != len(OPENFACE_GROUP_ORDER):
        raise ValueError(
            f"arch_key in {path} split into {tokens.shape[1]} tokens; "
            f"expected {len(OPENFACE_GROUP_ORDER)} ({OPENFACE_GROUP_ORDER})."
        )
    for position, group in enumerate(OPENFACE_GROUP_ORDER):
        frame[f"arch_{group}"] = tokens[position]

    for token in ARCH_TOKENS:
        frame[f"n_{token.lower()}"] = sum(
            (frame[f"arch_{g}"] == token).astype(int) for g in OPENFACE_GROUP_ORDER
        )
    frame["has_i3d"] = frame["model_type"] == "openface_temporal_i3d_hybrid"
    frame["variant"] = frame["has_i3d"].map({True: "Hybrid + I3D", False: "Hybrid (OpenFace only)"})
    frame["loss_display"] = frame["loss"].map(loss_display_name)
    frame["train_display"] = frame["train_group"].map(TRAIN_GROUP_DISPLAY).fillna(frame["train_group"])
    frame["test_display"] = frame["test_set"].map(TEST_SET_DISPLAY).fillna(frame["test_set"])
    return frame


def best_per_cell(frame: pd.DataFrame, metric: str = "quadratic_weighted_kappa") -> pd.DataFrame:
    """For each (train_group, test_set) cell return the row with the best ``metric``.

    "Best" maximizes the metric, except for the lower-is-better ordinal-distance metrics
    (``mae``, ``macro_mae``), which are minimized.
    """
    grouped = frame.groupby(["train_group", "test_set"])[metric]
    idx = grouped.idxmin() if metric in LOWER_BETTER_METRICS else grouped.idxmax()
    return frame.loc[idx].reset_index(drop=True)


def cell_matrix(frame: pd.DataFrame, metric: str = "quadratic_weighted_kappa") -> pd.DataFrame:
    """Pivot best-per-cell ``metric`` into a 3x3 train (rows) x test (cols) table."""
    best = best_per_cell(frame, metric)
    pivot = best.pivot(index="train_group", columns="test_set", values=metric)
    return pivot.reindex(index=TRAIN_GROUPS, columns=TEST_SETS)


# --------------------------------------------------------------------------------------
# Training histories (loss curves + confusion + per-class report)
# --------------------------------------------------------------------------------------
def load_training_histories(root: str | Path | None = None) -> pd.DataFrame:
    """Walk ``training_log/<dataset>/<model_folder>/<loss>/metrics.json`` into a DataFrame.

    One row per run. Heavy objects (loss curves, confusion matrix, per-class report) are
    stored as Python objects inside the cells so downstream plots can index by run.
    """
    root = Path(root) if root is not None else TRAINING_LOG_DIR
    rows: list[dict] = []
    for metrics_path in sorted(root.glob("*/*/*/metrics.json")):
        dataset = metrics_path.parts[-4]
        model_folder = metrics_path.parts[-3]
        loss_slug = metrics_path.parts[-2]
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = data.get("history", {})
        metrics = data.get("metrics", {})
        config = data.get("config", {})
        row = {
            "dataset": dataset,
            "model_folder": model_folder,
            "model": model_key_from_output_name(model_folder),
            "loss": loss_slug,
            "loss_display": loss_display_name(loss_slug),
            "train_losses": history.get("train_losses", []),
            "eval_losses": history.get("eval_losses", []),
            "best_epoch": history.get("best_epoch"),
            "stopped_early": history.get("stopped_early"),
            "n_epochs": len(history.get("train_losses", [])),
            "group_architectures": config.get("group_architectures"),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "confusion_matrix_normalized": metrics.get("confusion_matrix_normalized"),
            "classification_report_dict": metrics.get("classification_report_dict"),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = metrics.get(metric)
        row["model_display"] = display_model_name(row["model"])
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_f1(report_dict: dict | None) -> dict[str, float]:
    """Pull per-class F1 out of an sklearn classification_report dict, keyed by class label."""
    if not report_dict:
        return {}
    out: dict[str, float] = {}
    for label in CLASS_LABELS:
        entry = report_dict.get(label)
        if isinstance(entry, dict) and "f1-score" in entry:
            out[label] = float(entry["f1-score"])
    return out


# --------------------------------------------------------------------------------------
# Dataset class distributions
# --------------------------------------------------------------------------------------
def load_class_distribution_overall(root: str | Path | None = None) -> pd.DataFrame:
    root = Path(root) if root is not None else DATASET_ANALYSIS_DIR
    return pd.read_csv(root / "class_distribution_overall.csv")


def load_class_distribution_by_split(root: str | Path | None = None) -> pd.DataFrame:
    root = Path(root) if root is not None else DATASET_ANALYSIS_DIR
    return pd.read_csv(root / "class_distribution_by_split.csv")


def load_private_class_distribution(root: str | Path | None = None) -> pd.DataFrame | None:
    root = Path(root) if root is not None else DATASET_ANALYSIS_DIR
    path = root / "private" / "class_distribution.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------------------
# Per-clip predictions (for confusion matrices on any test set, incl. private)
# --------------------------------------------------------------------------------------
def load_naive_predictions(path: str | Path | None = None) -> pd.DataFrame:
    """Load the base-model per-clip prediction log (all train groups x test sets)."""
    path = Path(path) if path is not None else NAIVE_ASSESSMENT_DIR / "full_matrix_predictions.csv"
    frame = pd.read_csv(path, usecols=["train_group", "test_set", "model", "loss",
                                       "clip_id", "true_id", "predicted_id"])
    frame["model"] = frame["model"].map(model_key_from_output_name)
    return frame


def _is_lfs_pointer(path: Path) -> bool:
    """True if the file is an unfetched git-LFS pointer rather than real content."""
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as handle:
            return handle.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return True


def load_hybrid_predictions(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load the hybrid per-clip prediction log, or None if the LFS file isn't fetched.

    Large (~135 MB) — only the columns needed for confusion matrices are read. Run
    ``git lfs pull --include outputs/model_assessment/hybrid/hybrid_matrix_predictions.csv``
    to materialize it.
    """
    path = Path(path) if path is not None else HYBRID_ASSESSMENT_DIR / "hybrid_matrix_predictions.csv"
    if not path.exists() or _is_lfs_pointer(path):
        return None
    return pd.read_csv(path, usecols=["train_group", "test_set", "model_type", "arch_key",
                                      "loss", "clip_id", "true_id", "predicted_id"])


def confusion_from_predictions(frame: pd.DataFrame, *, normalize: bool = True) -> "pd.DataFrame":
    """Row-normalized 4x4 confusion (index=true, cols=pred) from a predictions slice."""
    cm = pd.crosstab(frame["true_id"], frame["predicted_id"]).reindex(
        index=range(len(CLASS_LABELS)), columns=range(len(CLASS_LABELS)), fill_value=0
    ).astype(float)
    if normalize:
        row_sums = cm.sum(axis=1).replace(0, 1)
        cm = cm.div(row_sums, axis=0)
    cm.index = CLASS_LABELS
    cm.columns = CLASS_LABELS
    return cm
