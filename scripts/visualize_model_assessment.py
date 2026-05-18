"""Generate model-assessment charts from prediction CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.output_paths import (
    CMOSE_TESTSET_ASSESSMENT_DIR,
    MODEL_ASSESSMENT_DIR,
    MODEL_COMPARISON_ASSESSMENT_DIR,
    PRIVATE_ASSESSMENT_DIR,
    loss_slug,
    loss_name_from_slug,
    model_key_from_output_name,
    model_output_name,
)
from src.visualization.style import (
    CLASS_LABELS,
    CLASS_LABEL_SHORT,
    LOSS_DISPLAY_NAMES,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    class_color_list,
    model_color,
)

METRIC_SPECS = [
    ("accuracy", "Accuracy"),
    ("macro_accuracy", "Macro Accuracy"),
    ("mae", "MAE"),
    ("f1_weighted", "Weighted F1"),
    ("f1_macro", "Macro F1"),
    ("mse", "MSE"),
]
LOSS_LABEL_ORDER = ["CE", "Weighted CE", "Ordinal"]
CLASS_IDS = list(range(len(CLASS_LABELS)))
STACKED_CLASS_LABELS_BOTTOM_TO_TOP = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _split_run_key(run_key: object) -> tuple[str, str, str, str]:
    model_part, _, loss_part = str(run_key).partition("/")
    model_key = model_key_from_output_name(model_part)
    loss_name = loss_name_from_slug(loss_part)
    return (
        model_key,
        loss_name,
        MODEL_DISPLAY_NAMES.get(model_key, model_part),
        LOSS_DISPLAY_NAMES.get(loss_name, loss_part),
    )


def _normalized_run_key(run_key: object) -> str:
    model_key, loss_name, _model_label, _loss_label = _split_run_key(run_key)
    return f"{model_output_name(model_key)}/{loss_slug(loss_name)}"


def _add_run_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "run" not in frame.columns:
        return frame.copy()
    out = frame.copy()
    parsed = out["run"].map(_split_run_key)
    out["base_model"] = parsed.map(lambda values: values[0])
    out["loss"] = parsed.map(lambda values: values[1])
    out["base_model_display"] = parsed.map(lambda values: values[2])
    out["loss_label"] = parsed.map(lambda values: values[3])
    out["base_model"] = pd.Categorical(out["base_model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values(["base_model", "loss_label", "run"]).reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _model_labels(frame: pd.DataFrame) -> list[str]:
    present = set(frame["base_model"].astype(str)) if "base_model" in frame else set()
    return [MODEL_DISPLAY_NAMES.get(model, model) for model in MODEL_ORDER if model in present]


def _loss_labels(frame: pd.DataFrame) -> list[str]:
    if "loss_label" not in frame:
        return []
    present = set(frame["loss_label"].astype(str))
    ordered = [label for label in LOSS_LABEL_ORDER if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _metric_panel(frame: pd.DataFrame, out_path: Path, title: str) -> None:
    if frame.empty:
        return
    model_labels = _model_labels(frame)
    if not model_labels:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=160)
    axes = axes.flatten()
    for ax, (metric, metric_title) in zip(axes, METRIC_SPECS):
        if metric not in frame.columns:
            ax.axis("off")
            continue
        pivot = frame.pivot_table(
            index="base_model_display",
            columns="loss_label",
            values=metric,
            aggfunc="first",
        ).reindex(index=model_labels, columns=_loss_labels(frame))
        pivot.plot(kind="bar", ax=ax, width=0.78)
        ax.set_title(metric_title)
        ax.set_xlabel("")
        ax.set_ylabel(metric_title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.18)
        if metric in {"accuracy", "macro_accuracy", "f1_macro", "f1_weighted"}:
            ax.set_ylim(0.0, 1.0)
        ax.legend(title="Loss", fontsize=8)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _ce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "loss" not in frame.columns:
        return frame.copy()
    return frame[frame["loss"].isin(["cross_entropy", "ce"])].copy()


def _has_all_models(frame: pd.DataFrame) -> bool:
    if frame.empty or "base_model" not in frame.columns:
        return False
    return set(MODEL_ORDER).issubset(set(frame["base_model"].astype(str)))


def _one_run_per_model(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "base_model" not in frame.columns:
        return frame.copy()
    ordered = frame.copy()
    ordered["loss_priority"] = ordered["loss"].map(
        {
            "cross_entropy": 0,
            "ce": 0,
            "weighted_cross_entropy": 1,
            "weighted_ce": 1,
            "ordinal": 2,
        }
    ).fillna(99)
    ordered["base_model_sort"] = ordered["base_model"].astype(str).map(
        lambda model: MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    )
    ordered = ordered.sort_values(["base_model_sort", "loss_priority", "run"])
    return ordered.drop_duplicates(subset=["base_model"], keep="first").drop(columns=["loss_priority", "base_model_sort"])


def _heatmap_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    ce_metrics = _ce_frame(metrics)
    if _has_all_models(ce_metrics):
        return ce_metrics
    return _one_run_per_model(metrics)


def plot_metric_summaries(metrics: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    metrics = _add_run_columns(metrics)
    if metrics.empty:
        return
    _metric_panel(metrics, out_dir / "main_metrics_all_models_losses.png", f"{title_prefix}: 6 Models x 3 Losses")
    heatmap(_heatmap_frame(metrics), out_dir / "metric_heatmap_ce_models.png", f"{title_prefix}: CE Metric Heatmap")


def heatmap(frame: pd.DataFrame, out_path: Path, title: str) -> None:
    if frame.empty:
        return
    metric_cols = [metric for metric, _title in METRIC_SPECS if metric in frame.columns]
    if not metric_cols:
        return
    heat = frame.pivot_table(index="base_model_display", values=metric_cols, aggfunc="first").reindex(_model_labels(frame))
    if heat.empty:
        return
    heat = heat[metric_cols]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="equal", cmap="viridis")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(len(metric_cols)), [dict(METRIC_SPECS)[metric] for metric in metric_cols], rotation=35, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = heat.iat[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "true_id" not in predictions.columns:
        return pd.DataFrame()
    rows = []
    for run_key, group in predictions.groupby("run"):
        labeled = group[group["true_id"].notna()]
        if labeled.empty:
            continue
        y_true = labeled["true_id"].to_numpy(dtype=int)
        y_pred = labeled["predicted_id"].to_numpy(dtype=int)
        confusion = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
        per_class_accuracy = np.divide(
            confusion.diagonal(),
            confusion.sum(axis=1),
            out=np.zeros(len(CLASS_IDS), dtype=np.float64),
            where=confusion.sum(axis=1) > 0,
        )
        distances = np.abs(np.arange(len(CLASS_IDS))[:, None] - np.arange(len(CLASS_IDS))[None, :])
        total = max(float(confusion.sum()), 1.0)
        rows.append(
            {
                "run": run_key,
                "accuracy": float((y_true == y_pred).mean()),
                "macro_accuracy": float(per_class_accuracy.mean()),
                "f1_macro": float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="macro", zero_division=0)),
                "f1_weighted": float(f1_score(y_true, y_pred, labels=CLASS_IDS, average="weighted", zero_division=0)),
                "mae": float((confusion * distances).sum() / total),
                "mse": float((confusion * (distances**2)).sum() / total),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrices(predictions: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    if predictions.empty or "true_id" not in predictions.columns:
        return
    out_root = out_dir / "confusion_matrices"
    rows = []
    for run_key, group in predictions.groupby("run"):
        labeled = group[group["true_id"].notna()]
        if labeled.empty:
            continue
        y_true = labeled["true_id"].to_numpy(dtype=int)
        y_pred = labeled["predicted_id"].to_numpy(dtype=int)
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
        model_key, loss_name, model_label, loss_label = _split_run_key(run_key)
        rows.append((model_key, loss_name, model_label, loss_label, run_key, matrix))
        run_dir = out_root / _normalized_run_key(run_key)
        run_dir.mkdir(parents=True, exist_ok=True)
        _plot_confusion(matrix, f"{model_label} [{loss_label}]", run_dir / "confusion_matrix.png")

    if not rows:
        return
    rows.sort(key=lambda item: (MODEL_ORDER.index(item[0]) if item[0] in MODEL_ORDER else len(MODEL_ORDER), item[1]))
    cols = 3
    nrows = max(1, (len(rows) + cols - 1) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(13, 4.1 * nrows), dpi=150)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (_model_key, _loss_name, model_label, loss_label, _run_key, matrix) in zip(axes_list, rows):
        ax.imshow(matrix, cmap="Blues", aspect="equal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{model_label} [{loss_label}]", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(CLASS_LABELS)), list(CLASS_LABEL_SHORT.values()))
        ax.set_yticks(range(len(CLASS_LABELS)), list(CLASS_LABEL_SHORT.values()))
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", fontsize=8)
    for ax in axes_list[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{title_prefix}: Confusion Matrices", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices_all_runs.png", bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(matrix: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    image = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_LABELS)), list(CLASS_LABEL_SHORT.values()))
    ax.set_yticks(range(len(CLASS_LABELS)), list(CLASS_LABEL_SHORT.values()))
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_distribution(predictions: pd.DataFrame, out_dir: Path) -> None:
    if predictions.empty:
        return
    ce_predictions = predictions[predictions["run"].map(lambda value: _split_run_key(value)[1] in {"cross_entropy", "ce"})]
    if ce_predictions.empty:
        return
    counts = (
        ce_predictions.groupby(["run", "predicted_label"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby("run")["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals
    pivot = counts.pivot_table(index="run", columns="predicted_label", values="proportion", fill_value=0.0)
    ordered_runs = sorted(
        pivot.index,
        key=lambda run_key: MODEL_ORDER.index(_split_run_key(run_key)[0]) if _split_run_key(run_key)[0] in MODEL_ORDER else len(MODEL_ORDER),
    )
    pivot = pivot.reindex(ordered_runs)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    bottoms = np.zeros(len(pivot))
    for label in STACKED_CLASS_LABELS_BOTTOM_TO_TOP:
        values = pivot[label].to_numpy() if label in pivot else np.zeros(len(pivot))
        ax.bar(
            [MODEL_DISPLAY_NAMES.get(_split_run_key(run)[0], _split_run_key(run)[0]) for run in pivot.index],
            values,
            bottom=bottoms,
            label=label,
            color=class_color_list([label])[0],
        )
        bottoms += values
    ax.set_title("Prediction Distribution: CE Runs")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0.0, 1.0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_distribution_stack_barchart_ce_models.png", bbox_inches="tight")
    plt.close(fig)


def plot_confidence_agreement(predictions: pd.DataFrame, out_dir: Path) -> None:
    if predictions.empty:
        return
    ce_predictions = predictions[predictions["run"].map(lambda value: _split_run_key(value)[1] in {"cross_entropy", "ce"})]
    if ce_predictions.empty:
        return
    rows = []
    for clip_id, group in ce_predictions.groupby("clip_id"):
        labels = group["predicted_label"].astype(str).tolist()
        if not labels:
            continue
        majority_label = pd.Series(labels).mode().iloc[0]
        rows.append(
            {
                "clip_id": clip_id,
                "agreement_rate": labels.count(majority_label) / len(labels),
                "mean_confidence": float(group["confidence"].mean()),
                "majority_label": majority_label,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame["agreement_rate"] = frame["agreement_rate"].round(6)
    agreement_counts = frame["agreement_rate"].value_counts().sort_index()
    x_values = agreement_counts.index.to_numpy(dtype=float)
    min_spacing = float(np.diff(x_values).min()) if len(x_values) > 1 else 0.08
    bar_width = min(0.08, min_spacing * 0.75)

    fig, (scatter_ax, bar_ax) = plt.subplots(
        2,
        1,
        figsize=(7.5, 6.2),
        dpi=160,
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    for label in CLASS_LABELS:
        subset = frame[frame["majority_label"] == label]
        if subset.empty:
            continue
        scatter_ax.scatter(
            subset["agreement_rate"],
            subset["mean_confidence"],
            label=label,
            color=class_color_list([label])[0],
            alpha=0.72,
            s=30,
            edgecolors="none",
        )
    scatter_ax.set_title("Confidence vs Agreement: CE Runs")
    scatter_ax.set_ylabel("Mean confidence")
    scatter_ax.set_ylim(0.0, 1.02)
    scatter_ax.grid(alpha=0.18)
    scatter_ax.legend(fontsize=8)

    bar_ax.bar(x_values, agreement_counts.to_numpy(dtype=int), width=bar_width, color="#4D4D4D", alpha=0.82)
    bar_ax.set_xlabel("Agreement rate")
    bar_ax.set_ylabel("Clips")
    bar_ax.set_xlim(0.0, 1.0)
    bar_ax.set_xticks(x_values)
    bar_ax.set_xticklabels([f"{value:.2f}" for value in x_values])
    bar_ax.grid(axis="y", alpha=0.18)
    fig.savefig(out_dir / "confidence_agreement_scatter_ce_models.png", bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(out_dir / "confidence_agreement_ce_models.csv", index=False)


def plot_performance_drop(cmose_metrics: pd.DataFrame, private_metrics: pd.DataFrame, out_dir: Path) -> None:
    cmose_metrics = _add_run_columns(cmose_metrics)
    private_metrics = _add_run_columns(private_metrics)
    cmose_ce = _ce_frame(cmose_metrics)
    private_ce = _ce_frame(private_metrics)
    if cmose_ce.empty or private_ce.empty:
        return
    metric_cols = [metric for metric, _title in METRIC_SPECS if metric in cmose_ce.columns and metric in private_ce.columns]
    merged = private_ce[["run", "base_model", "base_model_display", *metric_cols]].merge(
        cmose_ce[["run", *metric_cols]],
        on="run",
        suffixes=("_private", "_cmose"),
    )
    if merged.empty:
        return
    plot_ce_dataset_metric_lines(merged, metric_cols, out_dir)
    rows = []
    for row in merged.to_dict(orient="records"):
        for metric in metric_cols:
            rows.append(
                {
                    "run": row["run"],
                    "model": row["base_model_display"],
                    "metric": metric,
                    "private_minus_cmose": row[f"{metric}_private"] - row[f"{metric}_cmose"],
                }
            )
    delta = pd.DataFrame(rows)
    delta.to_csv(out_dir / "performance_drop_ce_models.csv", index=False)
    plot_metrics = [metric for metric in ("accuracy", "macro_accuracy", "f1_macro") if metric in metric_cols]
    if not plot_metrics:
        return
    pivot = delta[delta["metric"].isin(plot_metrics)].pivot_table(index="model", columns="metric", values="private_minus_cmose")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="#4D4D4D", linewidth=1)
    ax.set_title("Private Minus CMOSE Performance: CE Runs")
    ax.set_ylabel("Metric delta")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_drop_chart_ce_models.png", bbox_inches="tight")
    plt.close(fig)


def plot_ce_dataset_metric_lines(merged: pd.DataFrame, metric_cols: list[str], out_dir: Path) -> None:
    metric_titles = dict(METRIC_SPECS)
    ordered = merged.copy()
    ordered["base_model_sort"] = ordered["base_model"].astype(str).map(
        lambda model: MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    )
    ordered = ordered.sort_values(["base_model_sort", "base_model_display"])

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 11.5), dpi=160, sharex=True)
    axes_list = axes.flatten()
    x_values = [0, 1]
    x_labels = ["CMOSE Test", "Private"]
    legend_handles = []
    legend_labels = []
    for ax, metric in zip(axes_list, metric_cols):
        for row in ordered.to_dict(orient="records"):
            model_key = str(row["base_model"])
            model_label = str(row["base_model_display"])
            (line,) = ax.plot(
                x_values,
                [row[f"{metric}_cmose"], row[f"{metric}_private"]],
                marker="o",
                markersize=4,
                linewidth=2.0,
                color=model_color(model_key),
                label=model_label,
            )
            if model_label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(model_label)
        metric_title = metric_titles.get(metric, metric)
        ax.set_title(metric_title)
        ax.set_xticks(x_values, x_labels)
        ax.set_ylabel(metric_title)
        ax.grid(axis="y", alpha=0.18)
        if metric in {"accuracy", "macro_accuracy", "f1_macro", "f1_weighted"}:
            ax.set_ylim(0.0, 1.0)
    for ax in axes_list[len(metric_cols):]:
        ax.axis("off")
    fig.suptitle("CE Models: CMOSE Test vs Private Metrics", fontsize=14, y=0.995)
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    fig.savefig(out_dir / "ce_models_cmose_private_metric_lines.png", bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    assessment_root = _resolve(args.assessment_root)
    cmose_dir = assessment_root / CMOSE_TESTSET_ASSESSMENT_DIR.name
    private_dir = assessment_root / PRIVATE_ASSESSMENT_DIR.name
    comparison_dir = assessment_root / MODEL_COMPARISON_ASSESSMENT_DIR.name
    for directory in (cmose_dir, private_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cmose_metrics = _read_csv(cmose_dir / "summary_table_all_runs.csv")
    if not cmose_metrics.empty and "run_name" in cmose_metrics.columns:
        cmose_metrics = cmose_metrics.rename(columns={"run_name": "run"})
    if cmose_metrics.empty:
        cmose_metrics = _read_csv(cmose_dir / "supervised_metrics.csv")
    private_metrics = _read_csv(private_dir / "supervised_metrics.csv")
    cmose_predictions = _read_csv(cmose_dir / "predictions.csv")
    private_predictions = _read_csv(private_dir / "predictions.csv")

    if cmose_metrics.empty and not cmose_predictions.empty:
        cmose_metrics = _prediction_metrics(cmose_predictions)
        cmose_metrics.to_csv(cmose_dir / "supervised_metrics.csv", index=False)
    if private_metrics.empty and not private_predictions.empty:
        private_metrics = _prediction_metrics(private_predictions)
        private_metrics.to_csv(private_dir / "supervised_metrics.csv", index=False)

    plot_metric_summaries(cmose_metrics, cmose_dir, "CMOSE Test")
    plot_metric_summaries(private_metrics, private_dir, "Private Manual Labels")
    plot_prediction_distribution(cmose_predictions, cmose_dir)
    plot_confidence_agreement(cmose_predictions, cmose_dir)
    plot_confusion_matrices(private_predictions, private_dir, "Private Manual Labels")
    plot_prediction_distribution(private_predictions, private_dir)
    plot_confidence_agreement(private_predictions, private_dir)
    plot_performance_drop(cmose_metrics, private_metrics, comparison_dir)

    print(f"Saved model-assessment charts under {assessment_root}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-root", default=str(MODEL_ASSESSMENT_DIR))
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
