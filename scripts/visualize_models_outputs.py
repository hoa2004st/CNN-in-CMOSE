"""Generate training-log curves and raw metric summaries from experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.metrics import label_distance_metrics_from_confusion
from src.output_paths import (
    CMOSE_TESTSET_ASSESSMENT_DIR,
    TRAINING_LOG_DIR,
    model_key_from_output_name,
)
from src.visualization.style import (
    CLASS_LABEL_SHORT,
    CURVE_COLORS,
    LOSS_DISPLAY_NAMES,
    LOSS_SLUG_TO_NAME,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
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
LOSS_LABELS = LOSS_DISPLAY_NAMES
LOSS_SLUGS = LOSS_SLUG_TO_NAME
BAR_WIDTH = 0.52


@dataclass
class RunResult:
    run_name: str
    run_dir: Path
    metrics_path: Path
    config: dict
    history: dict
    metrics: dict

    @property
    def base_model(self) -> str:
        return str(self.config.get("model", model_key_from_output_name(self.run_dir.parent.name)))

    @property
    def loss_name(self) -> str:
        loss = self.config.get("loss")
        if loss:
            return str(loss)
        folder_name = self.run_dir.name
        return LOSS_SLUGS.get(folder_name, folder_name)

    @property
    def loss_label(self) -> str:
        return LOSS_LABELS.get(self.loss_name, self.loss_name)

    @property
    def model_label(self) -> str:
        return MODEL_DISPLAY_NAMES.get(self.base_model, self.base_model)

    @property
    def comparison_label(self) -> str:
        return f"{self.model_label} [{self.loss_label}]"

    @property
    def variant_label(self) -> str:
        return self.loss_label


def apply_model_order(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "base_model" not in frame.columns:
        return frame
    ordered = frame.copy()
    ordered["base_model"] = pd.Categorical(
        ordered["base_model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    return ordered.sort_values(["base_model", "loss_label", "run_name"], na_position="last").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize completed experiment runs under outputs/training_log/.",
    )
    parser.add_argument("--outputs_dir", default=str(TRAINING_LOG_DIR))
    parser.add_argument(
        "--assessment_dir",
        default=str(CMOSE_TESTSET_ASSESSMENT_DIR),
        help="Directory for cross-run CMOSE test-set assessment charts.",
    )
    return parser.parse_args()


def load_completed_runs(outputs_dir: Path) -> list[RunResult]:
    runs: list[RunResult] = []
    for metrics_path in sorted(outputs_dir.rglob("metrics.json")):
        if any(part in {"model_assessment", "dataset_analysis", "visualizations"} for part in metrics_path.parts):
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        run_name = str(run_dir.relative_to(outputs_dir)).replace("\\", "/")
        runs.append(
            RunResult(
                run_name=run_name,
                run_dir=run_dir,
                metrics_path=metrics_path,
                config=payload.get("config", {}),
                history=payload.get("history", {}),
                metrics=payload.get("metrics", {}),
            )
        )
    return runs


def filter_comparison_runs(runs: list[RunResult]) -> list[RunResult]:
    filtered = []
    for run in runs:
        parts = run.run_name.split("/")
        if any(part.startswith("smoke_") or part == "_smoke" for part in parts):
            continue
        filtered.append(run)
    return filtered


def build_summary_frame(runs: list[RunResult]) -> pd.DataFrame:
    rows = []
    for run in runs:
        history = run.history
        metrics = run.metrics
        distance_metrics = {
            "mae": metrics.get("mae"),
            "mse": metrics.get("mse"),
        }
        if any(value is None for value in distance_metrics.values()) and metrics.get("confusion_matrix"):
            distance_metrics.update(label_distance_metrics_from_confusion(metrics["confusion_matrix"]))
        rows.append(
            {
                "run_name": run.run_name,
                "model_label": run.model_label,
                "comparison_label": run.comparison_label,
                "base_model": run.base_model,
                "variant_label": run.variant_label,
                "loss": run.loss_name,
                "loss_label": run.loss_label,
                "best_epoch": history.get("best_epoch"),
                "accuracy": metrics.get("accuracy"),
                "macro_accuracy": metrics.get("macro_accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "f1_weighted": metrics.get("f1_weighted"),
                "mae": distance_metrics["mae"],
                "mse": distance_metrics["mse"],
                "epochs_requested": run.config.get("epochs"),
                "batch_size": run.config.get("batch_size"),
                "lr": run.config.get("lr"),
                "device": run.config.get("device"),
                "amp": run.config.get("amp"),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame["base_model_display"] = frame["base_model"].map(
            lambda name: MODEL_DISPLAY_NAMES.get(str(name), str(name))
        )
        for metric, _title in METRIC_SPECS:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = apply_model_order(frame)
    return frame


def pick_best_run_per_model(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    ordered = summary_df.sort_values(
        ["base_model", "f1_macro", "macro_accuracy", "accuracy", "f1_weighted", "mae", "mse"],
        ascending=[True, False, False, False, False, True, True],
    )
    return apply_model_order(
        ordered.drop_duplicates(subset=["base_model"], keep="first").reset_index(drop=True)
    )


def save_summary_csvs(summary_df: pd.DataFrame, best_df: pd.DataFrame, viz_dir: Path) -> None:
    summary_df.to_csv(viz_dir / "summary_table_all_runs.csv", index=False)
    best_df.to_csv(viz_dir / "summary_table_best_per_model.csv", index=False)


def _model_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for model in MODEL_ORDER:
        if model in set(frame["base_model"].astype(str)):
            labels.append(MODEL_DISPLAY_NAMES.get(model, model))
    return labels


def _loss_labels(frame: pd.DataFrame) -> list[str]:
    if "loss_label" not in frame:
        return []
    present = set(frame["loss_label"].astype(str))
    ordered = [label for label in LOSS_LABEL_ORDER if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _present_losses(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    if frame.empty or "loss" not in frame.columns:
        return []
    by_slug = {}
    for row in frame.to_dict(orient="records"):
        slug = str(row["run_name"]).split("/")[-1]
        by_slug[slug] = (str(row["loss"]), str(row["loss_label"]))
    ordered = []
    for label in LOSS_LABEL_ORDER:
        for slug, (loss_name, loss_label) in by_slug.items():
            if loss_label == label:
                ordered.append((loss_name, slug, loss_label))
                break
    ordered_slugs = {slug for _loss_name, slug, _loss_label in ordered}
    ordered.extend((loss_name, slug, loss_label) for slug, (loss_name, loss_label) in sorted(by_slug.items()) if slug not in ordered_slugs)
    return ordered


def _filter_loss_runs(summary_df: pd.DataFrame, loss_name: str) -> pd.DataFrame:
    return summary_df[summary_df["loss"].astype(str).eq(str(loss_name))].copy()


def _plot_metric_panel(frame: pd.DataFrame, out_path: Path, title: str) -> None:
    if frame.empty:
        return
    model_labels = _model_labels(frame)
    if not model_labels:
        return
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=160)
    axes = axes.flatten()
    for ax, (metric, metric_title) in zip(axes, METRIC_SPECS):
        pivot = frame.pivot_table(
            index="base_model_display",
            columns="loss_label",
            values=metric,
            aggfunc="first",
        ).reindex(index=model_labels, columns=_loss_labels(frame))
        pivot.plot(kind="bar", ax=ax, width=BAR_WIDTH)
        ax.set_title(metric_title)
        ax.set_xlabel("")
        ax.set_ylabel(metric_title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.18)
        if metric in {"accuracy", "macro_accuracy", "f1_macro", "f1_weighted"}:
            ax.set_ylim(0.0, 1.0)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Loss", fontsize=8)
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(summary_df: pd.DataFrame, best_df: pd.DataFrame, assessment_dir: Path) -> None:
    _plot_metric_panel(
        summary_df,
        assessment_dir / "main_metrics_all_models_losses.png",
        "CMOSE Test Metrics: 6 Models x 3 Losses",
    )


def plot_metric_heatmap(summary_df: pd.DataFrame, assessment_dir: Path) -> None:
    for loss_name, slug, loss_label in _present_losses(summary_df):
        _plot_metric_heatmap_for_loss(_filter_loss_runs(summary_df, loss_name), assessment_dir, slug, loss_label)


def _plot_metric_heatmap_for_loss(summary_df: pd.DataFrame, assessment_dir: Path, slug: str, loss_label: str) -> None:
    if summary_df.empty:
        return
    heat = summary_df.pivot_table(
        index="base_model_display",
        values=[metric for metric, _title in METRIC_SPECS],
        aggfunc="first",
    ).reindex(_model_labels(summary_df))
    if heat.empty:
        return
    heat = heat[[metric for metric, _title in METRIC_SPECS]]
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="equal", cmap="viridis")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(len(heat.columns)), [title for _metric, title in METRIC_SPECS], rotation=35, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    for row_idx in range(heat.shape[0]):
        for col_idx in range(heat.shape[1]):
            value = heat.iat[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title(f"CMOSE Test Metric Heatmap: {loss_label} Runs")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(assessment_dir / f"metric_heatmap_{slug}_models.png", bbox_inches="tight")
    plt.close(fig)


def _plot_single_confusion(confusion: list[list[int]], title: str, out_path: Path) -> None:
    matrix = pd.DataFrame(confusion, index=CLASS_LABEL_SHORT.values(), columns=CLASS_LABEL_SHORT.values())
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="equal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(int(matrix.iat[row_idx, col_idx])), ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(runs: list[RunResult], assessment_dir: Path) -> None:
    matrices = [run for run in runs if run.metrics.get("confusion_matrix")]
    if not matrices:
        return
    individual_root = assessment_dir / "confusion_matrices"
    for run in matrices:
        out_dir = individual_root / run.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_single_confusion(
            run.metrics["confusion_matrix"],
            run.comparison_label,
            out_dir / "confusion_matrix.png",
        )

    ordered = sorted(
        matrices,
        key=lambda run: (
            MODEL_ORDER.index(run.base_model) if run.base_model in MODEL_ORDER else len(MODEL_ORDER),
            run.loss_label,
        ),
    )
    cols = 3
    rows = max(1, (len(ordered) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.2 * rows), dpi=150)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, run in zip(axes_list, ordered):
        matrix = pd.DataFrame(
            run.metrics["confusion_matrix"],
            index=CLASS_LABEL_SHORT.values(),
            columns=CLASS_LABEL_SHORT.values(),
        )
        ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="equal")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(run.comparison_label, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(col_idx, row_idx, str(int(matrix.iat[row_idx, col_idx])), ha="center", va="center", fontsize=8)
    for ax in axes_list[len(ordered):]:
        ax.axis("off")
    fig.suptitle("CMOSE Test Confusion Matrices", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(assessment_dir / "confusion_matrices_all_runs.png", bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(run: RunResult, run_viz_dir: Path) -> None:
    train_losses = run.history.get("train_losses", [])
    eval_losses = run.history.get("eval_losses", [])
    if not train_losses:
        return

    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=160)

    axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.5, color=CURVE_COLORS["train_loss"])
    if eval_losses:
        axes[0].plot(
            epochs[: len(eval_losses)],
            eval_losses,
            label="Eval Loss",
            linewidth=1.5,
            color=CURVE_COLORS["eval_loss"],
        )
    axes[0].axvline(run.history.get("best_epoch", 0), color=CURVE_COLORS["best_epoch"], linestyle="--", linewidth=1)
    axes[0].set_title(f"Loss Curves: {run.comparison_label}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    _plot_combined_evaluation_metrics(run, axes[1], title=f"Evaluation Metrics: {run.comparison_label}")

    fig.tight_layout()
    fig.savefig(run_viz_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_evaluation_metrics(run, run_viz_dir / "evaluation_metrics.png")


def _plot_combined_evaluation_metrics(run: RunResult, ax: plt.Axes, title: str) -> None:
    epochs = list(range(1, len(run.history.get("train_losses", [])) + 1))
    score_series = [
        ("eval_accuracies", "Accuracy", CURVE_COLORS["eval_accuracy"]),
        ("eval_macro_accuracies", "Macro Accuracy", CURVE_COLORS["eval_macro_accuracy"]),
        ("eval_f1_macros", "Macro F1", CURVE_COLORS["eval_f1_macro"]),
        ("eval_f1_weighteds", "Weighted F1", CURVE_COLORS["eval_f1_weighted"]),
    ]
    error_series = [
        ("eval_maes", "MAE", CURVE_COLORS["eval_mae"]),
        ("eval_mses", "MSE", CURVE_COLORS["eval_mse"]),
    ]

    handles = []
    labels = []
    last_values = []
    for key, label, color in score_series:
        values = run.history.get(key, [])
        if not values:
            continue
        line = ax.plot(epochs[: len(values)], values, label=label, linewidth=1.5, color=color)[0]
        handles.append(line)
        labels.append(label)
        last_values.append(float(values[-1]))

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.18)
    ax.axvline(run.history.get("best_epoch", 0), color=CURVE_COLORS["best_epoch"], linestyle="--", linewidth=1)

    error_ax = ax.twinx()
    for key, label, color in error_series:
        values = run.history.get(key, [])
        if not values:
            continue
        line = error_ax.plot(
            epochs[: len(values)],
            values,
            label=label,
            linewidth=1.5,
            linestyle=":",
            color=color,
        )[0]
        handles.append(line)
        labels.append(label)
        last_values.append(float(values[-1]))
    error_ax.set_ylabel("Error")
    error_ax.set_ylim(bottom=0.0)

    if handles:
        legend_order = sorted(range(len(handles)), key=lambda index: last_values[index], reverse=True)
        ax.legend(
            [handles[index] for index in legend_order],
            [labels[index] for index in legend_order],
            loc="best",
            fontsize=8,
        )


def plot_evaluation_metrics(run: RunResult, out_path: Path) -> None:
    has_eval_metric = any(
        run.history.get(key)
        for key in (
            "eval_accuracies",
            "eval_macro_accuracies",
            "eval_f1_macros",
            "eval_f1_weighteds",
            "eval_maes",
            "eval_mses",
        )
    )
    if not run.history.get("train_losses") or not has_eval_metric:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    _plot_combined_evaluation_metrics(run, ax, title=f"Evaluation Metrics: {run.comparison_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_run_report(run: RunResult, run_viz_dir: Path) -> None:
    distance_metrics = {
        "mae": run.metrics.get("mae"),
        "mse": run.metrics.get("mse"),
    }
    if any(value is None for value in distance_metrics.values()) and run.metrics.get("confusion_matrix"):
        distance_metrics.update(label_distance_metrics_from_confusion(run.metrics["confusion_matrix"]))
    for key, value in distance_metrics.items():
        if value is None:
            distance_metrics[key] = float("nan")
    lines = [
        f"# {run.comparison_label}",
        "",
        f"- Run folder: `{run.run_name}`",
        f"- Metrics file: `{run.metrics_path}`",
        f"- Accuracy: {run.metrics.get('accuracy', float('nan')):.4f}",
        f"- Macro Accuracy: {run.metrics.get('macro_accuracy', float('nan')):.4f}",
        f"- Macro F1: {run.metrics.get('f1_macro', float('nan')):.4f}",
        f"- Weighted F1: {run.metrics.get('f1_weighted', float('nan')):.4f}",
        f"- MAE: {distance_metrics.get('mae', float('nan')):.4f}",
        f"- MSE: {distance_metrics.get('mse', float('nan')):.4f}",
        f"- Best epoch: {run.history.get('best_epoch')}",
        "",
        "## Classification Report",
        "",
        "```text",
        str(run.metrics.get("classification_report", "")).rstrip(),
        "```",
    ]
    (run_viz_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def visualize_run(run: RunResult) -> None:
    run.run_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(run, run.run_dir)
    write_run_report(run, run.run_dir)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    assessment_dir = Path(args.assessment_dir)
    assessment_dir.mkdir(parents=True, exist_ok=True)

    runs = load_completed_runs(outputs_dir)
    if not runs:
        raise SystemExit(f"No completed runs found under {outputs_dir}")

    comparison_runs = filter_comparison_runs(runs)
    if not comparison_runs:
        raise SystemExit(f"No comparison runs found under {outputs_dir}")

    summary_df = build_summary_frame(comparison_runs)
    best_df = pick_best_run_per_model(summary_df)
    save_summary_csvs(summary_df, best_df, assessment_dir)
    plot_metric_bars(summary_df, best_df, assessment_dir)
    plot_metric_heatmap(summary_df, assessment_dir)
    plot_confusion_matrices(comparison_runs, assessment_dir)

    for run in runs:
        visualize_run(run)

    print(
        f"Saved per-run training curves under {outputs_dir} and CMOSE assessment charts under {assessment_dir}"
    )


if __name__ == "__main__":
    main()
