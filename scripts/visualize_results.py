"""Generate per-run and cross-run visualizations from experiment metrics files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LABEL_NAMES = ["Highly Disengage", "Disengage", "Engage", "Highly Engage"]
METRIC_SPECS = [
    ("accuracy", "Accuracy"),
    ("macro_accuracy", "Macro Accuracy"),
    ("f1_macro", "Macro F1"),
    ("f1_weighted", "Weighted F1"),
]
LOSS_LABELS = {
    "cross_entropy": "CE",
    "weighted_cross_entropy": "Weighted CE",
    "ordinal": "Ordinal",
}
LOSS_SLUGS = {
    "ce": "cross_entropy",
    "weighted_ce": "weighted_cross_entropy",
    "ordinal": "ordinal",
}


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
        return str(self.config.get("model", self.run_dir.parent.name))

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
        return self.base_model

    @property
    def comparison_label(self) -> str:
        return f"{self.base_model} [{self.loss_label}]"

    @property
    def variant_label(self) -> str:
        return self.loss_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize completed experiment runs under outputs/.",
    )
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--viz_dir", default=None, help="Default: <outputs_dir>/visualizations")
    return parser.parse_args()


def load_completed_runs(outputs_dir: Path) -> list[RunResult]:
    runs: list[RunResult] = []
    for metrics_path in sorted(outputs_dir.rglob("metrics.json")):
        if "visualizations" in metrics_path.parts:
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
    return [run for run in runs if not run.run_name.startswith("smoke_")]


def build_summary_frame(runs: list[RunResult]) -> pd.DataFrame:
    rows = []
    for run in runs:
        history = run.history
        metrics = run.metrics
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
                "epochs_requested": run.config.get("epochs"),
                "batch_size": run.config.get("batch_size"),
                "lr": run.config.get("lr"),
                "device": run.config.get("device"),
                "amp": run.config.get("amp"),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if not frame.empty:
        frame = frame.sort_values(
            ["f1_macro", "macro_accuracy", "accuracy", "f1_weighted"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    return frame


def pick_best_run_per_model(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    ordered = summary_df.sort_values(
        ["base_model", "f1_macro", "macro_accuracy", "accuracy", "f1_weighted"],
        ascending=[True, False, False, False, False],
    )
    return ordered.drop_duplicates(subset=["base_model"], keep="first").reset_index(drop=True)


def save_summary_csvs(summary_df: pd.DataFrame, best_df: pd.DataFrame, viz_dir: Path) -> None:
    summary_df.to_csv(viz_dir / "summary_table_all_runs.csv", index=False)
    best_df.to_csv(viz_dir / "summary_table_best_per_model.csv", index=False)


def plot_metric_bars(
    summary_df: pd.DataFrame,
    viz_dir: Path,
    *,
    filename: str,
    title: str,
    label_column: str,
) -> None:
    if summary_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (column, metric_title) in zip(axes.flat, METRIC_SPECS, strict=False):
        ordered = summary_df.sort_values(column, ascending=False)
        sns.barplot(
            data=ordered,
            x=column,
            y=label_column,
            hue=label_column,
            dodge=False,
            legend=False,
            ax=ax,
            palette="crest",
        )
        ax.set_title(metric_title)
        ax.set_xlabel(metric_title)
        ax.set_ylabel("")
        ax.set_xlim(0.0, 1.0)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(viz_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_epoch_bars(summary_df: pd.DataFrame, viz_dir: Path, *, filename: str, title: str) -> None:
    if summary_df.empty or summary_df["best_epoch"].isna().all():
        return

    ordered = summary_df.sort_values("best_epoch", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(
        data=ordered,
        x="best_epoch",
        y="comparison_label",
        hue="comparison_label",
        dodge=False,
        legend=False,
        ax=ax,
        palette="flare",
    )
    ax.set_title(title)
    ax.set_xlabel("Best Epoch")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(viz_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_comparison_within_model(summary_df: pd.DataFrame, viz_dir: Path) -> None:
    if summary_df.empty:
        return

    model_names = [name for name, group in summary_df.groupby("base_model") if len(group) > 1]
    if not model_names:
        return

    fig, axes = plt.subplots(
        len(model_names),
        len(METRIC_SPECS),
        figsize=(22, 5 * len(model_names)),
        squeeze=False,
    )
    for row_idx, model_name in enumerate(model_names):
        model_df = summary_df[summary_df["base_model"] == model_name]
        for col_idx, (metric, metric_title) in enumerate(METRIC_SPECS):
            ax = axes[row_idx][col_idx]
            ordered = model_df.sort_values(metric, ascending=False)
            sns.barplot(
                data=ordered,
                x=metric,
                y="loss_label",
                hue="loss_label",
                dodge=False,
                legend=False,
                ax=ax,
                palette="mako",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_title(f"{model_name}: {metric_title}")
            ax.set_xlabel(metric_title)
            ax.set_ylabel("" if col_idx else "Loss")
    fig.suptitle("Loss Comparison Within Each Model", fontsize=14)
    fig.tight_layout()
    fig.savefig(viz_dir / "loss_comparison_within_model.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_within_loss(summary_df: pd.DataFrame, viz_dir: Path) -> None:
    if summary_df.empty:
        return

    loss_names = [name for name, group in summary_df.groupby("loss") if len(group) > 1]
    if not loss_names:
        return

    fig, axes = plt.subplots(
        len(loss_names),
        len(METRIC_SPECS),
        figsize=(22, 5 * len(loss_names)),
        squeeze=False,
    )
    for row_idx, loss_name in enumerate(loss_names):
        loss_df = summary_df[summary_df["loss"] == loss_name]
        loss_label = LOSS_LABELS.get(loss_name, loss_name)
        for col_idx, (metric, metric_title) in enumerate(METRIC_SPECS):
            ax = axes[row_idx][col_idx]
            ordered = loss_df.sort_values(metric, ascending=False)
            sns.barplot(
                data=ordered,
                x=metric,
                y="base_model",
                hue="base_model",
                dodge=False,
                legend=False,
                ax=ax,
                palette="rocket",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_title(f"{loss_label}: {metric_title}")
            ax.set_xlabel(metric_title)
            ax.set_ylabel("" if col_idx else "Model")
    fig.suptitle("Model Comparison Within Each Loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(viz_dir / "model_comparison_within_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmaps(summary_df: pd.DataFrame, viz_dir: Path) -> None:
    if summary_df.empty or {"base_model", "loss"}.difference(summary_df.columns):
        return

    for metric, metric_title in METRIC_SPECS:
        pivot = summary_df.pivot_table(index="base_model", columns="loss_label", values=metric, aggfunc="max")
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.8)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"{metric_title} Heatmap by Model and Loss")
        ax.set_xlabel("Loss")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(viz_dir / f"heatmap_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        pivot.to_csv(viz_dir / f"heatmap_{metric}.csv")


def write_comparison_report(summary_df: pd.DataFrame, best_df: pd.DataFrame, viz_dir: Path) -> None:
    lines = [
        "# Comparison Summary",
        "",
        "## Best Run Per Model",
        "",
    ]
    for _, row in best_df.sort_values(
        ["f1_macro", "macro_accuracy", "accuracy"], ascending=[False, False, False]
    ).iterrows():
        lines.extend(
            [
                f"- `{row['base_model']}`: run `{row['run_name']}` ({row['loss_label']})",
                f"  Macro F1={row['f1_macro']:.4f}, Macro Accuracy={row['macro_accuracy']:.4f}, Accuracy={row['accuracy']:.4f}, Weighted F1={row['f1_weighted']:.4f}, Best epoch={row['best_epoch']}",
            ]
        )

    if not summary_df.empty:
        lines.extend(["", "## Best Run Per Loss", ""])
        loss_best_df = (
            summary_df.sort_values(
                ["loss", "f1_macro", "macro_accuracy", "accuracy", "f1_weighted"],
                ascending=[True, False, False, False, False],
            )
            .drop_duplicates(subset=["loss"], keep="first")
            .reset_index(drop=True)
        )
        for _, row in loss_best_df.iterrows():
            lines.extend(
                [
                    f"- `{row['loss_label']}`: `{row['base_model']}` (`{row['run_name']}`)",
                    f"  Macro F1={row['f1_macro']:.4f}, Macro Accuracy={row['macro_accuracy']:.4f}, Accuracy={row['accuracy']:.4f}, Weighted F1={row['f1_weighted']:.4f}",
                ]
            )

    variant_models = [model for model, group in summary_df.groupby("base_model") if len(group) > 1]
    if variant_models:
        lines.extend(["", "## Loss Comparison By Model", ""])
        for model_name in variant_models:
            lines.append(f"### {model_name}")
            model_df = summary_df[summary_df["base_model"] == model_name].sort_values(
                ["f1_macro", "macro_accuracy", "accuracy"], ascending=[False, False, False]
            )
            for _, row in model_df.iterrows():
                lines.append(
                    f"- `{row['loss_label']}`: run `{row['run_name']}`, Macro F1={row['f1_macro']:.4f}, Macro Accuracy={row['macro_accuracy']:.4f}, Accuracy={row['accuracy']:.4f}, Weighted F1={row['f1_weighted']:.4f}"
                )
            lines.append("")

    grouped_losses = [loss for loss, group in summary_df.groupby("loss") if len(group) > 1]
    if grouped_losses:
        lines.extend(["## Model Comparison By Loss", ""])
        for loss_name in grouped_losses:
            loss_label = LOSS_LABELS.get(loss_name, loss_name)
            lines.append(f"### {loss_label}")
            loss_df = summary_df[summary_df["loss"] == loss_name].sort_values(
                ["f1_macro", "macro_accuracy", "accuracy"], ascending=[False, False, False]
            )
            for _, row in loss_df.iterrows():
                lines.append(
                    f"- `{row['base_model']}`: run `{row['run_name']}`, Macro F1={row['f1_macro']:.4f}, Macro Accuracy={row['macro_accuracy']:.4f}, Accuracy={row['accuracy']:.4f}, Weighted F1={row['f1_weighted']:.4f}"
                )
            lines.append("")

    (viz_dir / "comparison_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_training_curves(run: RunResult, run_viz_dir: Path) -> None:
    train_losses = run.history.get("train_losses", [])
    eval_losses = run.history.get("eval_losses", [])
    eval_accuracies = run.history.get("eval_accuracies", [])
    eval_macro_accuracies = run.history.get("eval_macro_accuracies", [])
    eval_f1_macros = run.history.get("eval_f1_macros", [])
    eval_f1_weighteds = run.history.get("eval_f1_weighteds", [])
    if not train_losses or not eval_accuracies:
        return

    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    if eval_losses:
        axes[0].plot(epochs[: len(eval_losses)], eval_losses, label="Eval Loss", linewidth=1.5)
    axes[0].axvline(run.history.get("best_epoch", 0), color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"Loss Curves: {run.comparison_label}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs[: len(eval_accuracies)], eval_accuracies, label="Eval Accuracy", linewidth=1.5)
    if eval_macro_accuracies:
        axes[1].plot(
            epochs[: len(eval_macro_accuracies)],
            eval_macro_accuracies,
            label="Eval Macro Accuracy",
            linewidth=1.5,
        )
    axes[1].axvline(run.history.get("best_epoch", 0), color="red", linestyle="--", linewidth=1)
    axes[1].set_title(f"Evaluation Accuracy: {run.comparison_label}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

    if eval_f1_macros:
        axes[2].plot(epochs[: len(eval_f1_macros)], eval_f1_macros, label="Eval Macro F1", linewidth=1.5)
    if eval_f1_weighteds:
        axes[2].plot(
            epochs[: len(eval_f1_weighteds)],
            eval_f1_weighteds,
            label="Eval Weighted F1",
            linewidth=1.5,
        )
    axes[2].axvline(run.history.get("best_epoch", 0), color="red", linestyle="--", linewidth=1)
    axes[2].set_title(f"Evaluation F1: {run.comparison_label}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(run_viz_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(run: RunResult, run_viz_dir: Path) -> None:
    matrix = run.metrics.get("confusion_matrix")
    if not matrix:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {run.comparison_label}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(run_viz_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_run_report(run: RunResult, run_viz_dir: Path) -> None:
    lines = [
        f"# {run.comparison_label}",
        "",
        f"- Run folder: `{run.run_name}`",
        f"- Metrics file: `{run.metrics_path}`",
        f"- Accuracy: {run.metrics.get('accuracy', float('nan')):.4f}",
        f"- Macro Accuracy: {run.metrics.get('macro_accuracy', float('nan')):.4f}",
        f"- Macro F1: {run.metrics.get('f1_macro', float('nan')):.4f}",
        f"- Weighted F1: {run.metrics.get('f1_weighted', float('nan')):.4f}",
        f"- Best epoch: {run.history.get('best_epoch')}",
        "",
        "## Classification Report",
        "",
        "```text",
        str(run.metrics.get("classification_report", "")).rstrip(),
        "```",
    ]
    (run_viz_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def visualize_run(run: RunResult, viz_dir: Path) -> None:
    run_viz_dir = viz_dir / run.run_name
    run_viz_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(run, run_viz_dir)
    plot_confusion_matrix(run, run_viz_dir)
    write_run_report(run, run_viz_dir)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    viz_dir = Path(args.viz_dir) if args.viz_dir else outputs_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    runs = load_completed_runs(outputs_dir)
    if not runs:
        raise SystemExit(f"No completed runs found under {outputs_dir}")

    comparison_runs = filter_comparison_runs(runs)
    if not comparison_runs:
        raise SystemExit(f"No comparison runs found under {outputs_dir}")

    summary_df = build_summary_frame(comparison_runs)
    best_df = pick_best_run_per_model(summary_df)
    save_summary_csvs(summary_df, best_df, viz_dir)
    plot_metric_bars(
        best_df,
        viz_dir,
        filename="comparison_metrics_best_per_model.png",
        title="Best Run Comparison Across Models",
        label_column="base_model",
    )
    plot_metric_bars(
        summary_df,
        viz_dir,
        filename="comparison_metrics_all_runs.png",
        title="All Run Comparison Across Model and Loss",
        label_column="comparison_label",
    )
    plot_best_epoch_bars(
        summary_df,
        viz_dir,
        filename="best_epoch_comparison_all_runs.png",
        title="Best Checkpoint Epoch Across All Runs",
    )
    plot_loss_comparison_within_model(summary_df, viz_dir)
    plot_model_comparison_within_loss(summary_df, viz_dir)
    plot_metric_heatmaps(summary_df, viz_dir)
    write_comparison_report(summary_df, best_df, viz_dir)

    for run in runs:
        visualize_run(run, viz_dir)

    print(f"Saved visualizations for {len(runs)} runs under {viz_dir}")


if __name__ == "__main__":
    main()
