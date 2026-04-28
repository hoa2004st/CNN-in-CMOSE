"""Generate comparison and per-run visualizations from outputs/ metrics files."""

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


@dataclass
class RunResult:
    run_name: str
    metrics_path: Path
    config: dict
    history: dict
    metrics: dict

    @property
    def model_label(self) -> str:
        model = self.config.get("model", self.run_name)
        method = self.config.get("method")
        if method:
            return f"{model} ({str(method).upper()})"
        return str(model)

    @property
    def base_model(self) -> str:
        return str(self.config.get("model", self.run_name))

    @property
    def variant_label(self) -> str:
        parts = []
        method = self.config.get("method")
        loss = self.config.get("loss")
        focal_gamma = self.config.get("focal_gamma")
        if method:
            parts.append(str(method).upper())
        if loss and loss != "cross_entropy":
            if loss == "focal" and focal_gamma is not None:
                parts.append(f"focal(g={focal_gamma})")
            else:
                parts.append(str(loss))
        if not parts:
            parts.append("baseline")
        return " + ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all completed experiment runs under outputs/.",
    )
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--viz_dir", default=None, help="Default: <outputs_dir>/visualizations")
    return parser.parse_args()


def load_completed_runs(outputs_dir: Path) -> list[RunResult]:
    runs: list[RunResult] = []
    for metrics_path in sorted(outputs_dir.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs.append(
            RunResult(
                run_name=metrics_path.parent.name,
                metrics_path=metrics_path,
                config=payload.get("config", {}),
                history=payload.get("history", {}),
                metrics=payload.get("metrics", {}),
            )
        )
    return runs


def build_summary_frame(runs: list[RunResult]) -> pd.DataFrame:
    rows = []
    for run in runs:
        history = run.history
        metrics = run.metrics
        rows.append(
            {
                "run_name": run.run_name,
                "model_label": run.model_label,
                "base_model": run.base_model,
                "variant_label": run.variant_label,
                "model": run.config.get("model"),
                "method": run.config.get("method"),
                "loss": run.config.get("loss", "cross_entropy"),
                "focal_gamma": run.config.get("focal_gamma"),
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
            ["f1_macro", "macro_accuracy", "accuracy"],
            ascending=[False, False, False],
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


def plot_metric_bars(summary_df: pd.DataFrame, viz_dir: Path, *, filename: str, title: str) -> None:
    if summary_df.empty:
        return

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("macro_accuracy", "Macro Accuracy"),
        ("f1_macro", "Macro F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (column, title) in zip(axes, metric_specs, strict=False):
        ordered = summary_df.sort_values(column, ascending=False)
        sns.barplot(
            data=ordered,
            x=column,
            y="model_label",
            hue="model_label",
            dodge=False,
            legend=False,
            ax=ax,
            palette="crest",
        )
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("")
        ax.set_xlim(0.0, 1.0)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(viz_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_epoch_bars(summary_df: pd.DataFrame, viz_dir: Path, *, filename: str, title: str) -> None:
    if summary_df.empty:
        return

    ordered = summary_df.sort_values("best_epoch", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=ordered,
        x="best_epoch",
        y="model_label",
        hue="model_label",
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


def plot_model_variant_comparison(summary_df: pd.DataFrame, viz_dir: Path) -> None:
    if summary_df.empty:
        return

    multi_variant_models = [
        model_name
        for model_name, group in summary_df.groupby("base_model")
        if len(group) > 1
    ]
    if not multi_variant_models:
        return

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("macro_accuracy", "Macro Accuracy"),
        ("f1_macro", "Macro F1"),
    ]
    fig, axes = plt.subplots(
        len(multi_variant_models),
        len(metric_specs),
        figsize=(18, 5 * len(multi_variant_models)),
        squeeze=False,
    )

    for row_idx, model_name in enumerate(multi_variant_models):
        model_df = summary_df[summary_df["base_model"] == model_name].sort_values(
            ["f1_macro", "macro_accuracy", "accuracy"], ascending=[False, False, False]
        )
        for col_idx, (metric, metric_title) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            ordered = model_df.sort_values(metric, ascending=False)
            sns.barplot(
                data=ordered,
                x=metric,
                y="variant_label",
                hue="variant_label",
                dodge=False,
                legend=False,
                ax=ax,
                palette="mako",
            )
            ax.set_xlim(0.0, 1.0)
            ax.set_title(f"{model_name}: {metric_title}")
            ax.set_xlabel(metric_title)
            ax.set_ylabel("" if col_idx else "Variant")

    fig.suptitle("Within-Model Variant Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(viz_dir / "same_model_variant_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


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
                f"- `{row['base_model']}`: run `{row['run_name']}` ({row['variant_label']})",
                f"  Macro F1={row['f1_macro']:.4f}, Macro Accuracy={row['macro_accuracy']:.4f}, Accuracy={row['accuracy']:.4f}, Best epoch={row['best_epoch']}",
            ]
        )

    variant_models = [model for model, group in summary_df.groupby("base_model") if len(group) > 1]
    if variant_models:
        lines.extend(["", "## Variant Comparison By Model", ""])
        for model_name in variant_models:
            lines.append(f"### {model_name}")
            model_df = summary_df[summary_df["base_model"] == model_name].sort_values(
                ["f1_macro", "macro_accuracy", "accuracy"], ascending=[False, False, False]
            )
            for _, row in model_df.iterrows():
                lines.append(
                    f"- `{row['variant_label']}`: run `{row['run_name']}`, Macro F1={row['f1_macro']:.4f}, Macro Accuracy={row['macro_accuracy']:.4f}, Accuracy={row['accuracy']:.4f}"
                )
            lines.append("")

    (viz_dir / "comparison_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_training_curves(run: RunResult, run_viz_dir: Path) -> None:
    train_losses = run.history.get("train_losses", [])
    eval_losses = run.history.get("eval_losses", [])
    eval_accuracies = run.history.get("eval_accuracies", [])
    if not train_losses or not eval_losses or not eval_accuracies:
        return

    epochs = list(range(1, len(train_losses) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, eval_losses, label="Eval Loss", linewidth=1.5)
    axes[0].axvline(run.history.get("best_epoch", 0), color="red", linestyle="--", linewidth=1)
    axes[0].set_title(f"Loss Curves: {run.model_label}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, eval_accuracies, label="Eval Accuracy", color="green", linewidth=1.5)
    axes[1].axvline(run.history.get("best_epoch", 0), color="red", linestyle="--", linewidth=1)
    axes[1].set_title(f"Eval Accuracy: {run.model_label}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()

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
    ax.set_title(f"Confusion Matrix: {run.model_label}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(run_viz_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_run_report(run: RunResult, run_viz_dir: Path) -> None:
    lines = [
        f"# {run.model_label}",
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

    summary_df = build_summary_frame(runs)
    best_df = pick_best_run_per_model(summary_df)
    save_summary_csvs(summary_df, best_df, viz_dir)
    plot_metric_bars(
        best_df,
        viz_dir,
        filename="comparison_metrics_best_per_model.png",
        title="Best Run Comparison Across Models",
    )
    plot_best_epoch_bars(
        best_df,
        viz_dir,
        filename="best_epoch_comparison_best_per_model.png",
        title="Best Checkpoint Epoch for Best Run Per Model",
    )
    plot_model_variant_comparison(summary_df, viz_dir)
    write_comparison_report(summary_df, best_df, viz_dir)

    for run in runs:
        visualize_run(run, viz_dir)

    print(f"Saved visualizations for {len(runs)} runs under {viz_dir}")


if __name__ == "__main__":
    main()
