"""Entry point: regenerate every thesis figure and table into outputs/thesis/.

Usage:
    python -m src.analysis.make_thesis_artifacts

Figures land in ``outputs/thesis/figures`` (PNG + PDF); tables in ``outputs/thesis/tables``
(Markdown + LaTeX). Pure post-processing of existing result CSVs / metrics.json — it never
loads a checkpoint or touches the GPU, so it runs in seconds on CPU.
"""

from __future__ import annotations

from src.analysis import tables, thesis_latex
from src.visualization import (
    figbase,
    figures_confusion,
    figures_crossdomain,
    figures_dataset,
    figures_hybrid,
    figures_loss,
    figures_models,
    figures_private,
)

_FIGURE_MODULES = [
    ("dataset", figures_dataset),
    ("base models", figures_models),
    ("loss / training", figures_loss),
    ("hybrid ablation", figures_hybrid),
    ("cross-domain", figures_crossdomain),
    ("private set", figures_private),
    ("confusion / error", figures_confusion),
]


def main() -> None:
    figbase.ensure_dirs()
    figure_paths = []
    for label, module in _FIGURE_MODULES:
        paths = module.make_all()
        figure_paths.extend(paths)
        print(f"[figures] {label}: {len(paths)} figure(s)")
        for path in paths:
            print(f"          - {path}")

    table_paths = tables.make_all()
    print(f"[tables] {len(table_paths)} table(s)")
    for path in table_paths:
        print(f"          - {path}")

    latex_paths = thesis_latex.make_all()
    print(f"[latex] {len(latex_paths)} Overleaf snippet/chapter file(s)")
    for path in latex_paths:
        print(f"          - {path}")

    print(f"\nDone. {len(figure_paths)} figures -> {figbase.FIGURE_DIR}")
    print(f"      {len(table_paths)} tables  -> {figbase.TABLE_DIR}")
    print(f"      {len(latex_paths)} LaTeX files (figure snippets + {thesis_latex.CHAPTERS_LATEX_DIR})")


if __name__ == "__main__":
    main()
