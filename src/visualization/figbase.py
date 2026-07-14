"""Shared plotting setup for thesis figures: consistent style + PNG saving.

Every ``figures_*.py`` module imports ``new_fig`` / ``save`` from here so figures share a
uniform look and land in ``outputs/thesis/figures`` as PNG. The save resolution is controlled
by ``SAVE_DPI`` (default 150; override with the ``THESIS_FIG_DPI`` environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; never tries to open a window
import matplotlib.pyplot as plt

from src.output_paths import OUTPUT_ROOT

THESIS_DIR = OUTPUT_ROOT / "thesis"
FIGURE_DIR = THESIS_DIR / "figures"
TABLE_DIR = THESIS_DIR / "tables"

# Resolution (dots-per-inch) of the saved PNGs in outputs/thesis/figures.
# Lower = smaller files / lower resolution. Change it here, or without editing code
# via the THESIS_FIG_DPI environment variable, e.g. `THESIS_FIG_DPI=200`.
SAVE_DPI = int(os.environ.get("THESIS_FIG_DPI", "150"))

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def apply_style() -> None:
    plt.rcParams.update(_RC)


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def new_fig(*args, **kwargs):
    apply_style()
    return plt.subplots(*args, **kwargs)


_CELL_TEXTS_ATTR = "_thesis_cell_texts"


def mark_cell_texts(ax, texts) -> None:
    """Register the ``ax.text`` annotations of one heatmap axes for later auto-sizing.

    Called by the heatmap drawing helpers; :func:`autosize_cell_text` reads the registration
    back once the whole figure (all panels, colorbars, titles) is assembled, so the cell
    width it measures is the final one.
    """
    setattr(ax, _CELL_TEXTS_ATTR, list(texts))


def autosize_cell_text(fig, *, frac: float = 0.55) -> None:
    """Scale every registered heatmap annotation so the widest spans ``frac`` of a cell width.

    Call once, after the figure is fully assembled and before saving. The canvas is drawn to
    fix the final axes geometry (constrained layout has by then accounted for colorbars and
    titles) and to obtain a renderer; for each registered axes the cell width and each label's
    width are measured in display pixels, and a single font size is applied so the widest label
    just reaches ``frac`` of the cell width — no label overflows and they stay uniform per axes.
    The width ratio is invariant to the save DPI, so the fraction holds in the exported PNG, and
    annotation font sizes do not feed back into constrained layout, so the axes do not move.
    """
    axes_with_text = [ax for ax in fig.axes if getattr(ax, _CELL_TEXTS_ATTR, None)]
    if not axes_with_text:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes_with_text:
        texts = getattr(ax, _CELL_TEXTS_ATTR)
        (x0, _), (x1, _) = ax.transData.transform([(0.0, 0.0), (1.0, 0.0)])
        cell_w = abs(x1 - x0)
        widest = max(t.get_window_extent(renderer=renderer).width for t in texts)
        if cell_w <= 0 or widest <= 0:
            continue
        new_size = texts[0].get_fontsize() * (frac * cell_w / widest)
        for t in texts:
            t.set_fontsize(new_size)


def save(fig, name: str, *, directory: Path | None = None) -> Path:
    """Save ``fig`` as ``<name>.png`` (at ``SAVE_DPI``); return the path.

    Saving to the canonical FIGURE_DIR also refreshes the thesis's downsized print copy in
    ``documents/thesis/Figure`` so regenerating a single figure can't leave the compiled thesis
    showing a stale version. A custom ``directory`` (e.g. a scratch dir) skips the publish step.
    """
    directory = directory or FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{name}.png"
    fig.savefig(png_path)
    plt.close(fig)
    if directory == FIGURE_DIR:
        # Lazy import: thesis_latex imports this module, so importing it at top level would cycle.
        from src.analysis.thesis_latex import publish_figure

        publish_figure(png_path)
    return png_path
