"""Shared plotting setup for thesis figures: consistent style + PNG saving.

Every ``figures_*.py`` module imports ``new_fig`` / ``save`` from here so figures share a
uniform look and land in ``outputs/thesis/figures`` as PNG (300 dpi).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; never tries to open a window
import matplotlib.pyplot as plt

from src.output_paths import OUTPUT_ROOT

THESIS_DIR = OUTPUT_ROOT / "thesis"
FIGURE_DIR = THESIS_DIR / "figures"
TABLE_DIR = THESIS_DIR / "tables"

_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
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


def save(fig, name: str, *, directory: Path | None = None) -> Path:
    """Save ``fig`` as ``<name>.png`` (300 dpi); return the path."""
    directory = directory or FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    png_path = directory / f"{name}.png"
    fig.savefig(png_path)
    plt.close(fig)
    return png_path
