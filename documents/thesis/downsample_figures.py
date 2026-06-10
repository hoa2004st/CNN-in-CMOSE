#!/usr/bin/env python3
r"""Downsample thesis figures to a target DPI at their *printed* size.

Each figure is placed by ``\includegraphics[width=FRAC\linewidth]{name}`` in the
thesis ``.tex`` sources, so its printed width is ``FRAC * textwidth``.  A PNG only
needs ``TARGET_DPI`` pixels per printed inch to look sharp at that size; anything
beyond that is wasted pixels that slow down compilation and PDF rendering.

This script scans the sources for the width each figure is shown at, then for
every image in ``Figure/`` resamples it so the printed width corresponds to
``TARGET_DPI`` (default 150).  It only ever *shrinks* an image -- a figure that is
already at or below 150 dpi is left untouched -- and writes the DPI metadata to
match.  Originals are regenerable (``outputs/thesis/figures``), and the files are
git-tracked, so an in-place rewrite is safe; use ``--dry-run`` to preview first.

Usage:
    python downsample_figures.py --dry-run     # report what would change
    python downsample_figures.py               # apply in place
    python downsample_figures.py --dpi 150 --textwidth-cm 15.0
"""
from __future__ import annotations

import argparse
import re
from io import BytesIO
from pathlib import Path

from PIL import Image

THESIS_DIR = Path(__file__).resolve().parent
FIGURE_DIR = THESIS_DIR / "Figure"

# \includegraphics[<opts>]{<name>}  -- opts optional, single line
INCLUDE_RE = re.compile(r"\\includegraphics\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{(?P<name>[^}]+)\}")
# width = <frac>\linewidth  (frac optional -> 1.0 for bare \linewidth/\textwidth)
WIDTH_RE = re.compile(r"width\s*=\s*(?P<frac>[0-9]*\.?[0-9]+)?\s*\\(?:line|text)width")
# Pillow resampling needs a true-colour/greyscale mode
RESAMPLE_SAFE = {"L", "RGB", "RGBA"}


def smallest_png(img: Image.Image, dpi: float) -> bytes:
    """Encode ``img`` as PNG, returning the smaller of truecolour vs palette.

    Matplotlib often writes compact palette PNGs; resampling produces a
    truecolour image, which can re-encode *larger* despite having fewer pixels.
    Plots and block diagrams use few colours, so an adaptive 256-colour palette
    is usually much smaller and visually identical -- but we only keep it if it
    actually wins, so photographic images stay truecolour.
    """
    # Drop a fully-opaque alpha channel so the palette path can apply.
    if img.mode == "RGBA" and img.getchannel("A").getextrema() == (255, 255):
        img = img.convert("RGB")

    candidates: list[bytes] = []
    buf = BytesIO()
    img.save(buf, format="PNG", dpi=(dpi, dpi), optimize=True)
    candidates.append(buf.getvalue())

    try:
        if img.mode == "RGBA":
            palette = img.quantize(colors=256, method=Image.FASTOCTREE)
        else:
            palette = img.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256)
        buf = BytesIO()
        palette.save(buf, format="PNG", dpi=(dpi, dpi), optimize=True)
        candidates.append(buf.getvalue())
    except Exception:
        pass

    return min(candidates, key=len)


def scan_printed_widths(tex_files: list[Path]) -> dict[str, float]:
    """Map figure basename -> largest width fraction it is displayed at."""
    widths: dict[str, float] = {}
    for tex in tex_files:
        text = tex.read_text(encoding="utf-8", errors="ignore")
        for m in INCLUDE_RE.finditer(text):
            name = Path(m.group("name").strip()).name
            wm = WIDTH_RE.search(m.group("opts") or "")
            if wm:
                frac = float(wm.group("frac")) if wm.group("frac") else 1.0
            else:
                frac = 1.0  # no width key -> assume it spans the full line
            widths[name] = max(widths.get(name, 0.0), frac)
    return widths


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dpi", type=float, default=150.0, help="target dots per inch (default 150)")
    ap.add_argument(
        "--textwidth-cm",
        type=float,
        default=15.0,
        help="LaTeX \\textwidth in cm; a4 with 3.5+2.5 cm margins = 15.0",
    )
    ap.add_argument(
        "--default-frac",
        type=float,
        default=1.0,
        help="width fraction assumed for figures not referenced in any .tex",
    )
    ap.add_argument("--dry-run", action="store_true", help="report only; do not modify files")
    args = ap.parse_args()

    if not FIGURE_DIR.is_dir():
        print(f"error: {FIGURE_DIR} not found")
        return 1

    textwidth_in = args.textwidth_cm / 2.54
    widths = scan_printed_widths(sorted(THESIS_DIR.rglob("*.tex")))

    images = sorted(FIGURE_DIR.glob("*.png"))
    total_before = total_after = 0
    changed = 0

    for path in images:
        frac = widths.get(path.name, args.default_frac)
        target_px = max(1, round(args.dpi * frac * textwidth_in))
        size_before = path.stat().st_size
        total_before += size_before

        with Image.open(path) as im:
            w, h = im.size
            if w <= target_px:
                total_after += size_before
                print(f"  skip   {path.name:<40s} {w}x{h}  @{frac:g}lw  (<= {target_px}px)")
                continue

            new_h = max(1, round(h * target_px / w))
            change = f"{w}x{h} -> {target_px}x{new_h}  @{frac:g}lw"

            if args.dry_run:
                total_after += size_before
                changed += 1
                print(f"  would  {path.name:<40s} {change}")
                continue

            work = im if im.mode in RESAMPLE_SAFE else im.convert("RGBA")
            work = work.resize((target_px, new_h), Image.LANCZOS)
            data = smallest_png(work, args.dpi)
        path.write_bytes(data)

        size_after = path.stat().st_size
        total_after += size_after
        changed += 1
        print(
            f"  resize {path.name:<40s} {change}  "
            f"{size_before // 1024}KB -> {size_after // 1024}KB"
        )

    verb = "would shrink" if args.dry_run else "shrank"
    print(
        f"\n{verb} {changed} image(s); {len(images) - changed} already <= {args.dpi:g} dpi.\n"
        f"Figure/ total: {total_before // 1024} KB -> {total_after // 1024} KB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
