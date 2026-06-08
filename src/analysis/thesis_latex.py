r"""Assemble the compilable LaTeX thesis project from the Markdown chapter drafts.

The thesis follows the HUST ``documents/Thesis_Template`` layout: a ``main.tex`` driver
``\subfile``s one ``Chapter/*.tex`` per chapter, pulls figures from ``Figure/`` (via
``\graphicspath``) and result tables from ``Table/``. The static scaffolding (``main.tex``,
``Cover.tex``, ``Cover2.tex``, ``glossary.tex``, ``lstlisting.tex``, ``reference.bib``) is
hand-authored once; *this* module regenerates the parts that depend on the analysis results:

* ``documents/thesis/Chapter/*.tex`` — each ``documents/thesis/*.md`` chapter rendered as a
  ``subfiles`` document (no top-level ``\chapter`` — the driver supplies that). The front
  matter splits into ``0_2_Acknowledgment.tex`` + ``0_3_Abstract.tex``.
* ``documents/thesis/Figure/`` — the analysis PNGs, copied from ``outputs/thesis/figures``.
* ``documents/thesis/Table/`` — the result tables rendered to LaTeX from the same specs
  :mod:`src.analysis.tables` builds, ``\input`` into the Numerical Results chapter at the point
  each is first referenced.

Figures are emitted as ``\includegraphics{<name>}`` (resolved through the driver's
``\graphicspath``); tables as ``\input{Table/<name>}``. Conversion of inline Markdown / escaping
lives in :mod:`src.analysis.latexfmt`.
"""

from __future__ import annotations

import io
import re
import shutil
from pathlib import Path

from PIL import Image

from src.analysis import latexfmt
from src.visualization.figbase import FIGURE_DIR, TABLE_DIR

CHAPTERS_DIR = Path("documents/thesis")            # Markdown source (authoring)
PROJECT_DIR = CHAPTERS_DIR                          # LaTeX project root
CHAPTER_TEX_DIR = PROJECT_DIR / "Chapter"
PROJECT_FIGURE_DIR = PROJECT_DIR / "Figure"
PROJECT_TABLE_DIR = PROJECT_DIR / "Table"

# Print target for the Figure/ copies. Figures are \includegraphics'd at INCLUDE_LINEWIDTH_FRAC
# of the text width (see latexfmt.figure_block). On A4 the text width is
# 21cm - 3.5cm(left) - 2.5cm(right) = 15cm, so the on-page width is ~13.5cm. At the 300 dpi print
# standard that needs only ~1594 px, yet the generated PNGs are 500-630 dpi (2400-3330 px wide) —
# oversampled, slower to ship/compile, no sharper in print. So the Figure/ copies are downsized to
# exactly 300 dpi on the page; the full-resolution originals stay in outputs/thesis/figures.
A4_TEXT_WIDTH_IN = 15.0 / 2.54
INCLUDE_LINEWIDTH_FRAC = 0.9
PRINT_DPI = 300
PRINT_TARGET_WIDTH_PX = round(A4_TEXT_WIDTH_IN * INCLUDE_LINEWIDTH_FRAC * PRINT_DPI)

# Markdown stem -> subfile path under Chapter/ (template numbering; Theoretical Analysis skipped
# so Numerical Results is Chapter 4 and Conclusions Chapter 5). Front matter is handled apart.
_CHAPTER_TEX_NAME = {
    "01_introduction": "1_Introduction.tex",
    "02_literature_review": "2_Literature_review.tex",
    "03_methodology": "3_Methodology.tex",
    "04_results": "4_Numerical_results.tex",
    "05_conclusion": "5_Conclusions.tex",
}
_FRONT_MATTER_STEM = "00_front_matter"

_FIGURE_EMBED_RE = re.compile(r"!\[(?P<cap>.*?)\]\((?P<path>[^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OLIST_RE = re.compile(r"^(\s*)\d+\.\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
# Markdown heading level -> LaTeX command. Level 1 (the chapter title) is dropped: the driver's
# \chapter provides it. Everything shifts up one rung relative to a standalone document.
_HEADING_CMDS = {2: "section", 3: "subsection", 4: "subsubsection", 5: "paragraph"}
_TABLE_REF_RE = re.compile(r"\bT([1-9])\b")


def _label_from_path(path: str) -> str:
    return "fig:" + Path(path).stem


def _normalize(md_text: str) -> str:
    """Drop Markdown HTML entities pdfLaTeX never sees as text."""
    return md_text.replace(" ", " ").replace("&nbsp;", " ")


# --- table embedding ----------------------------------------------------------

def _table_file_map() -> dict[str, str]:
    """``"1" -> "T1_dataset_stats"`` from the ``outputs/thesis/tables/T<n>_*.md`` previews.

    Keyed off the Markdown previews (always present) so chapter embedding does not depend on the
    ``.tex`` having been rendered yet; the ``.tex`` shares the stem.
    """
    mapping: dict[str, str] = {}
    for md in sorted(TABLE_DIR.glob("T[1-9]_*.md")):
        mapping[md.stem[1]] = md.stem
    return mapping


def _emit_tables(text: str, table_map: dict[str, str], seen: set[str], out: list[str]) -> None:
    """Append an ``\\input`` for every table first referenced (``T<n>``) in ``text``."""
    for num in _TABLE_REF_RE.findall(text):
        name = table_map.get(num)
        if name and name not in seen:
            seen.add(name)
            out.append(f"\\input{{Table/{name}}}")


# --- markdown body -> latex ---------------------------------------------------

def _strip_heading_number(text: str) -> str:
    text = re.sub(r"^Chapter\s+\d+\.?\s*", "", text)
    text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text)
    return text


def _heading(level: int, text: str) -> str | None:
    if level <= 1:  # the chapter title — supplied by \chapter in main.tex
        return None
    cmd = _HEADING_CMDS.get(level, "paragraph")
    return f"\\{cmd}{{{latexfmt.inline_md_to_latex(_strip_heading_number(text))}}}"


def _md_table(rows: list[str]) -> str:
    parsed = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = parsed[0], parsed[2:]  # row 1 is the |---| separator
    align = "l" * len(header)
    out = ["\\begin{center}", f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    out.append(" & ".join(latexfmt.inline_md_to_latex(c) for c in header) + r" \\")
    out.append("\\midrule")
    for r in body:
        out.append(" & ".join(latexfmt.inline_md_to_latex(c) for c in r) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(out)


def _convert_body(md_text: str, *, embed_tables: bool = False) -> str:
    """Render a chapter's Markdown body to LaTeX (no subfile wrapper, no chapter title)."""
    table_map = _table_file_map() if embed_tables else {}
    seen_tables: set[str] = set()
    lines = _normalize(md_text).splitlines()
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        if not line.strip():
            out.append("")
            i += 1
            continue

        m = _HEADING_RE.match(line)
        if m:
            rendered = _heading(len(m.group(1)), m.group(2))
            if rendered is not None:
                out.append(rendered)
            i += 1
            continue

        # blockquote -> LaTeX comments (TODO / note blocks)
        if line.lstrip().startswith(">"):
            out.append("% --- note from draft (Markdown blockquote) ---")
            while i < n and lines[i].lstrip().startswith(">"):
                content = lines[i].lstrip()[1:].strip()
                out.append("% " + content if content else "%")
                i += 1
            continue

        # standalone figure embed
        fm = _FIGURE_EMBED_RE.match(line.strip())
        if fm and fm.group(0) == line.strip():
            out.append(
                latexfmt.figure_block(
                    caption=fm.group("cap").strip(),
                    graphics_path=Path(fm.group("path")).name,  # resolved via \graphicspath
                    label=_label_from_path(fm.group("path")),
                )
            )
            i += 1
            continue

        # pipe table
        if line.lstrip().startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            block = []
            while i < n and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_md_table(block))
            continue

        # bullet (-/*/+) or ordered (1.) list, with wrapped continuation lines
        list_re = _LIST_RE if _LIST_RE.match(line) else (_OLIST_RE if _OLIST_RE.match(line) else None)
        if list_re is not None:
            env = "itemize" if list_re is _LIST_RE else "enumerate"
            items: list[str] = []
            while i < n:
                lm = list_re.match(lines[i])
                if lm:
                    items.append(lm.group(2).strip())
                    i += 1
                elif lines[i].strip() and lines[i].startswith(" ") and not _HEADING_RE.match(lines[i]):
                    items[-1] += " " + lines[i].strip()  # continuation
                    i += 1
                else:
                    break
            out.append(f"\\begin{{{env}}}")
            out += [f"  \\item {latexfmt.inline_md_to_latex(it)}" for it in items]
            out.append(f"\\end{{{env}}}")
            if embed_tables:
                _emit_tables(" ".join(items), table_map, seen_tables, out)
            continue

        # paragraph: gather until blank / next block
        para: list[str] = []
        while i < n and lines[i].strip() and not _HEADING_RE.match(lines[i]) \
                and not lines[i].lstrip().startswith(">") \
                and not lines[i].lstrip().startswith("|") \
                and not _LIST_RE.match(lines[i]) \
                and not _OLIST_RE.match(lines[i]) \
                and not _FIGURE_EMBED_RE.match(lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        joined = " ".join(para)
        out.append(latexfmt.inline_md_to_latex(joined))
        if embed_tables:
            _emit_tables(joined, table_map, seen_tables, out)

    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# --- subfile wrappers ---------------------------------------------------------

def _wrap_subfile(body: str, source_name: str) -> str:
    return "\n".join([
        "\\documentclass[../main.tex]{subfiles}",
        f"% Auto-generated from {source_name} by src.analysis.thesis_latex — do not edit by hand.",
        "\\begin{document}",
        "",
        body.strip(),
        "",
        "\\end{document}",
        "",
    ])


def _wrap_frontmatter(title: str, body: str, *, pagenumbering: str | None = None) -> str:
    parts = ["\\documentclass[../main.tex]{subfiles}"]
    if pagenumbering:
        parts.append(f"\\pagenumbering{{{pagenumbering}}}")
    parts += [
        "\\begin{document}",
        "\\begin{center}",
        f"    \\Large{{\\textbf{{{title}}}}}\\\\",
        "\\end{center}",
        "\\vspace{1cm}",
        body.strip(),
        "\\end{document}",
        "",
    ]
    return "\n".join(parts)


def _split_frontmatter(md_text: str) -> dict[str, str]:
    """Split ``00_front_matter.md`` into ``{lower_title: body_md}`` by its ``#`` headings."""
    sections: dict[str, str] = {}
    title, buf = None, []
    for line in _normalize(md_text).splitlines():
        m = re.match(r"^#\s+(.*)$", line)
        if m:
            if title is not None:
                sections[title.lower()] = "\n".join(buf).strip()
            title, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if title is not None:
        sections[title.lower()] = "\n".join(buf).strip()
    return sections


# --- public API ---------------------------------------------------------------

def convert_chapters() -> list[Path]:
    CHAPTER_TEX_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # front matter -> two subfiles
    fm_src = CHAPTERS_DIR / f"{_FRONT_MATTER_STEM}.md"
    if fm_src.exists():
        sections = _split_frontmatter(fm_src.read_text(encoding="utf-8"))
        ack = _convert_body(sections.get("acknowledgement", sections.get("acknowledgment", "")))
        out = CHAPTER_TEX_DIR / "0_2_Acknowledgment.tex"
        out.write_text(_wrap_frontmatter("ACKNOWLEDGMENT", ack, pagenumbering="roman"), encoding="utf-8")
        written.append(out)

        abstract = _convert_body(sections.get("abstract", ""))
        out = CHAPTER_TEX_DIR / "0_3_Abstract.tex"
        out.write_text(_wrap_frontmatter("ABSTRACT", abstract), encoding="utf-8")
        written.append(out)

    # numbered chapters
    for stem, tex_name in _CHAPTER_TEX_NAME.items():
        src = CHAPTERS_DIR / f"{stem}.md"
        if not src.exists():
            continue
        body = _convert_body(src.read_text(encoding="utf-8"), embed_tables=(stem == "04_results"))
        out = CHAPTER_TEX_DIR / tex_name
        out.write_text(_wrap_subfile(body, f"documents/thesis/{src.name}"), encoding="utf-8")
        written.append(out)

    return written


def _encode_png(im: Image.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True, dpi=(PRINT_DPI, PRINT_DPI))
    return buf.getvalue()


def _copy_figure_for_print(src: Path, dest: Path) -> None:
    """Write ``src`` to ``dest``, downsized to PRINT_TARGET_WIDTH_PX (300 dpi on A4) if wider.

    Figures already at/below the print width are copied verbatim — their original bytes are the
    best encoding we have. Wider ones are LANCZOS-downscaled, then encoded both as true-colour and
    as a 256-colour palette; the smaller is kept. matplotlib plots use few distinct colours, so the
    palette copy is normally far smaller and visually identical at 300 dpi — and keeping the smaller
    of the two means re-encoding can never inflate a figure past its true-colour size.
    """
    with Image.open(src) as im:
        if im.width <= PRINT_TARGET_WIDTH_PX:
            shutil.copyfile(src, dest)
            return
        if im.mode not in ("RGB", "RGBA", "L"):  # e.g. palette ('P') — LANCZOS needs true-colour
            im = im.convert("RGBA")
        new_height = round(im.height * PRINT_TARGET_WIDTH_PX / im.width)
        small = im.resize((PRINT_TARGET_WIDTH_PX, new_height), Image.LANCZOS)
        truecolour = _encode_png(small)
        palette = _encode_png(small.convert("RGB").quantize(colors=256, dither=Image.Dither.NONE))
        dest.write_bytes(min(truecolour, palette, key=len))


def publish_figure(src: Path) -> Path:
    """Copy one rendered figure into ``Figure/``, downsized to 300 dpi on A4 (see _copy_figure_for_print).

    The single point where a PNG in ``outputs/thesis/figures`` becomes the copy the thesis
    compiles. ``figbase.save`` calls this on every save so a freshly regenerated figure can never
    leave a stale print copy behind; ``copy_figures`` / the full pipeline re-run it in bulk.
    """
    PROJECT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROJECT_FIGURE_DIR / src.name
    _copy_figure_for_print(src, dest)
    return dest


def copy_figures() -> list[Path]:
    return [publish_figure(png) for png in sorted(FIGURE_DIR.glob("*.png"))]


def generate_tables() -> list[Path]:
    """Render every result table to ``Table/<name>.tex`` from :func:`tables.build_all` specs."""
    from src.analysis import tables  # local import: pulls in pandas/aggregate only when needed

    PROJECT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame, caption in tables.build_all():
        dest = PROJECT_TABLE_DIR / f"{name}.tex"
        dest.write_text(
            latexfmt.dataframe_to_latex(frame, caption=caption, label=f"tab:{name}"),
            encoding="utf-8",
        )
        written.append(dest)
    return written


def make_all() -> list[Path]:
    return [*convert_chapters(), *copy_figures(), *generate_tables()]
