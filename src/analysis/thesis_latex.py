r"""Convert the Markdown thesis artifacts into Overleaf-ready (pdfLaTeX) source.

Two products, both pure text post-processing (no GPU / no checkpoints):

* ``figure_snippets.tex`` — one ready-to-paste ``figure`` block per PNG in
  ``outputs/thesis/figures/``, captions reused from the chapter drafts.
* ``documents/thesis/latex/<chapter>.tex`` — each ``documents/thesis/*.md`` chapter
  rendered to LaTeX (sections, emphasis, lists, tables, figures, ``[@cite]``).

Assumed Overleaf preamble: ``\usepackage{graphicx,booktabs}``, a bibliography providing
the ``\cite`` keys, and figures uploaded under ``figures/`` (the graphics path the
snippets/chapters reference). Sectioning uses ``\chapter`` for ``#`` down to
``\subsubsection`` for ``####``; switch to starred forms if your front matter needs it.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.analysis import latexfmt
from src.visualization.figbase import FIGURE_DIR

CHAPTERS_DIR = Path("documents/thesis")
CHAPTERS_LATEX_DIR = CHAPTERS_DIR / "latex"

_FIGURE_EMBED_RE = re.compile(r"!\[(?P<cap>.*?)\]\((?P<path>[^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
_HEADING_CMDS = {1: "chapter", 2: "section", 3: "subsection", 4: "subsubsection"}


def _graphics_path(md_path: str) -> str:
    """Map a chapter's relative ``../../outputs/thesis/figures/foo.png`` to ``figures/foo.png``."""
    return f"figures/{Path(md_path).name}"


def _label_from_path(path: str) -> str:
    return "fig:" + Path(path).stem


# --- figure snippets ----------------------------------------------------------

def _caption_map() -> dict[str, str]:
    """stem -> caption, harvested from every chapter Markdown figure embed."""
    captions: dict[str, str] = {}
    for md in sorted(CHAPTERS_DIR.glob("*.md")):
        for m in _FIGURE_EMBED_RE.finditer(md.read_text(encoding="utf-8")):
            stem = Path(m.group("path")).stem
            captions.setdefault(stem, m.group("cap").strip())
    return captions


def _prettify(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def make_figure_snippets() -> Path:
    captions = _caption_map()
    blocks = [
        "% Auto-generated figure blocks for outputs/thesis/figures/*.png .",
        "% Requires \\usepackage{graphicx}; upload the PNGs under figures/ in Overleaf.",
        "",
    ]
    for png in sorted(FIGURE_DIR.glob("*.png")):
        caption = captions.get(png.stem) or _prettify(png.stem)
        blocks.append(
            latexfmt.figure_block(
                caption=caption,
                graphics_path=f"figures/{png.name}",
                label=_label_from_path(png.name),
            )
        )
    out = FIGURE_DIR / "figure_snippets.tex"
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


# --- chapter markdown -> latex ------------------------------------------------

def _strip_heading_number(text: str) -> str:
    text = re.sub(r"^Chapter\s+\d+\.?\s*", "", text)
    text = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text)
    return text


def _heading(level: int, text: str) -> str:
    cmd = _HEADING_CMDS.get(level, "paragraph")
    return f"\\{cmd}{{{latexfmt.inline_md_to_latex(_strip_heading_number(text))}}}"


def _table(rows: list[str]) -> str:
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


def convert_markdown(md_text: str, source_name: str = "") -> str:
    lines = md_text.splitlines()
    out: list[str] = []
    if source_name:
        out += [
            f"% Auto-generated from {source_name} by src.analysis.thesis_latex.",
            "% Needs \\usepackage{graphicx,booktabs} and your bibliography for \\cite.",
            "",
        ]
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        if not line.strip():
            out.append("")
            i += 1
            continue

        m = _HEADING_RE.match(line)
        if m:
            out.append(_heading(len(m.group(1)), m.group(2)))
            i += 1
            continue

        # blockquote -> LaTeX comments (these are TODO / note blocks)
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
                    graphics_path=_graphics_path(fm.group("path")),
                    label=_label_from_path(fm.group("path")),
                )
            )
            i += 1
            continue

        # table
        if line.lstrip().startswith("|") and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            block = []
            while i < n and lines[i].lstrip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block))
            continue

        # bullet list (with wrapped continuation lines)
        if _LIST_RE.match(line):
            items: list[str] = []
            while i < n:
                lm = _LIST_RE.match(lines[i])
                if lm:
                    items.append(lm.group(2).strip())
                    i += 1
                elif lines[i].strip() and lines[i].startswith(" "):
                    items[-1] += " " + lines[i].strip()  # continuation
                    i += 1
                else:
                    break
            out.append("\\begin{itemize}")
            out += [f"  \\item {latexfmt.inline_md_to_latex(it)}" for it in items]
            out.append("\\end{itemize}")
            continue

        # paragraph: gather until blank / next block
        para: list[str] = []
        while i < n and lines[i].strip() and not _HEADING_RE.match(lines[i]) \
                and not lines[i].lstrip().startswith(">") \
                and not lines[i].lstrip().startswith("|") \
                and not _LIST_RE.match(lines[i]) \
                and not _FIGURE_EMBED_RE.match(lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append(latexfmt.inline_md_to_latex(" ".join(para)))

    # collapse 3+ blank lines to one
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def convert_chapters() -> list[Path]:
    CHAPTERS_LATEX_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for md in sorted(CHAPTERS_DIR.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        tex = convert_markdown(md.read_text(encoding="utf-8"), source_name=f"documents/thesis/{md.name}")
        out = CHAPTERS_LATEX_DIR / f"{md.stem}.tex"
        out.write_text(tex, encoding="utf-8")
        written.append(out)
    return written


def make_all() -> list[Path]:
    return [make_figure_snippets(), *convert_chapters()]
