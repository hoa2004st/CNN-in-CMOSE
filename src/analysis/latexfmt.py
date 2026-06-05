"""Shared helpers to render thesis artifacts as Overleaf-ready (pdfLaTeX) source.

Everything here targets the default Overleaf engine (pdfLaTeX), so it:

* escapes LaTeX-special characters (``_ & % # $ { } ~ ^ \\``) that otherwise abort
  compilation — model names like ``i3d_mlp`` are the usual offenders;
* rewrites Unicode that pdfLaTeX cannot typeset (``κ Δ × → ∪ ² ⁵ ≈ – —`` …) into the
  equivalent LaTeX command / math;
* builds ``booktabs`` tables and ``figure`` blocks ready to paste or ``\\input``.

Required packages in the Overleaf preamble: ``booktabs`` (tables) and ``graphicx``
(figures). Nothing else.
"""

from __future__ import annotations

import re

import pandas as pd

# --- character-level escaping -------------------------------------------------

# Order is irrelevant because we substitute every special in a single regex pass,
# so a replacement that introduces e.g. ``{}`` is never re-escaped.
_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_SPECIALS_RE = re.compile("|".join(re.escape(k) for k in _SPECIALS))

# Unicode that pdfLaTeX (OT1/T1) cannot render -> LaTeX equivalent. Applied AFTER
# escaping so the backslashes/braces/dollars we introduce here are left intact.
_UNICODE = {
    "κ": r"$\kappa$",
    "Δ": r"$\Delta$",
    "δ": r"$\delta$",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "≈": r"$\approx$",
    "≃": r"$\simeq$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "≠": r"$\neq$",
    "∪": r"$\cup$",
    "∩": r"$\cap$",
    "∈": r"$\in$",
    "∼": r"$\sim$",
    "−": "-",        # U+2212 minus -> hyphen
    "·": r"$\cdot$",
    "°": r"$^\circ$",
    "±": r"$\pm$",
    "∞": r"$\infty$",
    "√": r"$\surd$",
    "²": r"$^2$",
    "³": r"$^3$",
    "⁴": r"$^4$",
    "⁵": r"$^5$",
    "½": r"$\tfrac{1}{2}$",
    "—": "---",      # em dash
    "–": "--",       # en dash
    "‑": "-",        # non-breaking hyphen
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "…": r"\ldots{}",
    " ": "~",   # non-breaking space
    "​": "",    # zero-width space
}
_UNICODE_RE = re.compile("|".join(re.escape(k) for k in _UNICODE))


def escape(text) -> str:
    """Escape LaTeX specials then map unrenderable Unicode. Safe for a table cell."""
    s = "" if text is None else str(text)
    s = _SPECIALS_RE.sub(lambda m: _SPECIALS[m.group()], s)
    s = _UNICODE_RE.sub(lambda m: _UNICODE[m.group()], s)
    return s


def unicode_only(text) -> str:
    """Map Unicode to LaTeX without escaping specials (for already-LaTeX strings)."""
    return _UNICODE_RE.sub(lambda m: _UNICODE[m.group()], "" if text is None else str(text))


# --- tables -------------------------------------------------------------------

def _fmt_value(v, float_fmt: str) -> str:
    if isinstance(v, float):
        if v == 0:  # collapse -0.0
            v = 0.0
        return float_fmt % v
    if isinstance(v, int):
        return str(v)
    return str(v)


def dataframe_to_latex(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    float_fmt: str = "%.3f",
    placement: str = "ht",
) -> str:
    """Render ``frame`` as a centered booktabs table that compiles under pdfLaTeX.

    Numeric columns are right-aligned (``r``), text columns left-aligned (``l``).
    All headers and cells are escaped; the caption only needs Unicode mapping.
    """
    cols = list(frame.columns)
    align = "".join(
        "r" if pd.api.types.is_numeric_dtype(frame[c]) else "l" for c in cols
    )
    header = " & ".join(escape(c) for c in cols) + r" \\"
    body = [
        " & ".join(escape(_fmt_value(v, float_fmt)) for v in row) + r" \\"
        for row in frame.itertuples(index=False)
    ]
    lines = [
        f"\\begin{{table}}[{placement}]",
        "\\centering",
        f"\\caption{{{escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --- figures ------------------------------------------------------------------

def figure_block(
    *,
    caption: str,
    graphics_path: str,
    label: str,
    width: str = r"0.9\linewidth",
    placement: str = "ht",
) -> str:
    """A ready-to-paste ``figure`` environment (needs ``graphicx``)."""
    return "\n".join(
        [
            f"\\begin{{figure}}[{placement}]",
            "\\centering",
            f"\\includegraphics[width={width}]{{{graphics_path}}}",
            f"\\caption{{{inline_md_to_latex(caption)}}}",
            f"\\label{{{label}}}",
            "\\end{figure}",
        ]
    ) + "\n"


# --- inline markdown -> latex -------------------------------------------------

_CODE_RE = re.compile(r"`([^`]+)`")
_CITE_RE = re.compile(r"\[@([^\]]+)\]")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\*)")
_PLACEHOLDER = "\x00{}\x00"


def inline_md_to_latex(text: str) -> str:
    """Convert one line/paragraph of Markdown inline syntax to LaTeX.

    Handles ``code``, ``[@cite]``, ``**bold**`` and ``*italic*`` while keeping
    LaTeX escaping correct: code spans and citations are protected before the
    surrounding prose is escaped, then restored.
    """
    protected: list[str] = []

    def _stash(latex: str) -> str:
        protected.append(latex)
        return _PLACEHOLDER.format(len(protected) - 1)

    # 1. inline code -> \texttt{...} (escape its contents, keep verbatim look)
    def _code(m: re.Match) -> str:
        return _stash(r"\texttt{" + escape(m.group(1)) + "}")

    text = _CODE_RE.sub(_code, text)

    # 2. citations [@a; @b] -> \cite{a,b}
    def _cite(m: re.Match) -> str:
        keys = [k.strip().lstrip("@").strip() for k in m.group(1).split(";")]
        return _stash(r"\cite{" + ",".join(k for k in keys if k) + "}")

    text = _CITE_RE.sub(_cite, text)

    # 3. escape the remaining prose
    text = escape(text)

    # 4. emphasis (operates on escaped text; * and ** survive escaping)
    text = _BOLD_RE.sub(lambda m: r"\textbf{" + m.group(1) + "}", text)
    text = _ITALIC_RE.sub(lambda m: r"\emph{" + m.group(1) + "}", text)

    # 5. restore protected LaTeX
    def _restore(m: re.Match) -> str:
        return protected[int(m.group(1))]

    return re.sub(r"\x00(\d+)\x00", _restore, text)
