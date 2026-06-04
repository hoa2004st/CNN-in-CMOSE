# Thesis draft

Chapter-per-file scaffold following `documents/ThesisGuideline.txt`. Each file already
contains substantive prose grounded in the actual results, the figures/tables to embed, and
`> TODO:` markers where you should expand with your own voice or add detail.

| File | Chapter | Target pages |
|---|---|---|
| `00_front_matter.md` | Acknowledgement + Abstract | ~1.5 |
| `01_introduction.md` | Ch.1 Introduction | ~5 |
| `02_literature_review.md` | Ch.2 Literature Review | ~7 |
| `03_methodology.md` | Ch.3 Methodology | ~9 |
| `04_results.md` | Ch.4 Numerical Result | ~10 |
| `05_conclusion.md` | Ch.5 Conclusion | ~3 |

**Figures/tables** live in `outputs/thesis/{figures,tables}` and are regenerated with
`python -m src.analysis.make_thesis_artifacts`. Figure paths in the chapters are relative
(`../../outputs/thesis/figures/...`). The per-figure interpretation is in
`documents/results_interpretation.md` — Chapter 4 prose is drawn from there.

**Citations** use keys from `documents/references.bib` (e.g. `[@cmose]`, `[@tcn]`). Entries
marked `TODO` must be completed from the PDFs in `documents/references/`.

**Overleaf-ready LaTeX (pdfLaTeX).** `python -m src.analysis.make_thesis_artifacts` also emits
paste-ready LaTeX with all underscores escaped and Unicode (`κ Δ × → ∪ …`) mapped to commands:
- `outputs/thesis/tables/*.tex` — the result tables (booktabs, centered, `\caption`+`\label`).
- `outputs/thesis/figures/figure_snippets.tex` — a `\begin{figure}` block per PNG, captions
  reused from the chapter drafts.
- `documents/thesis/latex/*.tex` — each chapter converted from Markdown (`\section`, emphasis,
  lists, tables, figures, `[@key]`→`\cite{key}`).

Overleaf preamble must provide `\usepackage{graphicx,booktabs}` and your bibliography; upload
the figure PNGs under a `figures/` folder (the path the snippets reference). Paste a table/chapter
directly, or `\input` the `.tex` files. Conversion logic lives in `src/analysis/latexfmt.py`
(escaping/tables/figures) and `src/analysis/thesis_latex.py` (chapters + snippets).

**Compiling to PDF/Word** (optional): with Pandoc + a LaTeX engine,
`pandoc 0*.md --citeproc --bibliography=../references.bib -o thesis.pdf` — but check your
program's required template/format first; this scaffold is content, not final typesetting.
