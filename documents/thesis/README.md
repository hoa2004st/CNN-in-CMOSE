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

**Compiling to PDF/Word** (optional): with Pandoc + a LaTeX engine,
`pandoc 0*.md --citeproc --bibliography=../references.bib -o thesis.pdf` — but check your
program's required template/format first; this scaffold is content, not final typesetting.
