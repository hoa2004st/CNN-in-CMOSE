# Thesis (HUST Thesis_Template layout)

This folder is a **compilable LaTeX thesis project** following `documents/Thesis_Template`.
`main.tex` is the driver; each chapter is a `subfiles` document under `Chapter/`, figures live
in `Figure/`, and the result tables in `Table/`.

## Layout

| Path | Role | Authored by |
|---|---|---|
| `main.tex` | Driver: preamble, front matter wiring, `\chapter` + `\subfile` per chapter, bibliography | hand (static) |
| `Cover.tex`, `Cover2.tex` | Title / inner title pages — **fill in author, supervisor, program** | hand (static) |
| `glossary.tex` | List of abbreviations (QWK, TCN, I3D, …) | hand (static) |
| `lstlisting.tex` | Code-listing styles | hand (static) |
| `reference.bib` | Bibliography (copied from `documents/references.bib`) | copied |
| `Chapter/0_2_Acknowledgment.tex`, `0_3_Abstract.tex` | Front matter | **generated** |
| `Chapter/1_Introduction.tex` … `5_Conclusions.tex` | Body chapters | **generated** |
| `Figure/*.png` | Analysis figures | **generated** (copied) |
| `Table/T*.tex` | Result tables | **generated** (copied) |
| `00_front_matter.md` … `05_conclusion.md` | Markdown **source** for the chapters | hand |

Chapter mapping (the template's optional *Theoretical Analysis* chapter is omitted):
1 Introduction · 2 Literature Review · 3 Methodology · **4 Numerical Results** · 5 Conclusions.

## Regenerating

The `.md` files are the source of truth for chapter prose. After editing them (or re-running the
analysis), regenerate the LaTeX project with:

```
python -m src.analysis.make_thesis_artifacts
```

This rebuilds the figures/tables in `outputs/thesis/`, then `src.analysis.thesis_latex`:
- renders each `*.md` chapter to a `subfiles` document in `Chapter/` (no top-level `\chapter` —
  the driver supplies it; `##`→`\section`, `###`→`\subsection`, …),
- copies `outputs/thesis/figures/*.png` → `Figure/` and `outputs/thesis/tables/T*.tex` → `Table/`,
- `\input`s each result table into Chapter 4 at the point it is first referenced (`T1`…`T7`).

The static scaffolding (`main.tex`, covers, `glossary.tex`, `lstlisting.tex`, `reference.bib`) is
**not** regenerated — edit it directly.

## Compiling

pdfLaTeX + biber/bibtex (the template's engine). Locally or on Overleaf:

```
pdflatex main      # or latexmk -pdf main
bibtex   main
pdflatex main
pdflatex main
```

Figures resolve through `\graphicspath{{Figure/}{../Figure/}}`, so `\includegraphics{name.png}`
finds them. Tables need `booktabs` (already in the preamble). On Overleaf, upload the whole
`documents/thesis/` folder and set `main.tex` as the root document.

**Citations** use keys from `reference.bib` (e.g. `\cite{cmose}`, `\cite{tcn}`); entries marked
`TODO` in the source `.bib` must still be completed from the PDFs in `documents/references/`.
Conversion logic lives in `src/analysis/latexfmt.py` (escaping/tables/figures) and
`src/analysis/thesis_latex.py` (chapters + project assembly).
