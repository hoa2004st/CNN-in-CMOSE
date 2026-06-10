# Thesis (HUST Thesis_Template layout)

This folder is a **compilable LaTeX thesis project** following `documents/Thesis_Template`.
`main.tex` is the driver; each chapter is a `subfiles` document under `Chapter/`, figures live
in `Figure/`, and the result tables in `Table/`.

Title (fixed): **Student's engagement detection in online classes**.

## Layout

| Path | Role | Authored by |
|---|---|---|
| `main.tex` | Driver: preamble, front-matter wiring, `\chapter` + `\subfile` per chapter, bibliography | hand |
| `Cover.tex`, `Cover2.tex` | Title / inner title pages (layout mirrors `Thesis_Template`; author/supervisor/program filled in) | hand |
| `glossary.tex` | List of abbreviations (QWK, TCN, I3D, …) | hand |
| `lstlisting.tex` | Code-listing styles | hand |
| `reference.bib` | Bibliography (entries marked `TODO` still need verifying from `documents/references/`) | hand |
| `Chapter/0_2_Acknowledgment.tex`, `0_3_Abstract.tex` | Front matter | **hand-authored** |
| `Chapter/1_Introduction.tex` … `6_Conclusions.tex` | Body chapters | **hand-authored** |
| `Figure/*.png` | Analysis figures (copied from `outputs/thesis/figures`) | **generated** (copied) |
| `Table/T*.tex` | Result tables (rendered from the analysis) | **generated** |

Chapter mapping (the template's optional *Theoretical Analysis* chapter is omitted):
1 Introduction · 2 Literature Review · 3 Methodology · **4 In-Domain Model Development** · **5 Generalization** · 6 Conclusions.

> **Note.** Earlier versions generated the chapter prose from Markdown drafts
> (`00_front_matter.md` … `05_conclusion.md`). Those drafts have been **removed**; the
> `Chapter/*.tex` files are now the source of truth and are **edited by hand**. The architecture
> diagrams are now **real `\includegraphics`** (no longer placeholders): the per-family building
> blocks (`openface_tcn_block.png`, `openface_lstm_cell.png`) sit in the matching Chapter 2
> subsection; the five baseline diagrams (`openface_mlp/tcn/lstm/transformer.png`, `i3d_mlp.png`)
> in the Chapter 3 Baseline Models section; and the hybrid overview (`hybrid.png`) plus the three
> per-group encoder options (`hybrid_tcn/transformer/lstm_encoder.png`) in the Chapter 3 Hybrid
> section. They were drawn by hand from the Mermaid source in `diagrams_mermaid.md` and exported to
> `Figure/`; keep their filenames parenthesis/space-free. See `AGENTS.md` for the full thesis
> `.tex` formatting rules.

## Regenerating figures and tables

Only the analysis artifacts (`Figure/*.png`, `Table/T*.tex`) are generated; the chapter prose is
not. After re-running the analysis, refresh them with:

```
python -m src.analysis.make_thesis_artifacts
```

This rebuilds the figures/tables in `outputs/thesis/`, then copies
`outputs/thesis/figures/*.png` → `Figure/` (downsized to 300 dpi for A4 print) and renders the
result tables → `Table/T*.tex` (labels `tab:T1_dataset_stats` … used by `\input{Table/...}` in
Chapter 4). Because the Markdown chapter drafts no longer exist, the chapter-rendering step is a
no-op and **will not overwrite the hand-written `Chapter/*.tex`**.

The static scaffolding (`main.tex`, covers, `glossary.tex`, `lstlisting.tex`, `reference.bib`) and
the chapter prose are **not** regenerated — edit them directly.

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
`TODO` in the `.bib` must still be completed from the PDFs in `documents/references/`.
Figure/table generation logic lives in `src/analysis/latexfmt.py` (escaping/tables/figures) and
`src/analysis/thesis_latex.py` (figure copy + table render; chapter rendering is legacy/inert).
