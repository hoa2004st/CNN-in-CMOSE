# Agent Instructions

## Before Starting Any Work
1. Run `./init.sh` to verify environment health
2. Read `progress.md` for context from last session
3. Check `git log --oneline -10` for recent changes

## Thesis Writing Instruction
1. Use folder structure of `documents/Thesis_Template` as example and guideline for folder `documents/thesis`.
2. Read files in folder `documents/Thesis_Template` to understand expected content of each chapter, section, ...

## Thesis .tex Formatting Rules (keep these on every edit)
- **One sentence per source line.** Write each sentence of prose on its own line of code (semantic line breaks); never hard-wrap a sentence across several short lines, and never merge a whole paragraph onto one line. Separate paragraphs with a blank line. Inside `itemize`/`enumerate`, each sentence of an item also gets its own line.
- **No source-tree paths in prose.** Do not mention repository directories, script paths, module names, commands, or manifest field names in the thesis text (e.g. `src/analysis/...`, `python -m ...`, `diagrams_mermaid.md`, `too_few_success_frames`). They bloat line length and are irrelevant to the reader. Describe what was done, not where the code lives.
- **Float ordering.** `placeins` is loaded in `main.tex`. Put `\FloatBarrier` immediately before every `\section` (chapters are already flushed by `\chapter`'s page break). Architecture diagrams use `[H]` (the `float` package) so they sit exactly where written and cannot reorder; result plots/tables stay `[ht]` and are bounded by the per-section `\FloatBarrier`.
- **Figure filenames must be LaTeX/Overleaf-safe.** No parentheses or spaces in `Figure/*.png` names (use underscores). Every `\includegraphics{...}` must resolve to a file that exists in `Figure/`.
- **Architecture charts belong to their own architecture only.** Building-block diagrams go in the Ch.2 foundational subsection for that family (TCN block → TCN subsection, LSTM cell → LSTM subsection). Full baseline model diagrams go in the Ch.3 Baseline Models section; the hybrid overview and the per-group encoder options go in the Ch.3 Hybrid section. Never place one architecture's chart inside another architecture's (sub)section.
- **Cover pages** (`Cover.tex`, `Cover2.tex`) must mirror `documents/Thesis_Template/Cover.tex` / `Cover2.tex` exactly in layout; only fill in author, email, program, supervisor, department/school, title, and date. Keep `\documentclass[main.tex]{subfiles}`.
- **Personal info (single source of truth):** author Phan Minh Hòa; email `hoa.pm225495@sis.hust.edu.vn`; program "Data Science and Artificial Intelligence"; supervisor Trịnh Văn Chiến; department "Faculty of Computer Science"; school "School of Information and Communications Technology". `\AUTHOR` lives in `main.tex`; the supervisor name also appears in `Chapter/0_2_Acknowledgment.tex`.
- **References** in `reference.bib` are verified from the PDFs in `documents/references/` — keep them complete (no `TODO`/`author = {TODO}` placeholders); confirm new entries against the actual PDF or an online search before citing.

## Rules
- Title is strictly `Student's engagement detection in online classes`
- Do not generate not requred files in different formats and file types.
- General use figures are stored in outputs, while thesis use figures and tables are stored in documents/thesis.
- When change/delete/move any files, reconcile it in other files or documents that mentioned them.
- Length of chaper must not differ too much (recommended 8-11 pages long).
- Number of chapter/section, name of chapter/section can be changed.
- Architecture must be drawn as diagram (can ask user to draw them manually).
- Losses, Metrics, Architectures, Pipeline must be clearly and fully described.
- Each result must come to a clear conclusion.
- Update `progress.md` after every session
- Commit only when the project is in a clean, resumable state


## Verification Checklist
- [ ] All tests pass
- [ ] Linter passes
- [ ] Type-check passes
- [ ] Feature works as specified
