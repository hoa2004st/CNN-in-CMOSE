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

## Fact-Check Protocol (Thesis Finalization Phase)

We are in the finalization phase: every number, equation, citation, architecture detail, and
diagram in `documents/thesis/` must be traceable to a source of truth and independently
checkable. **No number is ever stated from memory or "by eye".** When asked to verify, or
when editing any factual claim, follow this protocol.

### Core principle — every number has a provenance
Each numeric claim maps to exactly one of:
1. **A cell in a result CSV / JSON** (look it up directly), or
2. **A deterministic recompute** from those CSVs via a documented command, or
3. **A source PDF** (`documents/references/`) for any number attributed to prior work, or
4. **The code** (`src/`) for any architecture / hyperparameter / equation claim.

The living record of these mappings is **`documents/thesis/FACTCHECK.md`** (the provenance
ledger): one row per claim — `claim | location in .tex | source-of-truth file | exact lookup
command | expected value | status`. Keep it in sync whenever a number is added or changed.

### Source-of-truth map (single source per number type)
| Number type | Source of truth | Key columns / keys |
| --- | --- | --- |
| Dataset totals & class % (T1, Ch.1/3) | `outputs/dataset_analysis/class_distribution_overall.csv`, `class_distribution_by_split.csv` | `dataset,split,count,proportion` |
| Private set counts/% (366, 58.2%, 2.7%, …) | `outputs/dataset_analysis/private/dataset_summary.json` | `total_clips`, `class_distribution.*.count/proportion` |
| Baseline metrics (QWK 0.537, macro-acc, …) | `outputs/model_assessment/naive/full_matrix.csv` | `train_group,test_set,model,loss` → `quadratic_weighted_kappa,macro_accuracy,macro_mae,accuracy,mae,cohen_kappa` |
| Hybrid metrics (QWK 0.605/0.553/0.379, 243 cfg) | `outputs/model_assessment/hybrid/hybrid_matrix.csv` | same + `arch_key,model_type` |
| Prediction-level (oracle 97.6%, pairwise κ, exclusive-correct 8.6%, paired I3D Δ, per-class F1, confusion) | `*_predictions.csv` next to each matrix | `clip_id,true_id,predicted_id,is_correct` — recompute, do not eyeball |
| Table numbers T1–T13 | `outputs/thesis/tables/T*.md` (generated) → `documents/thesis/Table/T*.tex` | committed generated form |
| Citations / prior-work numbers | `documents/references/*.pdf` | read the PDF page |
| Architecture / hyperparameters / loss & metric equations | `src/models/models.py`, `src/training/*`, `src/evaluation/metrics.py` | read the code |

**Model-name tokens** (thesis prose → CSV `model`): `openface_tcn`→`tcn`, `openface_lstm`→`lstm`,
`openface_transformer`→`transformer`, `openface_mlp`→`openface_mlp`, `i3d_mlp`→`i3d_mlp`.
`loss` ∈ {`ce`,`weighted_ce`,`ordinal`}; `test_set` ∈ {`cmose_test`,`daisee_test`,`private`};
`train_group` ∈ {`cmose`,`daisee`,`combined`}.

### Standard lookup recipe (PowerShell — the user's shell, so they can rerun verbatim)
```powershell
# Any cell of the baseline / hybrid matrix:
Import-Csv outputs/model_assessment/naive/full_matrix.csv |
  Where-Object { $_.train_group -eq 'cmose' -and $_.test_set -eq 'cmose_test' `
                 -and $_.model -eq 'tcn' -and $_.loss -eq 'ce' } |
  Select-Object model,loss,quadratic_weighted_kappa,macro_accuracy,macro_mae
# -> quadratic_weighted_kappa 0.5371745...  ==  thesis "QWK 0.537"
```
Round the CSV value to the thesis's stated precision and compare; flag any mismatch beyond
the last shown digit. For derived numbers, the deterministic recompute is
`python -m src.analysis.make_thesis_artifacts` (pure post-processing of the CSVs, no GPU),
which regenerates every figure and `outputs/thesis/tables/T*.md`; a thesis number is correct
iff it equals the regenerated artifact.

### Verification workflow (what I do on every fact-check pass)
1. Locate the claim in the `.tex` (record `file:line`).
2. Classify it (number / equation / citation / architecture / diagram).
3. Pull the source of truth via the recipe above; record the exact lookup in `FACTCHECK.md`.
4. Compare at the stated rounding; mark `OK` / `MISMATCH (found X, says Y)` / `UNVERIFIABLE`.
5. For citations, open the PDF and confirm both the metadata **and** the attributed claim.
6. For equations, confirm the written form matches both the cited paper and `src/` implementation.
7. Never "fix" a thesis number to match an old CSV without confirming the CSV is current
   (`git log` the `outputs/` file); if results were retrained, the .tex follows the CSV.

### Rules of evidence
- A number with no entry in the source-of-truth map is **not citable** until one is added.
- If two chapters quote the same number, they must agree to the digit; cross-check duplicates.
- Citations: keep `reference.bib` free of `TODO`/placeholder fields; every entry verified from
  its PDF in `documents/references/` (see also the References rule above).
- When a check fails, surface it — do not silently overwrite the thesis or the CSV.

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
