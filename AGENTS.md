# Agent Instructions

## Before Starting Any Work
1. Run `./init.sh` to verify environment health
2. Read `progress.md` for context from last session
3. Check `git log --oneline -10` for recent changes

## Thesis Writing Instruction
1. Use folder structure of `documents/Thesis_Template` as example and guideline for folder `documents/thesis`.
2. Read files in folder `documents/Thesis_Template` to understand expected content of each chapter, section, ...

## Rules
- Do not generate not requred files in different formats and file types.
- General use figures are stored in outputs, while thesis use figures and tables are stored in documents/thesis.
- When change/delete/move any files, reconcile it in other files or documents that mentioned them.
- Length of chaper must not differ too much (recommended 8-11 pages long).
- Number of chapter/section, name of chapter/section can be changed.
- Architecture must be drawn as diagram (can ask user to draw them manually).
- Losses, Metrics, Architectures, Pipeline must be clearly and fully described.
- Update `progress.md` after every session
- Commit only when the project is in a clean, resumable state


## Verification Checklist
- [ ] All tests pass
- [ ] Linter passes
- [ ] Type-check passes
- [ ] Feature works as specified
