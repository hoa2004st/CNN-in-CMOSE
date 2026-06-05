# Agent Instructions

## Before Starting Any Work
1. Run `./init.sh` to verify environment health
2. Read `progress.md` for context from last session
3. Check `git log --oneline -10` for recent changes
4. In thesis writing task, use folder structure of `documents/Thesis_Template` as example and guideline for folder `documents/thesis`.
5. In thesis writing task, read files in folder `documents/Thesis_Template` to understand expected content of each chapter, section, ...

## Rules
- Do not generate not requred files in different formats and file types.
- General use figures are stored in outputs, while thesis use figures and tables are stored in documents/thesis.
- When change/delete/move any files, reconcile it in other files or documents that mentioned them.
- Work on exactly ONE feature at a time
- Never declare "done" without passing tests
- Run the full test suite before committing
- Update `progress.md` after every session
- Commit only when the project is in a clean, resumable state


## Verification Checklist
- [ ] All tests pass
- [ ] Linter passes
- [ ] Type-check passes
- [ ] Feature works as specified
