# Agent Instructions

## Before Starting Any Work
1. Run `./init.sh` to verify environment health
2. Read `progress.md` for context from last session
3. Check `git log --oneline -10` for recent changes

## Rules
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
