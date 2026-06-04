# Progress Log

## Session 1 - 2026-Jun-04
- Complete: Rerun training and create raw prediction of test set of all models
- In progress: None
- Block: None
- Next session should: Make decision of choosing visualization methods. Getting ready for result analyse and thesis writing

## Session 2 - 2026-Jun-04
- Decision: thesis primary contribution = semantic-group hybrid model; spectral/frequency idea dropped (future work only)
- Complete: greenfield analysis/viz pipeline (src/analysis/aggregate.py, tables.py, make_thesis_artifacts.py; src/visualization/figbase.py + figures_*.py) -> 12 figures + 5 tables in outputs/thesis/
- Complete: documents/results_interpretation.md (per-figure takeaways), documents/references.bib (canonical entries + TODO placeholders for local PDFs)
- Complete: thesis scaffold documents/thesis/ (front matter + Ch1-5) following ThesisGuideline.txt
- Block: no PDF text tool in env -> 6 references.bib entries left as TODO to verify from documents/references/
- Revision (same session): elevated the self-collected PRIVATE set as a core contribution; added figures_private.py (private_by_source, indomain_cmose_vs_daisee, private_confusion_combined), tables T6/T7; reframed Ch1 contributions, abstract, Ch3 (private collection TODO), Ch4 (new 4.6 private section + DaiSEE-in-domain justification in 4.3), Ch5. Now 15 figures + 7 tables.
- Key revised finding: hybrid+combined best on private QWK 0.365 vs 0.285 base (+0.080, wider than in-domain); DaiSEE in-domain weak (QWK 0.139) justifies CMOSE-anchored ablation.
- Next session should: fill in TODO citations + Figure 3.1 diagram + private-set collection methodology paragraph; expand prose; tighten to page targets; (optional) regenerate artifacts with `python -m src.analysis.make_thesis_artifacts`
