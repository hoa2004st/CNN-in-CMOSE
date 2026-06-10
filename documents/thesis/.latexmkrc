# latexmk configuration for the thesis (MiKTeX + Strawberry Perl)
# Build chain: pdflatex -> bibtex -> makeglossaries -> pdflatex x2

$pdf_mode = 1;                       # produce PDF via pdflatex
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;                     # always run bibtex when .bib/biblatex present

# --- glossaries / acronyms support -----------------------------------------
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');
sub run_makeglossaries {
    system("makeglossaries \"$_[0]\"");
}

# clean up the extra aux files these tools generate (latexmk -c)
push @generated_exts, 'glo', 'gls', 'glg', 'acn', 'acr', 'alg', 'ist', 'xdy', 'bbl', 'run.xml';
