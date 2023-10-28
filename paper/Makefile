FIRST_DIFF_TAG = initial-submission

pdf:
	pdflatex ollie-optimization.tex
	bibtex ollie-optimization.aux
	pdflatex ollie-optimization.tex
	pdflatex ollie-optimization.tex
trackchanges:
	git checkout $(FIRST_DIFF_TAG)
	cp ollie-optimization.tex $(FIRST_DIFF_TAG).tex
	git checkout main
	latexdiff $(FIRST_DIFF_TAG).tex ollie-optimization.tex > diff-main_$(FIRST_DIFF_TAG).tex
	rm $(FIRST_DIFF_TAG).tex
	pdflatex -interaction nonstopmode diff-main_$(FIRST_DIFF_TAG).tex
	bibtex diff-main_$(FIRST_DIFF_TAG).aux
	pdflatex -interaction nonstopmode diff-main_$(FIRST_DIFF_TAG).tex
	pdflatex -interaction nonstopmode diff-main_$(FIRST_DIFF_TAG).tex
clean:
	rm -rf *.blg *.out *.fdb_latexmk *.aux *.fls *.synctex.gz *.bbl  *.log
