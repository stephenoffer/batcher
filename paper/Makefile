# Build the Batcher paper.

PAPER   := main
CHECKER := $(HOME)/.claude/plugins/cache/research-paper/research-paper/0.3.0/skills/research-paper/scripts

.PHONY: all check pages clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib tikz-paper-styles.tex
	pdflatex -interaction=nonstopmode $(PAPER)
	bibtex $(PAPER) || true
	pdflatex -interaction=nonstopmode $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER)

# Static gates; these run with no LaTeX installed.
check:
	python3 $(CHECKER)/check_paper.py $(PAPER).tex
	python3 $(CHECKER)/check_references.py refs.bib $(PAPER).tex

pages: $(PAPER).pdf
	@pdftoppm -png -r 20 $(PAPER).pdf /tmp/_pg && ls /tmp/_pg*.png | wc -l && rm -f /tmp/_pg*.png

clean:
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).log $(PAPER).bbl $(PAPER).blg \
	      $(PAPER).out $(PAPER).toc
