# Minimal LaTeX build for ICML 2026 workshop submission.
LATEX  := pdflatex
BIBTEX := bibtex
MAIN   := main
TEXCOUNT := texcount

.PHONY: pdf clean wordcount blind

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib \
             icml2026.sty icml2026.bst algorithm.sty algorithmic.sty fancyhdr.sty
	$(LATEX)  -interaction=nonstopmode $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX)  -interaction=nonstopmode $(MAIN).tex
	$(LATEX)  -interaction=nonstopmode $(MAIN).tex

# Verifies blind-mode style flag is active before building (paranoid).
blind:
	@grep -E '^\\usepackage(\[[^]]*\])?\{icml2026\}' $(MAIN).tex \
	  | grep -qv '\[accepted\]' || \
	  (echo "ERROR: \\usepackage[accepted]{icml2026} present; remove for blind." && exit 1)
	@$(MAKE) pdf

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk \
	      $(MAIN).synctex.gz $(MAIN).pdf

wordcount:
	$(TEXCOUNT) -inc -sum -1 $(MAIN).tex
