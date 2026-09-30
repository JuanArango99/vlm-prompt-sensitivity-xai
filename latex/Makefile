# Makefile for compiling the LaTeX documents

.PHONY: all paper clean

all: paper

paper:
	latexmk -pdf paper.tex

# Keep running and automatically recompile paper on save
watch:
	latexmk -pdf -pvc paper.tex

clean:
	latexmk -C paper.tex
	rm -f *.bbl *.run.xml *.synctex.gz
