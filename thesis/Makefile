MAIN    := DoAn
LATEXMK := latexmk

.PHONY: all pdf watch clean distclean view

all: pdf

pdf:
	$(LATEXMK) $(MAIN).tex

# Build liên tục, tự động compile lại khi file thay đổi
watch:
	$(LATEXMK) -pvc -view=none $(MAIN).tex

# Mở file PDF (macOS)
view: pdf
	open $(MAIN).pdf

clean:
	$(LATEXMK) -c

distclean:
	$(LATEXMK) -C
	rm -f $(MAIN).pdf
