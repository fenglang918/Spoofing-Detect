# LaTeX Project

This directory contains the LaTeX source for the essay.

## Main File

- `essay.tex`: The main LaTeX source file.

## Compilation

To compile the project and generate the PDF (`essay.pdf`), you can use a LaTeX distribution (like TeX Live, MiKTeX). The recommended compiler is `xelatex`, which provides better support for Unicode and modern fonts.

Run the following command in this directory:

```bash
xelatex essay.tex
```

To ensure all cross-references are correctly updated, it is recommended to run the command multiple times (e.g., 2-3 times).

```bash
xelatex essay.tex
xelatex essay.tex
```

After successful compilation, the output file `essay.pdf` will be generated or updated.

## Included Files

- `*.png`: Image files used as figures in the document.
- `封面.pdf`: The cover page, included as a PDF.
