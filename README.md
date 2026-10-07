# From Ideal Points to Robot Joints — computational geometry and conformal geometric algebra

Seminar paper (Hochschule Darmstadt, B.Sc. Applied Mathematics) by Yana Osinchuk.
Everything in this archive is reproducible from source.

## Build

```bash
python3 code/generate_figures.py                 # all numbers, tables, figures (~1 min, seed 20260922)
python3 -m unittest discover -s code/tests -v    # 12 tests
latexmk -pdf main.tex                            # -> main.pdf
```
`FAST=1 python3 code/generate_figures.py` runs a reduced smoke version (not used in the paper).
On Overleaf: upload the folder, set the compiler to pdfLaTeX; the paper uses Latin Modern
automatically when `lmodern` is available. The submission date is the macro `\SubmissionDate`
near the top of `main.tex`.

## Contents

| Path | Content |
|------|---------|
| `main.tex`, `references.bib` | manuscript and bibliography |
| `figures/` | vector figures (PDF) + previews (PNG), h_da logo |
| `generated/results.tex`, `results.json` | every reported number (generated, do not edit) |
| `code/cga.py` | self-contained NumPy implementation of the conformal GA G(4,1) with CLUCalc counterparts |
| `code/geometry.py` | closed-form reference solutions of all six tasks (structured results) |
| `code/homework_cga.py` | line-by-line transcriptions of the CLUCalc homework, original and revised |
| `code/generate_figures.py` | all experiments and figures |
| `code/tests/` | unit tests; `stub/libcfcg` is a headless stand-in of the course library (testing only) |
| `source_tasks/` | original task archives, unchanged |
| `source_tasks_revised/` | revised CLUCalc (`clucalc/`) and Python (`python/`) programs |

## Main corrections with respect to the previous version

* Paper: empty numbers/tables (macros were not loaded), unresolved citations and cross-references,
  a foreign appendix and reference list (OHLC/DAX finance paper) removed, duplicated appendix,
  misplaced tables, inconsistent "five concerns" vs six-stage pipeline, date.
* Mathematics: Section 5.4 described a wrist on the line OT, but the homework actually constructs an
  isosceles trapezoid (Proposition 2); the construction has a representation singularity at d = 1,
  removed by a reflection E1 = -m E2 m^-1. Exact candidate count for the isosceles task
  (Proposition 1). Point-pair factorisation analysed (Lemma 1: weight -1, order depends on
  conventions). Tripod: stability margin shows the tripod tips over (-0.669) and only the outer
  foot stabilises it (+0.304).
* Programs: see Table 7 of the paper and the header comments of the revised programs.

The revised CLUCalc scripts use only constructs that occur in the original scripts, but they could
not be run in CLUCalc itself; their mathematics was verified in `code/cga.py`. The revised Python
scripts were run against the stub, not against the original `libcfcg`.
