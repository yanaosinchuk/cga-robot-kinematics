# CGA Robot Kinematics — From Ideal Points to Robot Joints

[![tests](https://github.com/yanaosinchuk/cga-robot-kinematics/actions/workflows/tests.yml/badge.svg)](https://github.com/yanaosinchuk/cga-robot-kinematics/actions/workflows/tests.yml)

[Technical report](paper/from_ideal_points_to_robot_joints.pdf) · [LaTeX source](paper/main.tex) · [Numerical results](results/results.json) · [MIT License](LICENSE) · [Citation](CITATION.cff)

Computational geometry for robotics with **conformal geometric algebra (CGA)**, projective geometry, and independent numerical validation.

The project turns inverse-kinematics and geometric-construction problems into intersections of geometric primitives, keeps all solution branches explicit, and separates geometric construction from application-specific branch selection.

## Robotics Problem

Inverse kinematics is often implemented as a collection of coordinate formulas. That works, but it can hide three things that matter in robotics: **multiple valid branches**, **singular configurations**, and **the geometric meaning of failure**.

This repository studies those issues through three robotics-oriented constructions:

- **Two-link inverse kinematics:** two link spheres and a helper plane produce the two elbow branches.
- **Three-link inverse kinematics:** an isosceles-trapezoid construction gives a three-link chain and exposes a representation singularity in the original formulation near target distance $d=1$.
- **Tripod support geometry:** three spheres determine the apex; a fourth support is selected from two candidates using a stability criterion.

![Two-link and three-link inverse-kinematics constructions](figures/kinematic_constructions.png)

The goal is not to replace standard matrix-based robotics methods. It is to show how a geometric representation can make branches, intersections, and singularities explicit and testable.

## Geometric Formulation

The common pattern is:

~~~text
encode -> construct -> intersect -> factor -> select -> validate
~~~

A geometric solver first constructs the **full solution set**. Only afterwards does a policy select the branch appropriate for the application. This prevents statements such as “upper elbow” or “outer support” from being confused with the underlying mathematics.

Projective geometry is used for the planar incidence layer:

- circumcentre as the meet of two perpendicular bisectors,
- perpendicular construction through an ideal point,
- all isosceles-triangle apices on a prescribed parallel line.

CGA is used for the 3D distance-geometry layer:

- spheres, planes, circles, and point pairs share one algebraic representation,
- meets compute intersections,
- point-pair factorisation exposes multiple solutions,
- reflection gives a robust reformulation of the three-link construction.

The tripod example also separates geometric feasibility from the engineering decision. In the numerical example, the stability margin is evaluated **assuming the load acts at the apex**; the outer fourth-support candidate changes the margin from approximately $-0.669$ to $+0.304$.

![Three-sphere meet and tripod support geometry](figures/tripod_construction.png)

## CGA Implementation

[`src/cga.py`](src/cga.py) is a transparent NumPy implementation of the conformal geometric algebra $G(4,1)$ used by the CLUCalc N3 model.

A multivector is stored with $2^5=32$ basis-blade coefficients. The implementation includes:

- geometric, outer, and inner products,
- left contraction,
- reverse and blade/versor inverse,
- duality,
- conformal point embedding,
- OPNS/IPNS spheres,
- two-object and three-object meets,
- conformal point normalisation,
- point-pair classification and extraction.

[`src/cga_constructions.py`](src/cga_constructions.py) contains executable CGA transcriptions of the robotics constructions, including both the audited original formulations and the revised formulations used for the robust solutions.

The portfolio-facing CLUCalc versions are in [`cga_kinematics/`](cga_kinematics/).

## Validation Strategy

The CGA implementation is checked against independent Euclidean and projective reference formulas in [`src/geometry.py`](src/geometry.py), rather than validating an implementation against itself.

The validation layer includes:

- algebraic tests for the $G(4,1)$ metric, null basis, duality, conformal distance, sphere incidence, and point-pair extraction,
- projective covariance tests under random homographies,
- randomized two-link and three-link constraint checks,
- exact tests at special isosceles-locus heights,
- tripod distance and support-polygon checks,
- regression tests for the three-link representation singularity at $d=1$,
- a reproducibility smoke test in GitHub Actions.

The main experiment uses the fixed random seed `20260922`. The source scripts in [`projective_geometry/`](projective_geometry/) require the external course library `libcfcg`; the automated validation therefore checks their mathematical constructions through the independent NumPy reference layer.

## Results at a Glance

The checked-in full experiment validates the regular configurations at approximately floating-point precision:

| Experiment | Samples | Result |
| --- | ---: | ---: |
| Projective join/meet covariance | 5,000 | max error $3.42\times10^{-14}$ |
| Two-link distance constraints | 10,000 | max residual $4.44\times10^{-16}$ |
| Three-link distance constraints | 10,000 | max residual $4.44\times10^{-16}$ |
| Isosceles-locus candidate count | 19,944 | 0 count mismatches |
| Isosceles-locus residual | 19,944 | max residual $5.16\times10^{-15}$ |

The strongest numerical result is the three-link singularity analysis. The original construction becomes increasingly inaccurate as $d\to1$, although the geometric three-link configuration remains regular there. The reflection-based reformulation removes that representation singularity: over the tested sequence down to $|d-1|=10^{-14}$, the revised construction stays below $6.27\times10^{-16}$ position error.

![Conditioning and singularity analysis](figures/stability_analysis.png)

The complete machine-readable ledger is available in [`results/results.json`](results/results.json).

## Projective Geometry Examples

The planar examples are kept because they show the same construction logic in a simpler setting before it is applied to robot kinematics.

![Projective constructions](figures/projective_constructions.png)

The isosceles-locus example makes solution multiplicity especially visible: depending on the distance between the parallel line and the base, the construction has one, three, or five distinct admissible apices.

![Isosceles-locus candidate structure](figures/isosceles_locus.png)

## Repository Structure

~~~text
cga-robot-kinematics/
│
├── src/
│   ├── cga.py
│   ├── cga_constructions.py
│   ├── geometry.py
│   └── generate_figures.py
│
├── cga_kinematics/
│   ├── two_link_inverse_kinematics.clu
│   ├── three_link_inverse_kinematics.clu
│   └── tripod_sphere_intersection.clu
│
├── projective_geometry/
│   ├── circumcircle_construction.py
│   ├── isosceles_triangle_locus.py
│   └── perpendicular_foot_via_ideal_point.py
│
├── tests/
│   ├── conftest.py
│   ├── test_cga_core.py
│   └── test_geometric_constructions.py
│
├── results/
│   ├── results.json
│   └── results.tex
│
├── figures/
│   ├── isosceles_locus.{png,pdf}
│   ├── kinematic_constructions.{png,pdf}
│   ├── projective_constructions.{png,pdf}
│   ├── stability_analysis.{png,pdf}
│   └── tripod_construction.{png,pdf}
│
├── paper/
│   ├── README.md
│   ├── logo_hda.png
│   ├── main.tex
│   ├── references.bib
│   └── from_ideal_points_to_robot_joints.pdf
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-dev.txt
~~~

## Technical Report

The accompanying seminar paper is:

**From Ideal Points to Robot Joints — Computational Geometry and Conformal Geometric Algebra**

The LaTeX source, bibliography, build notes, and compiled PDF are stored in [`paper/`](paper/).

The report develops the projective and conformal models, derives the geometric constructions, analyses singular and degenerate cases, and documents the numerical experiments.

## Installation

Clone the repository and install the Python dependencies:

~~~bash
git clone https://github.com/yanaosinchuk/cga-robot-kinematics.git
cd cga-robot-kinematics
python -m venv .venv
pip install -r requirements.txt
~~~

The standalone numerical core uses NumPy; figure generation uses Matplotlib. The pinned environment used for the checked-in numerical ledger is Python 3.11 with NumPy 2.4.6 and Matplotlib 3.11.2.

The scripts in [`projective_geometry/`](projective_geometry/) additionally use the course-specific `libcfcg` teaching library, which is not bundled with this repository. The CLUCalc scripts in [`cga_kinematics/`](cga_kinematics/) are intended for a CLUCalc environment.

## Testing

Install the development dependencies:

~~~bash
pip install -r requirements-dev.txt
~~~

Run the test suite from the repository root:

~~~bash
python -m pytest -q
~~~

The tests cover the G(4,1) metric signature, null-basis identities, outer-product antisymmetry, inverses and duality, conformal distance encoding, sphere incidence, point normalization, point-pair extraction, projective constructions, special isosceles-locus boundaries, two-link and three-link kinematic constraints, the three-link representation singularity, tripod stability, and agreement between the revised CGA constructions and independent reference solutions.

GitHub Actions runs the same tests on every push and pull request and also executes a reduced end-to-end smoke run of the reproducibility pipeline.

## Reproducing the Numerical Study

Run the full numerical experiment from the repository root:

~~~bash
python src/generate_figures.py
~~~

This regenerates:

~~~text
figures/*.pdf
figures/*.png
results/results.json
results/results.tex
~~~

The full experiment uses the fixed seed `20260922`.

For a faster smoke run:

~~~bash
FAST=1 python src/generate_figures.py
~~~

To redirect generated files to another directory:

~~~bash
OUT_DIR=/tmp/cga-output FAST=1 python src/generate_figures.py
~~~

## Building the Technical Report

First regenerate the numerical results and PDF figures, then compile from the `paper/` directory:

~~~bash
python src/generate_figures.py
cd paper
latexmk -pdf main.tex
~~~

The report reads its numerical macros from `../results/results.tex` and its generated figures from `../figures/`.

## Academic Context

This repository grew out of geometry and geometric-algebra coursework in the B.Sc. Applied Mathematics program at Hochschule Darmstadt.

The code, written implementation, numerical validation, technical report, and repository material published here were produced independently by Yana Osinchuk.

## License and Citation

The software and documentation in this repository are released under the [MIT License](LICENSE). They may be used, modified, and redistributed subject to the licence terms.

For academic use, citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Author

**Yana Osinchuk**  
Applied Mathematics, Hochschule Darmstadt
