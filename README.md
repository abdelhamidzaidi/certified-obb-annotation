# Conformally Certified Oriented Bounding-Box Annotation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24337537.svg)](https://doi.org/10.5281/zenodo.24337537)

Code and processed data accompanying the manuscript *Conformally Certified Oriented Bounding-Box Annotation from
Automatic Proposals* (A. T. Zaidi, Department of Mathematics, College of Science, Qassim
University). Running the numbered scripts in order reproduces every table and figure of the
manuscript; Table A2 of the manuscript and the table below map each file to the sections, results,
tables and figures it reproduces. The comments in the code use the notation of the manuscript.

Author: Abdelhamid Taieb Zaidi (ORCID [0000-0003-1305-4959](https://orcid.org/0000-0003-1305-4959)).
To cite this repository, use the metadata in `CITATION.cff` or the Zenodo DOI [10.5281/zenodo.24337537](https://doi.org/10.5281/zenodo.24337537).

## Quick start

```
pip install -r requirements.txt
unzip data/obb_certified_data.zip            # creates results/ and logs_reference/
ISAID=path/to/iSAID_val.json LABEL_DIR=path/to/DOTA/labels sh reproduce.sh
```

`reproduce.sh` runs steps 3 to 16 and the two independent scripts from the outputs of steps 1 and 2,
which are included in the data archive. `ISAID` is needed only for one block of step 16, and
`LABEL_DIR` is optional because the acquisition sources are stored in `results/image_sources.json`.

## Notation used in the code

| Code | Manuscript | Meaning |
|---|---|---|
| `hull` | K = conv(O) | reference hull of an iSAID outline O |
| `Astar` | A*(K) | area of the minimum-area rectangle MBR(K) |
| `b = (theta, a0, a1, b0, b1)` | R = [a0, a1] x [b0, b1] | rectangle in its own frame, sides w and h |
| `corners(b, q)` | R^(q) | proportional enlargement, sides (1 + 2q) w and (1 + 2q) h |
| `ea`, `eb` | e_a, e_b | largest excesses of K beyond R (Equation (28)) |
| `s` | s(K, R) | outer score max(e_a / w, e_b / h) (Equation (29)) |
| `L` | L(q) | uncovered fraction of K outside R^(q) on the grid Q (Section 3.7) |
| `raw` | area(R) / A*(K) | area ratio of the uncalibrated proposal |
| `cq(x, alpha)` | q_hat = s_(k) | k-th smallest score, k = ceil((n + 1)(1 - alpha)) (Equation (30)) |
| `Lj` | L_j(q) | mean uncovered fraction of calibration image I_j |
| `r` | r(K, P_in) | inner score (Section 3.8) |
| `iou_ds` | g | agreement statistic of selective certification (Section 3.9) |

## Data and model weights

| Item | Source |
|---|---|
| DOTA v1.0 validation images and labels | Ultralytics `DOTAv1.zip` (images `DOTAv1/images/val`) and the original DOTA-format label files (`LABEL_DIR`, one `.txt` per image) |
| iSAID validation annotations | iSAID project page, `iSAID_val.json` (instance polygons) |
| YOLO11m-OBB weights | Ultralytics release v8.3.0, `yolo11m-obb.pt` (SHA-256 prefix `41832a4349c08190`) |
| MobileSAM weights | Ultralytics `mobile_sam.pt` (SHA-256 prefix `6dbb90523a35330f`) |

Place the DOTA images under `dota/DOTAv1/images/val/` (needed for Figure 10) and set the environment
variable `MOBILE_SAM_WEIGHTS` to the path of `mobile_sam.pt` if it is not in the working directory.

## Installation

```
pip install -r requirements.txt
```

The versions used for the manuscript are listed in `requirements.txt` and in Table A1.

## Running the pipeline

In one command, from the outputs of steps 1 and 2 (`data/obb_certified_data.zip`, File S2 of the manuscript):

```
ISAID=path/to/iSAID_val.json LABEL_DIR=path/to/DOTA/labels sh reproduce.sh
```

Each step writes its printed output to `logs/stepNN.txt`; compare them with `logs_reference/` in File S2.
I checked this in a new Python 3.13 virtual environment with only the packages of `requirements.txt`:
all printed numbers were identical to the manuscript run, `riskgrid.json`, `scores.json` and
`calib2.json` were identical byte for byte, and the figures had identical content.

Run the scripts from this directory; intermediate files are written to `results/`. Steps 1 and 2
need the images, the annotations and the model weights (several hours on CPU); steps 3–16 run on
a CPU from `results/match.json` and `results/det_m/` in about 20 minutes.

| Step | Command | Manuscript |
|---|---|---|
| 1 | `python 01_detect.py IMG_DIR results/det_m yolo11m-obb.pt` | Section 3.12 (oriented detector) |
| 2 | `python 02_match_segment.py iSAID_val.json results/det_m LABEL_DIR IMG_DIR results/match.json sam` | Sections 3.11–3.12 |
| 3 | `python 03_score_curves.py results/match.json results/riskgrid.json` | Equations (28)–(29), Lemma 4, Section 3.7 |
| 4 | `python 04_object_scores.py results/match.json results/scores.json` | Sections 3.13, 4.2, 4.8 |
| 5 | `python 05_features.py results/match.json results/features.json` | Section 3.9 |
| 6 | `python 06_calibrate.py results/riskgrid.json results/calib2.json 200` | Theorems 3 and 4, Corollary 1; Figure 8 |
| 7 | `python 07_extra_analyses.py` | Proposition 3, Corollary 3; Sections 4.5, 4.7, 4.8 |
| 8 | `python 08_inner_scores.py results/match.json results/inner2.json` | Section 3.8 |
| 9 | `python 09_sandwich.py` | Theorem 5, Section 4.7 |
| 10 | `python 10_per_class.py` | Table 9 |
| 11 | `python 11_figures.py FIG_DIR` | Figures 8 and 10 |
| 12 | `python 12_audit_matching.py` | Section 4.8 |
| 13 | `python 13_tables.py` | Tables 2, 5, 6, 7, 8; values quoted in Sections 4.2–4.8 |
| 14 | `python 14_pooled_and_inner.py` | Section 4.4 (pooled calibration), Section 4.7 (inner scores) |
| 15 | `python 15_extra_figures.py FIG_DIR` | Figures 5, 6, 7 and 9 (Sections 4.4, 4.5 and 4.9) |
| 16 | `python 16_revision_analyses.py iSAID_val.json LABEL_DIR results/revision.json` | Tables 2, 5, 7, 8, 9 and A3; Sections 4.2–4.11 (common subset, omission on O, Beta and test-sample spread, object-weighted class coverage, sandwich informativeness, acquisition shift, observable eligibility, recall by orientation, proposal-based labelling rule) |

Independent scripts:

| Command | Manuscript |
|---|---|
| `python numerical_checks.py` | Section 4.1, Tables 3 and 4, Equations (9)–(10) |
| `python orientation_rule.py results/match.json` | Section 4.11, Table 10 |
| `python tikz/fig_notation_geometry.py` | coordinates of Figure 1 |
| `python tikz/fig_theorem1_geometry.py` | coordinates and curves of Figure 3 |

`tikz/standalone/` contains the LaTeX sources of Figures 1–4 as standalone documents.

## Random seeds and grids

- Development subset: 112 images drawn with `numpy.random.default_rng(7)` from the sorted names of
  the 449 images that contain an object with all five proposals; excluded from all reported results.
- Splits: 200 random partitions of the 337 evaluation images into 168 calibration and 169 test
  images, drawn with `numpy.random.default_rng(2026)`, which also draws the one-object-per-image
  selections. Step 15 replays the same sequence, so the statistics it prints for Figures 6, 7 and 9
  equal the corresponding entries of Tables 6, 7 and 9.
- Examples of Figure 10: `numpy.random.default_rng(5)`.
- Margin grid Q: 21 equally spaced values on [0, 0.05] followed by 35 on [0.06, 0.40].

## Processed data (File S2)

`data/obb_certified_data.zip` contains the `results/` directory written by steps 1 to 16: the detections
(`det_m/`), the reference hulls, matches and mask outlines (`match.json`), the scores and
uncovered-fraction curves (`riskgrid.json`, `scores.json`, `inner2.json`, `features.json`), the outputs of
the calibration steps (`calib2.json`, `extra.json`, `revision.json`) and the objects of Figure 10
(`figure10_objects.json`), the list of development and evaluation images with the 200 partitions of
step 6 (`splits.json`), the acquisition source of each image (`image_sources.json`), and the printed
output of every step of the reference run (`logs_reference/`). Unzip it in the repository root to run steps 3 to 16 without the images and the
models. The development images and the 200 splits are regenerated from the seeds listed above.

## Licenses

The code is released under the MIT License (`LICENSE`). The processed data in `data/` derive from
the iSAID annotations and the DOTA v1.0 images and labels and are released under the Creative
Commons Attribution-NonCommercial 4.0 International License (`data/LICENSE.md`); the original
datasets remain subject to their own terms of use.
