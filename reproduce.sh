#!/bin/sh
# Steps 3 to 16, numerical_checks.py and orientation_rule.py, from the outputs of steps 1 and 2
# (results/match.json and results/det_m/, both in File S2). Each step writes its printed output to
# logs/stepNN.txt; the logs of the manuscript run are in File S2 under logs_reference/.
#   ISAID       path to iSAID_val.json                       (step 16, block B)
#   LABEL_DIR   directory of the DOTA v1.0 label files        (step 16, block F; optional when
#               results/image_sources.json is present)
#   images under dota/DOTAv1/images/val/                     (Figure 10 only)
set -e
ISAID=${ISAID:-iSAID_val.json}
LABEL_DIR=${LABEL_DIR:-labels_val}
mkdir -p logs figs
run() { n=$1; shift; echo "step $n: $*"; python "$@" > logs/step$n.txt 2>&1; }
run 03 03_score_curves.py results/match.json results/riskgrid.json
run 04 04_object_scores.py results/match.json results/scores.json
run 05 05_features.py results/match.json results/features.json
run 06 06_calibrate.py results/riskgrid.json results/calib2.json 200
run 07 07_extra_analyses.py
run 08 08_inner_scores.py results/match.json results/inner2.json
run 09 09_sandwich.py
run 10 10_per_class.py
run 11 11_figures.py figs
run 12 12_audit_matching.py
run 13 13_tables.py
run 14 14_pooled_and_inner.py
run 15 15_extra_figures.py figs
run 16 16_revision_analyses.py "$ISAID" "$LABEL_DIR" results/revision.json
run NC numerical_checks.py
run OR orientation_rule.py results/match.json
echo done
