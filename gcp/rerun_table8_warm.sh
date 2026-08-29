#!/usr/bin/env bash
# Corrected Table 8 reproduction: pre-warm every cache first, so that all 15
# cells are scored under identical conditions.
#
# The previous grid was contaminated: the CoLA cache was built by an earlier
# run (all 5 CoLA cells warm), but STS-B and RTE built their caches on their
# OWN first cell (tau=0.0), making that cell cold and the other four warm.
# Measured cold-vs-warm effect on CoLA: 40.36 vs 45.54 (5.2 points).
set -euo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

echo "########## STEP 1: pre-warm all caches ##########"
bash prewarm_caches.sh cola stsb rte

echo
echo "########## STEP 2: Table 8 grid, all cells warm ##########"
rm -f results/repro_table8_warm.csv
"$PY" repro_paper.py \
    --tasks cola stsb rte \
    --taus 0.0 0.2 0.4 0.6 0.8 \
    --override_precompute 0 \
    --out results/repro_table8_warm.csv \
    2>&1 | tee logs_table8_warm.txt | grep -E "DONE|FAIL"

echo
echo "########## SUMMARY ##########"
tail -30 logs_table8_warm.txt
