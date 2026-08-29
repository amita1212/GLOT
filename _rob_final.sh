#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
for t in stsb cola; do
  f=results/campaign_rob_${t}.csv
  echo "=========================================================="
  echo "  RoBERTa $t   ($(wc -l < $f) rows)"
  echo "=========================================================="
  echo "--- absolute confirmation means + paired vs baseline ---"
  $PY paired_analysis.py "$f" 2>&1 | head -22
  echo "--- vs no_graph ---"
  $PY vs_reference.py "$f" no_graph 2>&1 | head -22
  echo "--- vs paper_tau (does the fix beat the broken graph?) ---"
  $PY vs_reference.py "$f" paper_tau 2>&1 | tail -10
done
