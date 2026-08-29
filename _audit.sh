#!/usr/bin/env bash
# Recompute, from the CSVs, every number the paper asserts, so each table can be
# diffed against its source rather than trusted.
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
echo "##################### FILES #####################"
ls -1 results/*.csv | sed 's/^/  /'
echo
echo "##################### RUN COUNTS #####################"
for f in results/campaign_wide_cola.csv results/campaign_wide_stsb.csv \
         results/campaign_wide_mrpc.csv results/campaign_wide_rte.csv \
         results/campaign_rob_cola.csv results/campaign_rob_stsb.csv \
         results/campaign_tiny_stsb.csv; do
  [ -f "$f" ] && printf "  %-38s %6s rows\n" "$f" "$(($(wc -l < $f)-1))"
done
echo "  ---- total rows across every results CSV ----"
cat results/*.csv 2>/dev/null | grep -vc '^run_key\|^timestamp' | sed 's/^/  /'

echo
echo "##################### tab:arms  BERT CoLA #####################"
[ -f results/campaign_wide_cola.csv ] && {
  $PY paired_analysis.py results/campaign_wide_cola.csv 2>&1 | head -20
  $PY vs_reference.py results/campaign_wide_cola.csv no_graph 2>&1 | tail -14; }
echo
echo "##################### tab:arms  BERT STS-B #####################"
[ -f results/campaign_wide_stsb.csv ] && {
  $PY paired_analysis.py results/campaign_wide_stsb.csv 2>&1 | head -20
  $PY vs_reference.py results/campaign_wide_stsb.csv no_graph 2>&1 | tail -14; }
echo
echo "##################### MRPC / RTE (n=?) #####################"
for t in mrpc rte; do
  f=$(ls results/campaign_wide_${t}*.csv 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "--- $t ($f) ---"; $PY paired_analysis.py "$f" 2>&1 | head -16; }
done
echo
echo "##################### app:stress #####################"
[ -f results/stress_followup.csv ] && $PY stress_table.py results/stress_followup.csv 2>&1 | head -20
echo
echo "##################### tab:repro estimators #####################"
$PY estimator_gap.py 2>&1 | head -25
echo
echo "##################### seed-42 grid (tab:repro right) #####################"
$PY seed42_best.py 2>&1 | head -20
