#!/usr/bin/env bash
# Preflight for the SST-2 reduced design on this machine.
cd ~/glot || exit 1
echo "=== config source seed_extend.py needs ==="
if [ -f results/campaign_wide_cola.csv ]; then
  echo "  PRESENT results/campaign_wide_cola.csv ($(wc -l < results/campaign_wide_cola.csv) lines)"
  echo "  confirmed arms available:"
  awk -F, 'NR>1 && /confirm/ {print $6}' results/campaign_wide_cola.csv | sort -u | tr '\n' ' ' | sed 's/^/    /'
  echo
else
  echo "  MISSING results/campaign_wide_cola.csv  <-- seed_extend cannot lift configs"
fi
echo
echo "=== SST-2 cache ==="
for s in train val; do
  D="data/bert-base-uncased_sst2_${s}_batches"
  if [ -f "$D/metadata.json" ]; then
    echo "  PRESENT $D ($(du -sh "$D" 2>/dev/null | cut -f1))"
  else
    echo "  MISSING $D  <-- must be prewarmed before any arm runs"
  fi
done
echo
echo "=== disk headroom (cache is ~26 GB) ==="
df -h ~ | tail -1 | sed 's/^/  /'
echo
echo "=== does anything already prewarm sst2? ==="
grep -l 'sst2' *.sh 2>/dev/null | sed 's/^/  /'
echo "  prewarm entry points:"
grep -n 'precompute_hidden_states\|--prewarm\|def precompute' main.py campaign.py 2>/dev/null | head -5 | sed 's/^/    /'
echo
echo "=== measured BERT CoLA run cost, for scaling ==="
if [ -f results/campaign_wide_cola.csv ]; then
  ~/glotenv/bin/python - <<'PY'
import csv
rs=[r for r in csv.DictReader(open('results/campaign_wide_cola.csv',encoding='utf-8',errors='ignore')) if r.get('stage')=='confirm']
t=[float(r['elapsed_sec']) for r in rs if r.get('elapsed_sec')]
if t: print(f"  CoLA confirm: {len(t)} runs, mean {sum(t)/len(t):.0f}s")
PY
fi
