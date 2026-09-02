#!/usr/bin/env bash
cd ~/glot || exit 1
echo "=== TinyLlama caches present ==="
ls -d data/TinyLlama*batches 2>/dev/null | sed 's/^/  /'
echo
echo "=== does the MRPC cache have metadata.json? (queue skips the job if not) ==="
for s in train val; do
  D="data/TinyLlama_TinyLlama-1.1B-Chat-v1.0_mrpc_${s}_batches"
  if [ -f "$D/metadata.json" ]; then echo "  PRESENT  $D"; else echo "  MISSING  $D"; fi
done
echo
echo "=== tl_cola rows (135 confirm + 90 tune = 225 target) ==="
wc -l < results/campaign_t1_tl_cola.csv
echo
echo "=== queue log tail ==="
tail -6 logs/queue_table1_gap.log
