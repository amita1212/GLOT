#!/bin/bash
# Block until >=2 ABfix2 results (threshold+knn on CoLA) land, or the probe
# finishes. Then print the completed arms. Timeout ~25 min.
CSV=~/GLOT/results/hyperglot_abfix2_results.csv
for i in $(seq 1 50); do
  n=$(grep -c 'ABfix2' "$CSV" 2>/dev/null); n=${n:-0}
  running=$(pgrep -f 'configs ABfix2' | wc -l)
  if [ "$n" -ge 2 ] || [ "$running" -eq 0 ]; then break; fi
  sleep 30
done
echo "ABFIX2_DONE:"
tail -n +2 "$CSV" 2>/dev/null | cut -d, -f4,6,31,33 | sort
echo "PROBE_RUNNING: $running"
