#!/bin/bash
# Block until >=2 ACgat results appear in the improved-probe CSV (or the probe
# finishes), then print all completed arms. Timeout ~28 min. grep -c always
# prints a count (0 on no match); its exit code 1 is harmless here.
CSV=~/GLOT/results/hyperglot_improved_results.csv
for i in $(seq 1 56); do
  n=$(grep -c 'ACgat' "$CSV" 2>/dev/null)
  n=${n:-0}
  running=$(pgrep -f 'configs ABfix' | wc -l)
  if [ "$n" -ge 2 ] || [ "$running" -eq 0 ]; then break; fi
  sleep 30
done
echo "DONE_ARMS:"
tail -n +2 "$CSV" | cut -d, -f4,6 | sort
echo "PROBE_RUNNING: $running"
