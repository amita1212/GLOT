#!/usr/bin/env bash
cd ~/glot || exit 1
echo "=== roberta mentions in the log (tqdm CR can hide line starts) ==="
grep -a 'FacebookAI/roberta-base' logs/mteb_table3.log | tail -2 | cut -c1-110
echo
echo "=== running driver, model argument ==="
for p in $(pgrep -f run_all_experiments.py); do
  printf '  pid %s: ' "$p"
  tr '\0' ' ' < "/proc/$p/cmdline" | grep -o -- '--models [^ ]*'
done
echo
echo "=== how long the current unit has been running ==="
ps -eo etime,cmd | grep run_all_experiments | grep -v grep | awk '{print "  "$1}'
echo
echo "=== BERT block wall-clock, for the per-unit rate ==="
first=$(grep -a -oE '^--- [0-9-]+ [0-9:]+  bert' logs/mteb_table3.log | head -1)
last=$(grep -a -oE '^--- [0-9-]+ [0-9:]+  bert' logs/mteb_table3.log | tail -1)
echo "  first: $first"
echo "  last : $last"
