#!/usr/bin/env bash
# Block until the reproduction driver finishes, then print the summary table.
cd /home/t-amitalfasi/glot
while pgrep -f "repro_paper.py" > /dev/null; do
    sleep 20
done
echo "=== FINISHED ==="
grep -E "DONE|FAIL" logs_table8.txt
echo
tail -25 logs_table8.txt
