#!/usr/bin/env bash
# Compact pipeline status report.
cd /home/t-amitalfasi/glot

echo "=== stages reached ==="
grep -E '^# (STAGE|PIPELINE)' logs_pipeline.txt 2>/dev/null || echo "(no log yet)"

echo
echo "=== alive? ==="
if pgrep -f run_everything.sh > /dev/null; then echo RUNNING; else echo STOPPED; fi
echo "current python:"
pgrep -af "ablation_fair.py|sweep_paper_grid.py|diagnostic_stress_test.py|main.py" | head -3

echo
echo "=== result row counts ==="
for f in results/ablation_fair.csv results/stress_warm.csv \
         results/sweep_cola.csv results/sweep_rte.csv results/sweep_stsb.csv; do
    if [ -f "$f" ]; then
        n=$(( $(wc -l < "$f") - 1 ))
        echo "  $f : $n rows"
    else
        echo "  $f : (not started)"
    fi
done

echo
echo "=== ablation: latest 25 runs ==="
if [ -f results/ablation_fair.csv ]; then
    cut -d, -f2,3,4,5,6,7,9 results/ablation_fair.csv | tail -25
fi

echo
echo "=== tail of pipeline log ==="
tail -6 logs_pipeline.txt 2>/dev/null
