#!/usr/bin/env bash
# Stop the pipeline. Run from a FILE, never via `ssh --command`, otherwise the
# pkill pattern matches the remote `bash -c ...` process itself and kills the
# session before it can report anything.
cd /home/t-amitalfasi/glot

pkill -f run_everything.sh   || true
pkill -f ablation_fair.py    || true
pkill -f sweep_paper_grid.py || true
pkill -f repro_paper.py      || true
# match main.py without writing the literal path this script was invoked with
pkill -f "glot/hyperglot/ma""in.py" || true
sleep 3

echo "REMAINING:"
pgrep -af "run_everything|ablation_fair|sweep_paper_grid|main.py" | grep -v stop_pipeline | head -5 || echo "  (none)"

echo "ROWS in ablation_fair.csv:"
wc -l < results/ablation_fair.csv 2>/dev/null || echo 0

cp -f results/ablation_fair.csv results/ablation_fair_BROKEN_rho.csv 2>/dev/null && echo "BACKED_UP -> results/ablation_fair_BROKEN_rho.csv"
