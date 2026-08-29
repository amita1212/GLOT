#!/usr/bin/env bash
# Launch the Table 8 reproduction grid detached, so it survives SSH disconnects.
set -euo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python
nohup "$PY" repro_paper.py \
    --tasks cola stsb rte \
    --taus 0.0 0.2 0.4 0.6 0.8 \
    --resume \
    > logs_table8.txt 2>&1 &
echo "LAUNCHED_PID=$!"
