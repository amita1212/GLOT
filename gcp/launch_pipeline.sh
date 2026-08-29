#!/usr/bin/env bash
# Wait for any in-flight job to finish, then launch the full pipeline detached
# so it survives SSH disconnects and keeps running unattended.
cd /home/t-amitalfasi/glot

while pgrep -f "ablation_fair.py|repro_paper.py|sweep_paper_grid.py" > /dev/null; do
    sleep 15
done

rm -f results/_smoke.csv
nohup bash run_everything.sh > logs_pipeline.txt 2>&1 &
echo "PIPELINE_PID=$!"
sleep 5
echo "--- first lines ---"
head -20 logs_pipeline.txt 2>/dev/null || true
