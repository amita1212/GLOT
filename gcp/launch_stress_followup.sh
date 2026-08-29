#!/bin/bash
# Launch the stress-test follow-up sweep detached from the SSH session.
cd /home/t-amitalfasi/glot || exit 1
mkdir -p logs results
rm -f results/_stress_smoke.csv
if pgrep -f stress_sweep.py > /dev/null; then
  echo "ALREADY_RUNNING"
  exit 0
fi
nohup /home/t-amitalfasi/glotenv/bin/python -u stress_sweep.py \
  > logs/stress_followup.log 2>&1 &
echo "LAUNCHED pid=$!"
sleep 8
head -4 logs/stress_followup.log
