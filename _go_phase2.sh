#!/usr/bin/env bash
# Detach phase2 from the SSH session (gcloud ssh --command mangles nohup inline).
cd /home/t-amitalfasi/glot || exit 1
mkdir -p logs
if pgrep -f "phase2.sh" | grep -qv $$ 2>/dev/null; then :; fi
if pgrep -f "factorial_scale.py" > /dev/null; then
    echo "ALREADY_RUNNING factorial"; exit 0
fi
sed -i 's/\r$//' phase2.sh factorial_scale.py stress_poolers.py backbone_recipe.py
setsid nohup bash phase2.sh > logs/phase2.log 2>&1 < /dev/null &
echo "LAUNCHED phase2 pid=$!"
sleep 5
tail -5 logs/phase2.log
