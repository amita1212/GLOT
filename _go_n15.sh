#!/usr/bin/env bash
cd /home/t-amitalfasi/glot || exit 1
mkdir -p logs
if pgrep -f "factorial_scale_n15" > /dev/null; then echo "ALREADY_RUNNING"; exit 0; fi
sed -i 's/\r$//' factorial_n15.sh factorial_scale.py
setsid nohup bash factorial_n15.sh > logs/factorial_n15.log 2>&1 < /dev/null &
echo "LAUNCHED pid=$!"
sleep 8
cat logs/factorial_n15.log
