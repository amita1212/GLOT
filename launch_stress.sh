#!/bin/bash
# Detached launcher for the stress sweep (keeps the & inside the file).
chmod +x ~/GLOT/run_stress_sweep.sh
setsid nohup bash ~/GLOT/run_stress_sweep.sh > /dev/null 2>&1 &
echo "LAUNCHED_PID: $!"
