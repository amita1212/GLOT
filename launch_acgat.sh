#!/bin/bash
# Detached launcher for the ACgat sweep (keeps the & inside the file so the
# SSH command line never contains a bare &, which the agent terminal mangles).
chmod +x ~/run_acgat_sweep.sh
setsid nohup bash ~/run_acgat_sweep.sh > /dev/null 2>&1 &
echo "LAUNCHED_PID: $!"
