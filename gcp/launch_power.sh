#!/usr/bin/env bash
# Detached launcher for power_test.sh. Exists because `nohup ... &` is mangled
# when passed through `gcloud compute ssh --command` on Windows.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
nohup bash power_test.sh > logs/power_test.log 2>&1 &
echo "launched power_test pid $!"
