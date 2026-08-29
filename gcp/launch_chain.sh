#!/usr/bin/env bash
# Launcher for chain_modernbert.sh.
#
# WHY A FILE: `nohup ... &`, `>` and `2>&1` get mangled when passed through
# `gcloud compute ssh --command` on Windows (plink re-parses the string), so
# every long job must be launched from a shell script that lives on the VM.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
nohup bash chain_modernbert.sh "4 8 12 16 20 22" cola > logs/chain_modernbert.log 2>&1 &
echo "launched pid $!"
