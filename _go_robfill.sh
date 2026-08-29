#!/usr/bin/env bash
# Detach the RoBERTa fill campaign. Kept as a file because `&` and `nohup` get
# mangled when passed through `gcloud compute ssh --command` from PowerShell.
cd /home/t-amitalfasi/glot
mkdir -p logs results
sed -i 's/\r$//' robfill_worker.sh robfill_all.sh 2>/dev/null
nohup bash robfill_all.sh > logs/robfill_all.log 2>&1 &
echo "launched pid $!"
