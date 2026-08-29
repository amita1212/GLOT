#!/usr/bin/env bash
# Detach the RoBERTa campaign. Kept as a file because `&` and `nohup` get
# mangled when passed through `gcloud compute ssh --command` from PowerShell.
cd /home/t-amitalfasi/glot
mkdir -p logs results
sed -i 's/\r$//' roberta_worker.sh roberta_all.sh merge_glob.py vs_reference.py 2>/dev/null
nohup bash roberta_all.sh > logs/roberta_all.log 2>&1 &
echo "launched pid $!"
