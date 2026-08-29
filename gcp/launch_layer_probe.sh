#!/usr/bin/env bash
# Stop the layer-2 campaign (dead layer: baseline ~4 MCC vs 45.5 at L12),
# archive its partial CSV, and launch the layer probe in the background.
# All of this lives in a file because `&`, `>` and `;` get mangled when passed
# inline through `gcloud compute ssh --command`.
set -u
cd /home/t-amitalfasi/glot
bash stop_campaign.sh
mv results/campaign_glue_colaL2.csv \
   results/campaign_glue_colaL2_ABORTED_deadlayer.csv 2>/dev/null || true
nohup bash layer_probe.sh cola "4 6 8 10 12" > logs/layer_probe.log 2>&1 &
echo "layer probe launched pid $!"
