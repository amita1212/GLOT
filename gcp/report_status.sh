#!/usr/bin/env bash
# Print the layer-transfer probe results for every task, plus the equal-budget
# CoLA table. Lives in a file because loops and $-vars get mangled through
# `gcloud compute ssh --command` on Windows.
set -u
cd /home/t-amitalfasi/glot

echo "=== LAYER TRANSFER PROBE (L8 vs L12, cosine baseline only) ==="
for t in cola stsb rte mrpc; do
    for f in "results/layer_probe_bert-base-uncased_${t}.csv" "results/layer_probe_${t}.csv"; do
        if [ -f "$f" ]; then
            echo "--- $t  ($f) ---"
            ~/glotenv/bin/python - "$f" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
best = defaultdict(lambda: float("-inf"))
for r in rows:
    try:
        best[r["setting"]] = max(best[r["setting"]], float(r["score"]))
    except (KeyError, TypeError, ValueError):
        pass
for k in sorted(best):
    print(f"    {k:<22} {best[k]:>8.2f}")
PY
            break
        fi
    done
done

echo
echo "=== CoLA layer 8, EQUAL BUDGET (10 trials per arm) ==="
~/glotenv/bin/python show_rows.py results/campaign_glue_colaL8.csv | tail -14
