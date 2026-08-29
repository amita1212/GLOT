#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo "=== now: $(date -Is) ==="
echo "--- roberta_all.log tail ---"
tail -4 logs/roberta_all.log
echo "--- per-worker row counts (minus header) ---"
wc -l results/campaign_rob_*_w*.csv 2>/dev/null
echo "--- chain (mrpc/rte) ---"
tail -2 logs/chain_mrpc_rte.log 2>/dev/null
echo "--- machine ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
uptime
echo "--- stage breakdown so far ---"
~/glotenv/bin/python - <<'PY'
import csv, glob
from collections import Counter
rows = []
for p in glob.glob("results/campaign_rob_*_w*.csv"):
    rows += list(csv.DictReader(open(p)))
print("total rows:", len(rows))
print("by task/stage:", dict(Counter((r.get("task"), r.get("stage")) for r in rows)))
print("by arm:", dict(Counter(r.get("arm") for r in rows)))
d = [float(r["mean_density"]) for r in rows
     if r.get("mean_density") not in (None, "", "nan")]
if d:
    d.sort()
    print(f"density n={len(d)} min={d[0]:.3f} med={d[len(d)//2]:.3f} max={d[-1]:.3f}")
PY
