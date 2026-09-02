#!/usr/bin/env python
"""Along which axis were past campaigns sharded -- seed or arm?

This decides whether a third machine can safely take part of a campaign. If
shards split by ARM, a paired baseline-vs-arm test would straddle two machines,
which is the cross-machine version of the splice app:determinism forbids. If
they split by SEED, every pair stays on one machine and sharding is sound.
"""
import csv
import glob
import os
import collections

for pat in ("results/campaign_wide_cola_w*.csv", "results/campaign_wide_stsb_w*.csv"):
    files = sorted(glob.glob(os.path.expanduser("~/glot/" + pat)))
    if not files:
        continue
    print(f"=== {pat} ===")
    for f in files:
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8", errors="ignore"))
                if r.get("stage") == "confirm"]
        arms = sorted({r["arm"] for r in rows})
        seeds = sorted({int(r["seed"]) for r in rows if r.get("seed", "").isdigit()})
        print(f"  {os.path.basename(f):<32} rows={len(rows):<4} "
              f"arms={len(arms):<2} seeds={seeds}")
    print()

print("=== environment fingerprint (must match on any new machine) ===")
try:
    import geoopt, torch, torch_geometric, transformers
    print(f"  geoopt            {geoopt.__version__}")
    print(f"  torch             {torch.__version__}")
    print(f"  torch_geometric   {torch_geometric.__version__}")
    print(f"  transformers      {transformers.__version__}")
except Exception as e:  # pragma: no cover
    print("  could not import:", e)
