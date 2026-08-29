"""What did the search actually SELECT? One row per arm per campaign.

Also checks the paper's claim that "every arm including the baseline selected
lr = 2e-4", which the CoLA confirmed configs appear to contradict.
"""
import csv
import glob
import os

KEYS = ["lr", "num_layers", "gat_hidden_dim", "proj_dim", "scorer_hidden",
        "jk_mode", "weight_decay", "tau_quantile", "rho_quantile", "tau",
        "curvature", "graph_curvature", "feature_mode", "hyp_gnn_type"]

lrs = {}
for path in sorted(glob.glob("results/campaign_*.csv")):
    base = os.path.basename(path)
    if any(base.endswith(f"_w{i}.csv") for i in range(4)):
        continue                      # worker shards duplicate the merged file
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8"))
            if r.get("stage") == "confirm"]
    if not rows:
        continue
    print("=" * 100)
    print(base)
    print("=" * 100)
    arms = sorted({r["arm"] for r in rows})
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        cfgs = {r["detail"] for r in sub}
        sel = {}
        for k in KEYS:
            v = {r.get(k) for r in sub if r.get(k) not in ("", None)}
            if v:
                sel[k] = "|".join(sorted(v))
        lr = sel.get("lr", "?")
        lrs.setdefault(lr, []).append(f"{base.replace('campaign_','').replace('.csv','')}/{arm}")
        flag = "" if len(cfgs) == 1 else "  [!! multiple configs]"
        print(f"  {arm:12s} n={len(sub):2d}  " +
              "  ".join(f"{k}={sel[k]}" for k in KEYS if k in sel) + flag)
    print()

print("=" * 100)
print("LEARNING RATE ACTUALLY SELECTED  (paper claims 2e-4 for every arm)")
print("=" * 100)
for lr, who in sorted(lrs.items()):
    print(f"  lr={lr:10s} n_arms={len(who):3d}   {', '.join(sorted(who)[:8])}"
          + (" ..." if len(who) > 8 else ""))
