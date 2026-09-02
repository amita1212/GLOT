"""Is a (arm, seed) run reproducible? Look for repeated run_keys with different scores."""
import glob
import os
from collections import defaultdict

import pandas as pd

RES = "results_vm"
print("=== duplicate run_key with DIFFERENT score (same config, same seed) ===")
for stem in ["campaign_wide_cola", "campaign_wide_stsb", "campaign_wide_mrpc",
             "campaign_wide_rte", "campaign_rob_cola", "campaign_rob_stsb",
             "campaign_robfill_cola", "campaign_robfill_stsb"]:
    files = sorted(glob.glob(f"{RES}/{stem}.csv") + glob.glob(f"{RES}/{stem}_w*.csv"))
    d = pd.concat([pd.read_csv(f).assign(src=os.path.basename(f)) for f in files])
    g = d.groupby("run_key")["score"].agg(["count", "nunique", "min", "max"])
    bad = g[(g["count"] > 1) & (g["nunique"] > 1)]
    print(f"  {stem:24s} rows {len(d):5d}  unique keys {len(g):5d}  "
          f"repeated {int((g['count']>1).sum()):5d}  repeated-and-DIFFERENT {len(bad):4d}")
    if len(bad):
        bad = bad.assign(spread=bad["max"] - bad["min"]).sort_values("spread", ascending=False)
        print(bad.head(8).to_string())
        print(f"    mean |spread| over disagreeing keys: {bad['spread'].mean():.3f}"
              f"  max {bad['spread'].max():.3f}")

print()
print("=== factorial base_at_base cell vs wide baseline: same cfg, different campaign ===")
f = pd.read_csv(f"{RES}/factorial_geom_cola.csv")
f = f[f.stage == "confirm"].drop_duplicates("run_key")
w = pd.read_csv(f"{RES}/campaign_wide_cola.csv")
w = w[w.stage == "confirm"].drop_duplicates("run_key")
a = f[f.arm == "base_at_base"].set_index("seed")["score"].reindex(range(1, 16))
b = w[w.arm == "baseline"].set_index("seed")["score"].reindex(range(1, 16))
dd = (a - b).dropna()
print(f"  n={len(dd)}  mean diff {dd.mean():+.3f}  sd {dd.std(ddof=1):.3f}  "
      f"max|diff| {dd.abs().max():.2f}  identical {int((dd.abs()<1e-9).sum())}")
print("  => the SAME configuration at the SAME seed, re-run, moves by this much.")
print("  compare: the effect the paper wants to measure is ~1.4 MCC")

print()
print("=== how many seeds does the 65-seed factorial still need? ===")
for arm, g in f.groupby("arm"):
    seeds = sorted(g.seed.unique())
    print(f"  {arm:14s} n={len(seeds):3d}  max seed {max(seeds)}")
print("  arm C (4th cell) exists only in campaign_wide_cola at n=15")
