"""Third pass: why does the factorial's own base cell disagree with the wide baseline?"""
import glob
import os
from math import comb, sqrt

import pandas as pd

RES = "results_vm"
pd.set_option("display.width", 200)

f = pd.read_csv(f"{RES}/factorial_geom_cola.csv")
f = f[f.stage == "confirm"].drop_duplicates(subset="run_key")
w = pd.read_csv(f"{RES}/campaign_wide_cola.csv")
w = w[w.stage == "confirm"].drop_duplicates(subset="run_key")

print("=== detail strings, one row per factorial arm ===")
for arm, g in f.groupby("arm"):
    print(f"\n{arm}  (n={len(g)}, trials {sorted(g.trial.unique())})")
    print("   ", g.iloc[0]["detail"])
    print("    run_key example:", g.iloc[0]["run_key"])
print("\nwide baseline:")
print("   ", w[w.arm == "baseline"].iloc[0]["detail"])
print("    run_key example:", w[w.arm == "baseline"].iloc[0]["run_key"])
print("wide C:")
print("   ", w[w.arm == "C"].iloc[0]["detail"])

print("\n=== per-seed: factorial base_at_base vs wide baseline ===")
a = f[f.arm == "base_at_base"].set_index("seed")["score"]
b = w[w.arm == "baseline"].set_index("seed")["score"]
cmp = pd.DataFrame({"factorial_base": a.reindex(range(1, 16)),
                    "wide_baseline": b.reindex(range(1, 16))})
cmp["diff"] = cmp.factorial_base - cmp.wide_baseline
print(cmp.to_string())
print(f"\n  identical on {int((cmp['diff'].abs() < 1e-9).sum())} of 15 seeds")

print("\n=== base_at_base across all 65 seeds ===")
print(a.sort_index().to_string())

print("\n=== C_at_base seeds ===")
print(f[f.arm == "C_at_base"].set_index("seed")["score"].sort_index().to_string())

# ---------------------------------------------------------------- dead lr, fixed
print("\n" + "=" * 78)
print("dead-lr, deduped, ALL campaign files (bug in previous pass fixed)")
print("=" * 78)
frames = []
for p in sorted(glob.glob(f"{RES}/campaign_*.csv")) + sorted(glob.glob(f"{RES}/factorial_*.csv")):
    base = os.path.basename(p)
    if "ABORTED" in base:
        continue
    # shard files are  *_w0.csv .. *_w9.csv
    if base[:-4].split("_")[-1].startswith("w") and base[:-4].split("_")[-1][1:].isdigit():
        continue
    d = pd.read_csv(p)
    if "lr" not in d.columns or "stage" not in d.columns:
        print(f"  (skip {base}: no lr column)")
        continue
    d = d[d["stage"] == "tune"].drop_duplicates(subset="run_key")
    if len(d):
        frames.append(d.assign(src=base)[["src", "model", "task", "lr", "score"]])
        print(f"  {base:34s} tune rows {len(d)}")
d = pd.concat(frames)
print(f"\n  TOTAL deduped tuning rows: {len(d)}")
g = d.groupby("lr").agg(n=("score", "size"), zero=("score", lambda s: int((s <= 0).sum())))
g["pct"] = (100 * g["zero"] / g["n"]).round(1)
print(g.to_string())
print("\n  by (model, task, lr):")
g2 = d.groupby(["model", "task", "lr"]).agg(
    n=("score", "size"), zero=("score", lambda s: int((s <= 0).sum())))
g2["pct"] = (100 * g2["zero"] / g2["n"]).round(1)
print(g2.to_string())
