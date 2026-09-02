"""Second pass: dedup tuning rows, factorial detail, stress/mteb, density."""
import glob
import os
from math import comb, sqrt

import pandas as pd

RES = "results_vm"
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
     8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
     14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000,
     64: 2.000, 120: 1.980}


def tcrit(n):
    df = max(1, n - 1)
    return T.get(df, T[min(T, key=lambda k: abs(k - df))])


def signp(d):
    p = sum(1 for x in d if x > 0); m = sum(1 for x in d if x < 0)
    n = p + m
    if n == 0:
        return 1.0
    k = min(p, m)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b):
    """a, b are Series indexed by seed."""
    idx = a.dropna().index.intersection(b.dropna().index)
    d = (a[idx] - b[idx]).dropna()
    n = len(d)
    m = d.mean(); sd = d.std(ddof=1); se = sd / sqrt(n)
    t = m / se if se else float("nan")
    return dict(n=n, delta=m, sd=sd, t=t, p=signp(d.tolist()),
                pos=int((d > 0).sum()), neg=int((d < 0).sum()))


print("=" * 78)
print("A.  TUNING TRIALS THAT FAIL TO TRAIN  (deduped on run_key)")
print("=" * 78)
frames = []
for p in glob.glob(f"{RES}/campaign_*.csv") + glob.glob(f"{RES}/factorial_*.csv"):
    if "_w" in os.path.basename(p) or "ABORTED" in p:
        continue
    d = pd.read_csv(p)
    if "lr" not in d.columns or "stage" not in d.columns:
        continue
    d = d[d["stage"] == "tune"].drop_duplicates(subset="run_key")
    if len(d):
        frames.append(d.assign(src=os.path.basename(p))[
            ["src", "model", "task", "lr", "score"]])
d = pd.concat(frames)
print(f"  total deduped tuning rows across all campaigns: {len(d)}")
g = d.groupby("lr").agg(n=("score", "size"), zero=("score", lambda s: int((s <= 0).sum())))
g["pct"] = (100 * g["zero"] / g["n"]).round(1)
print(g.to_string())
print("\n  by (model, task, lr) where any failure:")
g2 = d.groupby(["model", "task", "lr"]).agg(
    n=("score", "size"), zero=("score", lambda s: int((s <= 0).sum())))
g2["pct"] = (100 * g2["zero"] / g2["n"]).round(1)
print(g2[g2["zero"] > 0].to_string())
print("\n  CoLA budget wasted (BERT):")
bc = d[(d.model == "bert-base-uncased") & (d.task == "cola")]
print(f"    total tune rows {len(bc)}, zero-scoring {int((bc.score<=0).sum())} "
      f"= {100*(bc.score<=0).mean():.1f}%")

print()
print("=" * 78)
print("B.  STAGE C FACTORIAL  -- current state of the 65-seed run")
print("=" * 78)
f = pd.read_csv(f"{RES}/factorial_geom_cola.csv")
f = f[f.stage == "confirm"].drop_duplicates(subset="run_key")
w = pd.read_csv(f"{RES}/campaign_wide_cola.csv")
w = w[w.stage == "confirm"].drop_duplicates(subset="run_key")
cells = {
    "base_euc  (q .10, K4, h256, euclid)": f[f.arm == "base_at_base"].set_index("seed")["score"],
    "base_hyp  (q .10, K4, h256, hyper )": f[f.arm == "C_at_base"].set_index("seed")["score"],
    "Ccfg_euc  (q .38, K2, h128, euclid)": f[f.arm == "base_at_C"].set_index("seed")["score"],
}
cells["Ccfg_hyp  (q .38, K2, h128, hyper) = arm C"] = w[w.arm == "C"].set_index("seed")["score"]
wide_base = w[w.arm == "baseline"].set_index("seed")["score"]
cells["wide baseline arm (same cfg as base_euc)"] = wide_base
for k, v in cells.items():
    print(f"  {k:<48} n={len(v):3d}  mean {v.mean():7.3f}  sd {v.std(ddof=1):5.3f}")

s15 = list(range(1, 16))
print("\n  -- restricted to seeds 1..15 --")
c15 = {k: v.reindex(s15).dropna() for k, v in cells.items()}
for k, v in c15.items():
    print(f"  {k:<48} n={len(v):3d}  mean {v.mean():7.3f}  sd {v.std(ddof=1):5.3f}")

be = c15["base_euc  (q .10, K4, h256, euclid)"]
bh = c15["base_hyp  (q .10, K4, h256, hyper )"]
ce = c15["Ccfg_euc  (q .38, K2, h128, euclid)"]
ch = c15["Ccfg_hyp  (q .38, K2, h128, hyper) = arm C"]
print("\n  decomposition at n=15, using the FACTORIAL's own base_at_base cell:")
for name, (x, y) in [("geometry alone (base cfg)", (bh, be)),
                     ("configuration alone (euclid)", (ce, be)),
                     ("total C - base", (ch, be))]:
    r = paired(x, y)
    print(f"    {name:<30} {r['delta']:+7.3f}  sd {r['sd']:5.2f}  t {r['t']:6.2f}"
          f"  {r['pos']}/{r['neg']}  signp {r['p']:.4f}")
inter = (ch.reindex(s15) - ce.reindex(s15)) - (bh.reindex(s15) - be.reindex(s15))
inter = inter.dropna()
print(f"    {'interaction':<30} {inter.mean():+7.3f}  sd {inter.std(ddof=1):5.2f}"
      f"  t {inter.mean()/(inter.std(ddof=1)/sqrt(len(inter))):6.2f}"
      f"  {int((inter>0).sum())}/{int((inter<0).sum())}  signp {signp(inter.tolist()):.4f}")
print("\n  decomposition at n=15, using the WIDE baseline arm (what the paper did):")
for name, (x, y) in [("geometry alone", (bh, wide_base.reindex(s15).dropna())),
                     ("configuration alone", (ce, wide_base.reindex(s15).dropna())),
                     ("total C - baseline", (ch, wide_base.reindex(s15).dropna()))]:
    r = paired(x, y)
    print(f"    {name:<30} {r['delta']:+7.3f}  sd {r['sd']:5.2f}  t {r['t']:6.2f}"
          f"  {r['pos']}/{r['neg']}  signp {r['p']:.4f}")

print("\n  realised density per cell (factorial file):")
print(f[f.stage == "confirm"].groupby("arm")["mean_density"].agg(["mean", "count"]).round(5).to_string())

print()
print("=" * 78)
print("C.  DENSITY OF THE no_graph ARM AND THE SELF-LOOP TELEMETRY OFFSET")
print("=" * 78)
for t_ in ["cola", "stsb", "mrpc", "rte"]:
    dd = pd.read_csv(f"{RES}/campaign_wide_{t_}.csv")
    dd = dd[(dd.stage == "confirm")].drop_duplicates(subset="run_key")
    ng = dd[dd.arm == "no_graph"]["mean_density"]
    print(f"  {t_:5s} no_graph logged density mean {ng.mean():.5f}")

print()
print("=" * 78)
print("D.  STRESS TEST")
print("=" * 78)
for p in ["results/hyperglot_stress_results.csv", "results/stress_poolers.csv",
          f"{RES}/campaign_stress_hard.csv"]:
    if not os.path.exists(p):
        continue
    dd = pd.read_csv(p)
    print(f"\n  -- {p}  shape {dd.shape}")
    print("     cols:", list(dd.columns)[:20])
    for c in ["arm", "pooler", "ratio", "distractor_ratio"]:
        if c in dd.columns:
            print(f"     {c}: {sorted(dd[c].dropna().unique().tolist())[:15]}")

print()
print("=" * 78)
print("E.  MTEB / IMDB")
print("=" * 78)
p = "results/hyperglot_docmteb_results.csv"
if os.path.exists(p):
    dd = pd.read_csv(p)
    print(dd.to_string()[:4000])
