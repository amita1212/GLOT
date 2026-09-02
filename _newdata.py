"""Analyse the three datasets the main queue produced on 30-31 August.

Everything here uses the paper's own protocol: confirmation rows only, paired on
the shared seed set, paired t-test beside the exact two-sided sign test.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

R = "results_vm/"


def confirm(df):
    return df[df["stage"] == "confirm"]


def paired(a, b, label, seedcol="seed"):
    """a - b, paired on seed. Returns nothing; prints the row."""
    m = pd.merge(a[[seedcol, "score"]], b[[seedcol, "score"]],
                 on=seedcol, suffixes=("_a", "_b"))
    d = m["score_a"].values - m["score_b"].values
    n = len(d)
    if n < 2:
        print(f"{label:44s} n={n} -- too few")
        return
    t, pt = stats.ttest_rel(m["score_a"], m["score_b"])
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    k, nn = pos, pos + neg
    ps = stats.binomtest(k, nn, 0.5).pvalue if nn else 1.0
    print(f"{label:44s} n={n:3d}  d={100*d.mean():+7.3f}  sd={100*d.std(ddof=1):6.2f} "
          f" t={t:7.2f}  sign_p={ps:9.5f}  {pos}/{neg}")


print("=" * 100)
print("1. STAGE C FACTORIAL, n=65 -- replaces tab:factorial and tab:decomp")
print("=" * 100)
f = confirm(pd.read_csv(R + "factorial_geom_cola.csv"))
cells = {a: g.sort_values("seed") for a, g in f.groupby("arm")}
for a in ("base_at_base", "C_at_base", "base_at_C", "C_at_C"):
    g = cells[a]
    print(f"  {a:14s} n={len(g):3d}  mean={100*g['score'].mean():6.2f} "
          f" sd={100*g['score'].std(ddof=1):5.2f}")

print("\n  decomposition (all four cells from ONE campaign, one code path):")
bb, cb, bc, cc = (cells[k] for k in ("base_at_base", "C_at_base",
                                     "base_at_C", "C_at_C"))
paired(cb, bb, "geometry alone (at base config)")
paired(bc, bb, "configuration alone (Euclidean MP)")
paired(cc, bb, "TOTAL  (C_at_C - base_at_base)")
# interaction needs all four aligned on seed
mm = bb[["seed", "score"]].rename(columns={"score": "bb"})
for nm, gg in (("cb", cb), ("bc", bc), ("cc", cc)):
    mm = mm.merge(gg[["seed", "score"]].rename(columns={"score": nm}), on="seed")
inter = (mm["cc"] - mm["cb"] - mm["bc"] + mm["bb"]).values
t, pt = stats.ttest_1samp(inter, 0.0)
pos, neg = int((inter > 0).sum()), int((inter < 0).sum())
ps = stats.binomtest(pos, pos + neg, 0.5).pvalue
print(f"{'interaction':44s} n={len(inter):3d}  d={100*inter.mean():+7.3f} "
      f" sd={100*inter.std(ddof=1):6.2f}  t={t:7.2f}  sign_p={ps:9.5f}  {pos}/{neg}")
print(f"\n  check: components sum to total? "
      f"{100*((mm['cb']-mm['bb']).mean()+(mm['bc']-mm['bb']).mean()+inter.mean()):.3f}"
      f" vs {100*(mm['cc']-mm['bb']).mean():.3f}")

print()
print("=" * 100)
print("2. STAGE A on CoLA at n=50 -- settles sec:stageA")
print("=" * 100)
sa = confirm(pd.read_csv(R + "stageA_n50_cola.csv"))
print("  confirm rows per arm:", dict(sa.groupby("arm").size()))
base = sa[sa["arm"] == "baseline"].sort_values("seed")
A = sa[sa["arm"] == "A"].sort_values("seed")
print(f"  baseline n={len(base)} mean={100*base['score'].mean():.2f} "
      f"sd={100*base['score'].std(ddof=1):.2f}")
print(f"  A        n={len(A)} mean={100*A['score'].mean():.2f} "
      f"sd={100*A['score'].std(ddof=1):.2f}")
paired(A, base, "Stage A - baseline, CoLA")

print()
print("=" * 100)
print("3. DECODER STS-B, arms B and C ALONE -- fills the PEND cells")
print("=" * 100)
dbc = confirm(pd.read_csv(R + "campaign_decoder_stsb_BC.csv"))
print("  confirm rows per arm:", dict(dbc.groupby("arm").size()))
for a, g in dbc.groupby("arm"):
    print(f"  {a:3s} n={len(g):3d} mean={100*g['score'].mean():6.2f} "
          f"sd={100*g['score'].std(ddof=1):5.2f}")

# baseline for the decoder campaign lives in the original decoder file
cand = [p for p in glob.glob(R + "*decoder*.csv") if "BC" not in os.path.basename(p)]
print("  decoder baseline candidates:", [os.path.basename(c) for c in cand])
for c in cand:
    d = pd.read_csv(c)
    if "arm" not in d.columns:
        continue
    d = confirm(d)
    b = d[d["arm"] == "baseline"]
    if len(b) == 0:
        continue
    print(f"  using {os.path.basename(c)}: baseline n={len(b)} "
          f"mean={100*b['score'].mean():.2f} sd={100*b['score'].std(ddof=1):.2f}")
    for a, g in dbc.groupby("arm"):
        paired(g.sort_values("seed"), b.sort_values("seed"),
               f"decoder {a} - baseline")
    break
