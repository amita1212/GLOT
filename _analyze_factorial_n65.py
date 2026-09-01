"""Analyse the completed n=65 geometry factorial (results/factorial_geom_cola_n65.csv).

2x2: {base config, C config} x {Euclidean MP, hyperbolic MP}

    base_at_base  BASE_CFG, Euclidean   (baseline)
    C_at_base     BASE_CFG, hyperbolic  (curvature ONLY)
    base_at_C     C_CFG,    Euclidean   (config ONLY)
    C_at_C        C_CFG,    hyperbolic  (full Stage C)

Decomposition of total = C_at_C - base_at_base:
    curvature main effect (at base cfg) = C_at_base   - base_at_base
    config    main effect (at Euclid)   = base_at_C   - base_at_base
    interaction                         = C_at_C - C_at_base - base_at_C + base_at_base

All four cells share seeds 1..65 and ran in ONE script, ONE code path, ONE cache
state, so every contrast is PAIRED by seed and internally consistent. This is the
whole point of the n=65 rerun: the n=15 version spliced its diagonal from a
different campaign.

3 planned contrasts -> Bonferroni alpha = 0.05/3 = 0.0167.
"""
import itertools
import pandas as pd
from scipy import stats

CSV = "results/factorial_geom_cola_n65.csv"
CELLS = ["base_at_base", "C_at_base", "base_at_C", "C_at_C"]
ALPHA = 0.05 / 3

d = pd.read_csv(CSV)
before = len(d)
d = d.drop_duplicates(subset="run_key")
print(f"rows {before} -> {len(d)} after dedupe by run_key")

piv = d.pivot_table(index="seed", columns="arm", values="score")
piv = piv.dropna(subset=CELLS)
n = len(piv)
print(f"seeds complete in ALL four cells: n={n}  (range {piv.index.min()}..{piv.index.max()})\n")

print("=== cell means (MCC) ===")
for c in CELLS:
    v = piv[c]
    print(f"  {c:<14} {v.mean():7.3f}  sd {v.std(ddof=1):5.3f}  sem {v.sem():5.3f}")

dens = d.groupby("arm")["mean_density"].mean()
print("\n=== mean graph density per cell (sanity: C cfg should be ~3.6x base) ===")
for c in CELLS:
    print(f"  {c:<14} {dens.get(c, float('nan')):.4f}")


def paired(name, vec):
    t, p = stats.ttest_rel(vec, [0] * len(vec)) if False else stats.ttest_1samp(vec, 0)
    wins = int((vec > 0).sum())
    sig = "SIG" if p < ALPHA else "ns"
    ci = stats.t.interval(0.95, len(vec) - 1, loc=vec.mean(), scale=vec.sem())
    print(f"  {name:<34} {vec.mean():+7.3f}  95%CI [{ci[0]:+6.3f},{ci[1]:+6.3f}]  "
          f"t={t:+7.2f}  p={p:.2e}  {wins}/{len(vec)}  {sig}")
    return vec.mean(), p


print(f"\n=== paired contrasts, n={n}, Bonferroni alpha={ALPHA:.4f} ===")
total, _ = paired("TOTAL  C_at_C - base_at_base", piv["C_at_C"] - piv["base_at_base"])
curv, p_c = paired("  curvature only (C_at_base-base)", piv["C_at_base"] - piv["base_at_base"])
conf, p_g = paired("  config only    (base_at_C-base)", piv["base_at_C"] - piv["base_at_base"])
inter, p_i = paired("  interaction", piv["C_at_C"] - piv["C_at_base"] - piv["base_at_C"] + piv["base_at_base"])

print(f"\n  additivity check: curv+config+inter = {curv + conf + inter:+.3f}  (total {total:+.3f})")
if abs(total) > 1e-9:
    print(f"  share of total explained by CURVATURE : {100 * curv / total:5.1f}%")
    print(f"  share of total explained by CONFIG    : {100 * conf / total:5.1f}%")
    print(f"  share of total explained by INTERACT  : {100 * inter / total:5.1f}%")

print("\n=== curvature effect measured at the OTHER config (C_at_C - base_at_C) ===")
paired("  curvature at C cfg", piv["C_at_C"] - piv["base_at_C"])

print("\n=== reproducibility: seeds 1-15 vs the wide campaign's published cells ===")
s15 = piv[piv.index <= 15]
for cell, published in [("base_at_base", 45.368), ("C_at_C", 46.784)]:
    got = s15[cell].mean()
    print(f"  {cell:<14} n=15 here {got:7.3f}   campaign {published:7.3f}   diff {got - published:+6.3f}")

print("\n=== n=15 vs n=65 for each contrast (how much the small-n estimate moved) ===")
for nm, a, b in [("curvature", "C_at_base", "base_at_base"),
                 ("config", "base_at_C", "base_at_base"),
                 ("total", "C_at_C", "base_at_base")]:
    small = (s15[a] - s15[b]).mean()
    big = (piv[a] - piv[b]).mean()
    print(f"  {nm:<10} n=15 {small:+7.3f}   n=65 {big:+7.3f}   moved {big - small:+6.3f}")
