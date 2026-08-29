"""Is the Stage C gain an INTERACTION between graph density and curvature?

Interaction per seed = (C - base_at_C) - (C_at_base - baseline)
                     = C - base_at_C - C_at_base + baseline

If the two factors were additive this is 0. Tested paired over the 15 seeds.
"""
import csv
from math import comb, sqrt

def load(path, arms):
    per = {a: {} for a in arms}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r.get("stage") != "confirm":
            continue
        a = r.get("arm")
        if a in per:
            try:
                per[a][int(r["seed"])] = float(r["score"])
            except (ValueError, KeyError):
                pass
    return per

def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def report(vals, label):
    n = len(vals)
    m = sum(vals) / n
    sd = sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    t = m / (sd / sqrt(n)) if sd else float("inf")
    pos = sum(1 for x in vals if x > 0)
    neg = sum(1 for x in vals if x < 0)
    p = sign_p(pos, neg)
    sig = "  <-- SIGNIFICANT on both" if (p < 0.05 and abs(t) > 2.145) else ""
    print(f"  {label:38s} {m:+7.3f}  sd={sd:5.2f}  t={t:6.2f}  "
          f"{pos:2d}/{neg:<2d}  p={p:.4f}{sig}")
    return m

w = load("results/campaign_wide_cola.csv", ["baseline", "C"])
f = load("results/factorial_geom_cola.csv", ["C_at_base", "base_at_C"])
seeds = sorted(set(w["baseline"]) & set(w["C"]) &
               set(f["C_at_base"]) & set(f["base_at_C"]))
print(f"seeds common to all four cells: {len(seeds)}")
print()

print("MAIN EFFECTS AND INTERACTION (CoLA MCC)")
print("-" * 78)
geo_base = [f["C_at_base"][s] - w["baseline"][s] for s in seeds]
cfg_eucl = [f["base_at_C"][s] - w["baseline"][s] for s in seeds]
total = [w["C"][s] - w["baseline"][s] for s in seeds]
inter = [w["C"][s] - f["base_at_C"][s] - f["C_at_base"][s] + w["baseline"][s]
         for s in seeds]

g = report(geo_base, "geometry alone (base config)")
c = report(cfg_eucl, "config alone (Euclidean MP)")
i = report(inter, "INTERACTION")
t = report(total, "total: C - baseline")

print()
print("DECOMPOSITION")
print("-" * 78)
print(f"  geometry alone      {g:+7.3f}")
print(f"  config alone        {c:+7.3f}")
print(f"  interaction         {i:+7.3f}")
print(f"  {'-' * 26}")
print(f"  sum                 {g + c + i:+7.3f}   (observed total {t:+.3f})")
print()
print(f"  interaction is {abs(i) / max(abs(g + c), 1e-9):.1f}x the sum of "
      f"the two main effects")
