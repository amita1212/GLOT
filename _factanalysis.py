"""Decompose the CoLA C-vs-baseline gap into geometry and configuration.

                        Euclidean          hyperbolic
    base-config     baseline           C_at_base
    C-config        base_at_C          C

Two independent readings of the geometry effect (one per config column), two of
the configuration effect (one per geometry row). All comparisons are paired on
the shared seed set 1..15, and every contrast gets both the paired t-test and
the exact two-sided sign test -- the same both-tests bar the paper imposes.
"""
import csv
import math
import os
from itertools import combinations  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
WIDE = os.path.join(HERE, "results", "campaign_wide_cola.csv")
GEOM = os.path.join(HERE, "results", "factorial_geom_cola.csv")


def load(path, arms):
    out = {a: {} for a in arms}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r.get("stage") == "confirm" and r.get("arm") in arms:
            try:
                out[r["arm"]][int(r["seed"])] = float(r["score"])
            except (TypeError, ValueError):
                pass
    return out


def sign_test(diffs):
    """Exact two-sided sign test; zeros discarded."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), pos, neg


def ttest(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) if var > 0 else 0.0
    return m, (m / se if se > 0 else float("inf"))


def contrast(name, a, b, data):
    """paired a - b"""
    seeds = sorted(set(data[a]) & set(data[b]))
    if not seeds:
        print(f"  {name:34s}  (no shared seeds yet)")
        return
    d = [data[a][s] - data[b][s] for s in seeds]
    m, t = ttest(d)
    p, pos, neg = sign_test(d)
    star = ""
    if p < 0.05 and abs(t) > 2.145:          # t_{.975,14}
        star = "  <-- significant on BOTH tests"
    print(f"  {name:34s}  d={m:+6.3f}  t={t:+6.2f}  {pos:2d}/{neg:<2d} "
          f"sign p={p:.4f}  n={len(seeds)}{star}")


ARMS = ["baseline", "C", "C_at_base", "base_at_C"]
data = {}
data.update(load(WIDE, ["baseline", "C"]))
data.update(load(GEOM, ["C_at_base", "base_at_C"]))

print("=" * 78)
print("CELL MEANS (CoLA MCC)")
print("=" * 78)
for a in ARMS:
    v = data.get(a, {})
    if v:
        print(f"  {a:12s} n={len(v):2d}  mean={sum(v.values())/len(v):7.3f}")
    else:
        print(f"  {a:12s} -- not available yet")

full = all(len(data.get(a, {})) == 15 for a in ARMS)
print("\n" + "=" * 78)
print("GEOMETRY EFFECT  (hyperbolic - Euclidean, holding config fixed)")
print("=" * 78)
contrast("at base-config: C_at_base-baseline", "C_at_base", "baseline", data)
contrast("at C-config:    C - base_at_C", "C", "base_at_C", data)

print("\n" + "=" * 78)
print("CONFIGURATION EFFECT  (C-config - base-config, holding geometry fixed)")
print("=" * 78)
contrast("Euclidean: base_at_C-baseline", "base_at_C", "baseline", data)
contrast("hyperbolic: C - C_at_base", "C", "C_at_base", data)

print("\n" + "=" * 78)
print("THE PAPER'S HEADLINE (confounds both)")
print("=" * 78)
contrast("C - baseline", "C", "baseline", data)

if not full:
    print("\n[partial: some arms incomplete, treat as provisional]")
