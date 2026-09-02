#!/usr/bin/env python
"""Full MTEB Table-3 block: paired differences vs baseline, per arm per task.

Paired t-test beside the exact two-sided sign test, both required to agree,
Bonferroni over the whole complete arm-by-task family for the backbone.
"""
import csv
import os
import math
import collections
from statistics import mean, stdev

try:
    from scipy import stats as sp
except ImportError:
    sp = None


def sign_test(diffs):
    """Exact two-sided sign test; zeros discarded."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    c = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * c / (2 ** n)), pos, neg


def ttest(diffs):
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m, s = mean(diffs), stdev(diffs)
    if s == 0:
        return float("inf") if m else 0.0, 0.0
    t = m / (s / math.sqrt(n))
    if sp:
        return t, float(sp.t.sf(abs(t), n - 1) * 2)
    return t, float("nan")


ROOT = os.path.expanduser("~/glot")
rows = [r for r in csv.DictReader(
    open(os.path.join(ROOT, "results", "mteb_table3.csv"), encoding="utf-8",
         errors="ignore")) if r.get("task") == "mteb"]

MODEL = "bert-base-uncased"
score = {}
for r in rows:
    if r.get("model") != MODEL:
        continue
    try:
        score[(r["arm"], r["mteb_task"], int(r["seed"]))] = float(r["mteb_score"])
    except (ValueError, KeyError):
        pass

arms = sorted({a for a, _, _ in score})
tasks = sorted({t for _, t, _ in score})
base = "baseline"
others = [a for a in arms if a != base]

# family size = complete arm-by-task contrasts
family = 0
results = {}
for a in others:
    for t in tasks:
        seeds = sorted({s for (aa, tt, s) in score
                        if aa == a and tt == t} &
                       {s for (aa, tt, s) in score
                        if aa == base and tt == t})
        if len(seeds) < 15:
            continue
        family += 1
alpha = 0.05 / family if family else float("nan")
print(f"model={MODEL}  arms={len(others)} tasks={len(tasks)}")
print(f"complete arm-by-task contrasts m={family}  ->  alpha=0.05/{family}={alpha:.5f}\n")

print(f"{'task':<28} {'arm':<16} {'base':>7} {'delta':>8} {'t':>8} {'sign p':>9} {'p/n':>7}  mark")
for t in tasks:
    bmean = mean([v for (a, tt, s), v in score.items() if a == base and tt == t]) * 100
    for a in others:
        seeds = sorted({s for (aa, tt, s) in score if aa == a and tt == t} &
                       {s for (aa, tt, s) in score if aa == base and tt == t})
        if len(seeds) < 15:
            continue
        d = [(score[(a, t, s)] - score[(base, t, s)]) * 100 for s in seeds]
        tv, tp = ttest(d)
        sp_, pos, neg = sign_test(d)
        mark = ""
        if not math.isnan(tp) and tp < 0.05 and sp_ < 0.05:
            mark = "dagger"
            if tp < alpha and sp_ < alpha:
                mark = "DDAGGER"
        print(f"{t:<28} {a:<16} {bmean:>7.2f} {mean(d):>+8.2f} {tv:>8.2f} "
              f"{sp_:>9.5f} {pos:>3}/{neg:<3} {mark}")
    print()
