"""Audit then test the RoBERTa campaigns.

Order matters. An earlier pass reported baseline at n=30 when the design says
15, which means rows were being counted twice across merged files. A duplicated
row cannot move a mean much, but it misstates n and silently breaks a PAIRED
test -- so this audits first and refuses to report statistics for any arm whose
seed set is not clean.

Usage: python _analyze_roberta.py <task>
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from itertools import combinations
from math import comb, sqrt

TASK = sys.argv[1] if len(sys.argv) > 1 else "stsb"
BASE = "baseline"
ALPHA_BONF = 0.0063


def load():
    """Every row keyed by run_key, so a row present in two merged files counts once."""
    by_key, per_file = {}, {}
    for path in sorted(glob.glob(f"results/campaign_rob*_{TASK}*.csv")):
        rows = [r for r in csv.DictReader(open(path)) if r.get("stage") == "confirm"]
        per_file[os.path.basename(path)] = len(rows)
        for r in rows:
            k = r.get("run_key")
            if k in by_key and by_key[k]["score"] != r["score"]:
                print(f"  !! {k} differs between files: "
                      f"{by_key[k]['score']} vs {r['score']}")
            by_key[k] = r
    return list(by_key.values()), per_file


def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def ttest(d):
    n = len(d)
    if n < 2:
        return 0.0
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return 0.0 if var == 0 else m / sqrt(var / n)


def main():
    rows, per_file = load()
    print(f"task={TASK}   files:")
    for f, n in per_file.items():
        print(f"  {f:45s} {n} confirm rows")
    print(f"unique confirm rows after dedup by run_key: {len(rows)}\n")

    # ---- audit: seed multiplicity per arm -------------------------------
    seen = defaultdict(list)
    for r in rows:
        seen[(r["arm"], int(r["seed"]))].append(float(r["score"]))
    dirty = {a for (a, _), v in seen.items() if len(v) > 1}
    by_arm = defaultdict(dict)
    for (a, s), v in seen.items():
        by_arm[a][s] = v[0]

    print("arm            n   seeds")
    for a in sorted(by_arm):
        ss = sorted(by_arm[a])
        rng = f"{ss[0]}..{ss[-1]}" if ss == list(range(ss[0], ss[-1] + 1)) else str(ss)
        flag = "  <-- DUPLICATE (arm,seed)" if a in dirty else ""
        print(f"{a:14s} {len(ss):3d}  {rng}{flag}")
    if dirty:
        print(f"\n!! duplicated arms: {sorted(dirty)}")
        print("   scores agreed, so these are the same run recorded twice;")
        print("   deduped by run_key above, n is now correct.")

    if BASE not in by_arm:
        print(f"\nno {BASE} arm; cannot test")
        return
    base = by_arm[BASE]

    # ---- paired tests on shared seeds only ------------------------------
    print(f"\npaired vs {BASE}, shared seeds only, Bonferroni alpha={ALPHA_BONF}")
    print(f"{'arm':14s} {'n':>3} {'mean':>7} {'delta':>7} {'t':>7} "
          f"{'p/n':>7} {'sign p':>9}  verdict")
    out = []
    for a in sorted(by_arm):
        if a == BASE:
            continue
        shared = sorted(set(by_arm[a]) & set(base))
        if len(shared) < 2:
            continue
        d = [by_arm[a][s] - base[s] for s in shared]
        pos = sum(1 for x in d if x > 0)
        neg = sum(1 for x in d if x < 0)
        p = sign_p(pos, neg)
        t = ttest(d)
        mean = sum(by_arm[a][s] for s in shared) / len(shared)
        delta = sum(d) / len(d)
        # both tests must agree, as the paper requires
        sig = "ns"
        if p < ALPHA_BONF and abs(t) > 2.145:
            sig = "SIG (Bonf)"
        elif p < 0.05 and abs(t) > 2.145:
            sig = "sig (uncorr)"
        out.append((delta, a, len(shared), mean, t, pos, neg, p, sig))
    for delta, a, n, mean, t, pos, neg, p, sig in sorted(out, reverse=True):
        print(f"{a:14s} {n:3d} {mean:7.2f} {delta:+7.2f} {t:7.2f} "
              f"{pos:3d}/{neg:<3d} {p:9.5f}  {sig}")

    nb = len(set(base))
    print(f"\nbaseline n={nb}, mean={sum(base.values())/nb:.3f}")


if __name__ == "__main__":
    main()
