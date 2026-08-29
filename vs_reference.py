#!/usr/bin/env python3
"""Paired comparison against an ARBITRARY reference arm, not just the baseline.

WHY THIS EXISTS
    Every arm is normally differenced against `baseline`. But on CoLA the
    `no_graph` control ALSO beats the baseline (+0.833). If simply perturbing the
    pooler is worth ~+0.8, then Stage C's +1.42 is not +1.42 of "hyperbolic
    geometry" -- the honest quantity is C minus no_graph.

    This project's own rule: no arm counts as a win until it beats the arm that
    deletes the graph entirely. That comparison needs the reference to be
    configurable, which paired_analysis.py does not allow.

    usage: vs_reference.py <csv> [reference_arm]
"""
import csv
import sys
from collections import defaultdict

T_CRIT = {14: 2.145, 13: 2.160, 12: 2.179, 11: 2.201, 10: 2.228,
          9: 2.262, 8: 2.306, 7: 2.365, 6: 2.447, 5: 2.571, 4: 2.776,
          3: 3.182, 2: 4.303, 1: 12.706}


def nCr(n, r):
    from math import comb
    return comb(n, r)


def sign_p(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = max(pos, neg)
    tail = sum(nCr(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path):
    arms = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r.get("stage") != "confirm":
            continue
        try:
            arms[r["arm"]][int(r["seed"])] = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return arms


def main():
    path = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else "no_graph"
    arms = load(path)
    if ref not in arms:
        print(f"reference arm {ref!r} not in {path}")
        return

    base = arms[ref]
    seeds = sorted(base)
    print(f"\n===== {path} =====")
    print(f"  reference = {ref}   (n={len(seeds)} seeds)")

    # Absolute confirmation means, so the reader can see where everything sits.
    print(f"\n  {'arm':<10} {'mean':>8}")
    for a in sorted(arms, key=lambda k: -sum(arms[k].values()) / max(1, len(arms[k]))):
        v = list(arms[a].values())
        if v:
            print(f"  {a:<10} {sum(v)/len(v):>8.3f}")

    print(f"\n  paired vs {ref}:")
    print(f"  {'arm':<10} {'n':>3} {'mean':>8} {'SE':>6} {'t':>7} {'signP':>9} {'pos/neg':>9}")
    rows = []
    for arm, byseed in arms.items():
        if arm == ref:
            continue
        common = [s for s in seeds if s in byseed]
        if len(common) < 2:
            continue
        d = [byseed[s] - base[s] for s in common]
        n = len(d)
        m = sum(d) / n
        var = sum((x - m) ** 2 for x in d) / (n - 1)
        se = (var / n) ** 0.5
        t = m / se if se > 0 else 0.0
        rows.append((m, arm, n, se, t, sign_p(d),
                     sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)))
    for m, arm, n, se, t, sp, pos, neg in sorted(rows, reverse=True):
        print(f"  {arm:<10} {n:>3} {m:>+8.3f} {se:>6.3f} {t:>7.2f} {sp:>9.5f} "
              f"{pos:>4}/{neg:<4}")


if __name__ == "__main__":
    main()
