#!/usr/bin/env python3
"""Is the reproduction gap a real gap, or a difference of ESTIMATOR?

WHY THIS EXISTS
    We report confirmation-stage MEANS over 15 seeds. The published GLOT numbers
    come from Table 8, which reports the BEST tau per task on the dev split that
    is also the reported split -- i.e. a MAX over a grid, at (as far as the paper
    states) a single seed.

    Mean-over-seeds and max-over-grid are not the same quantity. Before blaming
    an implementation difference we must check whether OUR OWN runs, re-scored
    with THEIR estimator, land on THEIR number. If they do, there is no
    reproduction gap to explain -- only a selection-bias difference.

    usage: estimator_gap.py <csv> <published_value>
"""
import csv
import sys
from collections import defaultdict


def load(path):
    """Return (confirm[arm][seed] = score, tune[arm] = [scores])."""
    confirm = defaultdict(dict)
    tune = defaultdict(list)
    for r in csv.DictReader(open(path)):
        arm = r.get("arm")
        try:
            s = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if r.get("stage") == "confirm":
            try:
                confirm[arm][int(r["seed"])] = s
            except (KeyError, TypeError, ValueError):
                pass
        elif r.get("stage") == "tune":
            tune[arm].append(s)
    return confirm, tune


def main():
    path = sys.argv[1]
    published = float(sys.argv[2]) if len(sys.argv) > 2 else None
    confirm, tune = load(path)

    print(f"\n===== {path} =====")
    if published is not None:
        print(f"  published GLOT = {published:.2f}\n")

    hdr = (f"  {'arm':<12}{'n':>4}{'mean':>9}{'std':>7}{'min':>8}{'max':>8}"
           f"{'maxTune':>9}{'nTune':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for arm, byseed in confirm.items():
        v = list(byseed.values())
        if not v:
            continue
        n = len(v)
        mean = sum(v) / n
        std = (sum((x - mean) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
        t = tune.get(arm, [])
        rows.append((arm, n, mean, std, min(v), max(v),
                     max(t) if t else float("nan"), len(t)))

    for arm, n, mean, std, lo, hi, mt, nt in sorted(rows, key=lambda r: -r[2]):
        print(f"  {arm:<12}{n:>4}{mean:>9.2f}{std:>7.2f}{lo:>8.2f}{hi:>8.2f}"
              f"{mt:>9.2f}{nt:>7}")

    if published is None:
        return

    base = next((r for r in rows if r[0] == "baseline"), None)
    if base is None:
        return
    _, n, mean, std, lo, hi, mt, nt = base

    print("\n  baseline, published minus each estimator:")
    print(f"    mean over {n} seeds        {mean:8.2f}   delta {published - mean:+7.2f}")
    print(f"    max  over {n} seeds        {hi:8.2f}   delta {published - hi:+7.2f}")
    if nt:
        print(f"    max  over {nt} tune runs   {mt:8.2f}   delta {published - mt:+7.2f}")
    print(f"\n    seed spread (max-min) = {hi - lo:.2f}"
          f"   |  seed std = {std:.2f}")
    if std > 0:
        print(f"    published sits {(published - mean) / std:+.2f} seed-std "
              f"above our mean")


if __name__ == "__main__":
    main()
