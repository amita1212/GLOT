#!/usr/bin/env python3
"""Stress-test results in the shape of GLOT's Table 7 (accuracy vs distractor ratio).

The original reports one row per pooling method per backbone at four distractor
ratios. Ours has more seeds and more GLOT variants but only one backbone, so we
report mean +/- std across seeds at each ratio.
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "results/stress_followup.csv"
rows = list(csv.DictReader(open(path)))

acc = defaultdict(list)
ratios = set()
for r in rows:
    try:
        a = float(r["acc"]) * (100.0 if float(r["acc"]) <= 1.0 else 1.0)
    except (KeyError, TypeError, ValueError):
        continue
    ratio = r.get("distractor_ratio")
    ratios.add(ratio)
    acc[(r.get("arm"), ratio)].append(a)

ratios = sorted(ratios, key=float)
arms = sorted({a for a, _ in acc})

print(f"\n  {path}   n_seeds per cell = "
      f"{len(acc[(arms[0], ratios[0])]) if arms else 0}\n")
hdr = f"  {'arm':<14}" + "".join(f"{r+' dist':>16}" for r in ratios)
print(hdr)
print("  " + "-" * (len(hdr) - 2))

order = []
for arm in arms:
    v = acc[(arm, ratios[-1])]
    order.append((sum(v) / len(v) if v else -1, arm))
for _, arm in sorted(order, reverse=True):
    line = f"  {arm:<14}"
    for r in ratios:
        v = acc[(arm, r)]
        if not v:
            line += f"{'--':>16}"
            continue
        m = sum(v) / len(v)
        s = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        line += f"{m:>10.1f}+-{s:<4.1f}"
    print(line)

print("\n  GLOT paper Table 7, BERT row, for reference:")
print(f"  {'[CLS]':<14}{'70.8':>16}{'58.2':>16}{'57.2':>16}{'67.6':>16}")
print(f"  {'Mean':<14}{'68.0':>16}{'58.6':>16}{'64.2':>16}{'53.4':>16}")
print(f"  {'Max':<14}{'57.4':>16}{'50.8':>16}{'51.6':>16}{'50.4':>16}")
print(f"  {'AdaPool':<14}{'91.4':>16}{'78.8':>16}{'65.6':>16}{'61.6':>16}")
print(f"  {'GLOT':<14}{'97.2':>16}{'97.0':>16}{'97.8':>16}{'98.8':>16}")
print()
