#!/usr/bin/env python3
"""Best score per task at the paper's seed 42, restricted to the paper's own grid.

Pooling every results/*.csv mixes score semantics (probe accuracies, stress
ratios) and produced a nonsense CoLA max of 97.70. Only the sweep_* and
repro_table8* files share the paper's metric convention, so restrict to those.
"""
import csv
import glob
import os
from collections import defaultdict

PUB = {"cola": 47.49, "stsb": 83.86, "mrpc": 82.58, "rte": 59.21}

best = defaultdict(lambda: (-1e9, "", ""))
n = defaultdict(int)

files = sorted(glob.glob("results/sweep_*.csv"))
files += sorted(glob.glob("results/repro_table8*.csv"))
files = [f for f in files if "OLD" not in f]

for p in files:
    for r in csv.DictReader(open(p)):
        if str(r.get("seed", "")).strip() != "42":
            continue
        task = r.get("task")
        try:
            s = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
        n[task] += 1
        if s > best[task][0]:
            cfg = ("tau=%s K=%s h=%s lr=%s jk=%s"
                   % (r.get("tau"), r.get("num_layers"),
                      r.get("gat_hidden_dim"), r.get("lr"), r.get("jk_mode")))
            best[task] = (s, cfg, os.path.basename(p))

print("\n  files: " + ", ".join(os.path.basename(f) for f in files))
print("\n  %-7s%6s%12s%11s%8s   %s"
      % ("task", "n@42", "ourBest@42", "published", "delta", "config"))
print("  " + "-" * 88)
for task in sorted(best):
    s, cfg, src = best[task]
    p = PUB[task]
    print("  %-7s%6d%12.2f%11.2f%+8.2f   %s" % (task, n[task], s, p, s - p, cfg))
print()
