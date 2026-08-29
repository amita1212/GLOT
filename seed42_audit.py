#!/usr/bin/env python3
"""Which of our runs used the paper's seed (42), and what did they score?

The paper states: "For all experiments, we used a fixed random seed of 42."
So every published number is a SINGLE draw. If we have runs at seed 42 under
their config, that is an apples-to-apples comparison and the only fair one.
"""
import csv
import glob
import os
from collections import defaultdict

KEYS = ("tau", "tau_quantile", "graph_metric", "num_layers",
        "gat_hidden_dim", "lr", "arm", "task", "stage")

rows = []
for path in sorted(glob.glob("results/*.csv")):
    try:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if str(r.get("seed", "")).strip() != "42":
                    continue
                rows.append((os.path.basename(path), r))
    except Exception as e:
        print(f"  ! {path}: {e}")

if not rows:
    print("no seed-42 rows found")
    raise SystemExit

by_file = defaultdict(list)
for f, r in rows:
    by_file[f].append(r)

for f, rs in by_file.items():
    print(f"\n===== {f}  ({len(rs)} seed-42 rows) =====")
    for r in rs[:40]:
        desc = "  ".join(f"{k}={r[k]}" for k in KEYS
                         if r.get(k) not in (None, "", "nan"))
        score = r.get("score", "?")
        print(f"  score={score:>8}  {desc}")
    if len(rs) > 40:
        print(f"  ... {len(rs) - 40} more")
