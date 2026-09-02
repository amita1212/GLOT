#!/usr/bin/env python
"""Every TinyLlama run on this machine: counts, seeds, and GPU-time invested."""
import csv
import glob
import os
import collections

ROOT = os.path.expanduser("~/glot")
rows_by_file = {}
for f in sorted(glob.glob(os.path.join(ROOT, "results", "*.csv"))):
    try:
        rs = list(csv.DictReader(open(f, encoding="utf-8", errors="ignore")))
    except OSError:
        continue
    keep = [r for r in rs
            if "tinyllama" in (r.get("model") or r.get("model_name") or "").lower()]
    if keep:
        rows_by_file[os.path.basename(f)] = keep

if not rows_by_file:
    print("no TinyLlama rows on this machine")
    raise SystemExit

# find a duration column once
sample = next(iter(rows_by_file.values()))[0]
tcol = next((c for c in ("elapsed_sec", "seconds", "secs", "wall", "elapsed",
                         "duration", "runtime", "time_s") if c in sample), None)

print(f"{'file':<34} {'task':<7} {'stage':<8} {'runs':>5} {'arms':>5} {'seeds':>6} {'GPU-h':>7}")
print("-" * 80)
tot_runs = tot_h = 0
grand = collections.defaultdict(int)
for fn, rs in rows_by_file.items():
    groups = collections.defaultdict(list)
    for r in rs:
        groups[(r.get("task", "?"), r.get("stage", "?"))].append(r)
    for (task, stage), g in sorted(groups.items()):
        arms = {r.get("arm") for r in g}
        seeds = {r.get("seed") for r in g}
        hrs = 0.0
        if tcol:
            for r in g:
                try:
                    hrs += float(r[tcol]) / 3600.0
                except (TypeError, ValueError, KeyError):
                    pass
        print(f"{fn:<34} {task:<7} {stage:<8} {len(g):>5} {len(arms):>5} "
              f"{len(seeds):>6} {hrs:>7.1f}")
        tot_runs += len(g)
        tot_h += hrs
        grand[stage] += len(g)

print("-" * 80)
print(f"{'TOTAL':<34} {'':<7} {'':<8} {tot_runs:>5} {'':>5} {'':>6} {tot_h:>7.1f}")
print(f"\nby stage: " + ", ".join(f"{k}={v}" for k, v in sorted(grand.items())))
print(f"duration column used: {tcol}")
