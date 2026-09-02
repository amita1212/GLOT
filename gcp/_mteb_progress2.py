#!/usr/bin/env python
"""MTEB Table-3 campaign progress, by backbone x arm x task."""
import csv
import collections
import os

PATH = os.path.expanduser("~/glot/results/mteb_table3.csv")
rows = list(csv.DictReader(open(PATH, encoding="utf-8", errors="ignore")))
print(f"{len(rows)} rows")
if not rows:
    raise SystemExit

cols = rows[0].keys()
mk = next((c for c in ("model", "backbone", "model_name") if c in cols), None)
ak = next((c for c in ("arm", "method") if c in cols), None)
tk = next((c for c in ("task", "dataset", "mteb_task") if c in cols), None)
sk = next((c for c in ("seed",) if c in cols), None)
print(f"columns used: model={mk} arm={ak} task={tk} seed={sk}\n")

seen = collections.defaultdict(set)
for r in rows:
    m = (r.get(mk) or "?").split("/")[-1]
    seen[(m, r.get(ak))].add((r.get(tk), r.get(sk)))

tasks = sorted({r.get(tk) for r in rows if r.get(tk)})
print(f"{len(tasks)} distinct tasks: {', '.join(tasks)}\n")

models = []
for m, a in seen:
    if m not in models:
        models.append(m)

for m in models:
    print(f"=== {m} ===")
    for (mm, a), cells in sorted(seen.items()):
        if mm != m:
            continue
        seeds = {s for _, s in cells}
        ntask = len({t for t, _ in cells})
        full = len(cells)
        print(f"  {str(a):<16} tasks={ntask:<2} seeds={len(seeds):<3} "
              f"cells={full:<4} {'COMPLETE' if full >= 7 * 15 else ''}")
    print()
