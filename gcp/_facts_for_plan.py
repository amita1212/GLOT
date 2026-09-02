#!/usr/bin/env python
"""Facts needed for the plan and the paper update:
   1. cost of the n=65 factorial (sizes the corrective rerun)
   2. MTEB Table-3 completeness per arm, per actual MTEB task
"""
import csv
import os
import collections

ROOT = os.path.expanduser("~/glot")

# ---------------------------------------------------------------- 1. factorial
f = os.path.join(ROOT, "results", "factorial_geom_cola.csv")
if os.path.exists(f):
    rs = [r for r in csv.DictReader(open(f, encoding="utf-8", errors="ignore"))
          if r.get("stage") == "confirm"]
    hrs = sum(float(r["elapsed_sec"]) for r in rs if r.get("elapsed_sec")) / 3600.0
    cells = collections.Counter(r["arm"] for r in rs)
    print(f"=== n=65 factorial actual cost ===")
    print(f"  {len(rs)} confirm runs, {hrs:.1f} GPU-h  ({hrs/max(len(rs),1)*60:.1f} min/run)")
    print(f"  cells: {dict(cells)}")
    print(f"  -> corrective rerun (same 260 runs) ~= {hrs:.1f} GPU-h\n")

# ------------------------------------------------------------------- 2. MTEB
m = os.path.join(ROOT, "results", "mteb_table3.csv")
rows = list(csv.DictReader(open(m, encoding="utf-8", errors="ignore")))
print(f"=== mteb_table3.csv: {len(rows)} rows ===")
print(f"  columns: {list(rows[0].keys())}\n")

# the per-MTEB-task scores must live somewhere other than the 'task' column
for r in rows:
    if r.get("task") == "mteb":
        for k, v in r.items():
            if v and k not in ("run_key", "detail"):
                pass
        break

# completeness by model x arm, counting distinct seeds on the 'mteb' rows
seen = collections.defaultdict(set)
for r in rows:
    if r.get("task") != "mteb":
        continue
    mm = (r.get("model") or "?").split("/")[-1]
    seen[(mm, r.get("arm"))].add(r.get("seed"))

print("=== completeness (mteb evaluation rows), by model x arm ===")
for (mm, a), seeds in sorted(seen.items()):
    flag = "COMPLETE" if len(seeds) >= 15 else ""
    print(f"  {mm:<32} {str(a):<16} seeds={len(seeds):<3} {flag}")

# which per-task columns exist on an mteb row?
sample = next((r for r in rows if r.get("task") == "mteb"), None)
if sample:
    print("\n=== a sample 'mteb' row's non-empty fields ===")
    for k, v in sample.items():
        if v and len(str(v)) < 200:
            print(f"  {k:<24} = {v}")
