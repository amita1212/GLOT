"""Dump a campaign CSV as a compact table, arm-by-arm, with density.

Exists because inlining python through `gcloud compute ssh --command` on Windows
mangles quotes and backslashes (plink re-parses the string). Ship a file instead.
"""
import csv
import sys

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
print(f"n rows = {len(rows)}")

cols = ["arm", "stage", "seed", "score", "density"]
print(" ".join(f"{c:<10}" for c in cols) + " detail")
print("-" * 110)
for r in rows:
    line = " ".join(f"{str(r.get(c, ''))[:10]:<10}" for c in cols)
    print(line + " " + str(r.get("detail", ""))[:70])

print()
print("BEST PER ARM (tuning stage, seed 42):")
best = {}
for r in rows:
    if r.get("stage") != "tune":
        continue
    try:
        s = float(r["score"])
    except (TypeError, ValueError):
        continue
    a = r["arm"]
    if a not in best or s > best[a][0]:
        best[a] = (s, r.get("density", ""), r.get("detail", ""))
for a, (s, d, det) in sorted(best.items(), key=lambda kv: -kv[1][0]):
    n = sum(1 for r in rows if r.get("arm") == a and r.get("stage") == "tune")
    print(f"  {a:<10} best={s:>7.2f}  n_trials={n:<3} density={str(d)[:7]:<8} {det[:60]}")
