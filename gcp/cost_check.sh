#!/usr/bin/env bash
# Per-run cost comparison: BERT campaigns vs the TinyLlama decoder smoke.
cd "$(dirname "$0")" || exit 1
PY=/home/t-amitalfasi/glotenv/bin/python

$PY - <<'PYEOF'
import csv, glob, os, statistics

files = sorted(glob.glob("results/campaign_*.csv")) + ["results/_smoke_decoder.csv"]
print(f"{'file':<42} {'n':>4} {'mean_s':>8} {'med_s':>8} {'max_s':>8}")
for f in files:
    if not os.path.exists(f):
        continue
    try:
        rows = list(csv.DictReader(open(f)))
    except Exception:
        continue
    v = []
    for r in rows:
        x = r.get("elapsed_sec")
        if x:
            try:
                v.append(float(x))
            except ValueError:
                pass
    if v:
        print(f"{os.path.basename(f):<42} {len(v):>4} {statistics.mean(v):>8.1f} "
              f"{statistics.median(v):>8.1f} {max(v):>8.1f}")

print()
print("=== BUDGET ARITHMETIC FOR THE DECODER SWEEP ===")
try:
    rows = list(csv.DictReader(open("results/_smoke_decoder.csv")))
    v = [float(r["elapsed_sec"]) for r in rows if r.get("elapsed_sec")]
    per = statistics.median(v)
except Exception:
    per = 924.0
    print("(no smoke timings on disk; using observed median 924 s)")

for arms in (9, 7, 6):
    for tasks in (2, 1):
        for trials, seeds in ((10, 15), (6, 10), (6, 5)):
            runs = arms * tasks * (trials + seeds)
            hrs = runs * per / 3600
            print(f"  arms={arms} tasks={tasks} trials={trials:>2} confirm={seeds:>2} "
                  f"-> {runs:>4} runs, {hrs:>6.1f} h  ({hrs/24:>4.1f} days)")
PYEOF
