#!/usr/bin/env bash
# ETA for the wide sweep, from MEASURED per-run time.
#
# The old 201 s/run figure came from runs pinned at num_layers=2,
# gat_hidden_dim=128. The wide grid also draws K=4 and h=256, which are strictly
# more expensive, so the ETA must be based on wide runs only.
cd "$(dirname "$0")" || exit 1
/home/t-amitalfasi/glotenv/bin/python - <<'PYEOF'
import csv, glob, os, statistics

ARMS, TRIALS, SEEDS, TASKS = 9, 40, 15, 2
per_task = ARMS * (TRIALS + SEEDS)
total_runs = per_task * TASKS

times, done = [], 0
for f in glob.glob("results/_smoke_wide.csv") + glob.glob("results/campaign_wide_*.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("elapsed_sec"):
            try:
                times.append(float(r["elapsed_sec"]))
            except ValueError:
                pass
        if "campaign_wide" in f:
            done += 1

print(f"planned: {ARMS} arms x ({TRIALS} trials + {SEEDS} seeds) x {TASKS} tasks"
      f" = {total_runs} runs")
print(f"         ({per_task} runs per task)")
print(f"completed in the real campaign: {done}")
print()

if not times:
    print("no timed runs yet -- cannot estimate")
    raise SystemExit

med = statistics.median(times)
print(f"measured per-run seconds (n={len(times)}): "
      f"min={min(times):.0f} median={med:.0f} mean={statistics.mean(times):.0f} "
      f"max={max(times):.0f}")
print()

for label, s in (("optimistic (min)", min(times)),
                 ("expected  (median)", med),
                 ("pessimistic (max)", max(times))):
    rem = (total_runs - done) * s
    print(f"  {label:<20} {s:6.0f} s/run -> {rem/3600:6.1f} h "
          f"({rem/86400:4.1f} days) remaining")

print()
print("per-stage, at the median:")
for name, n in (("stsb tune", ARMS * TRIALS), ("stsb confirm", ARMS * SEEDS),
                ("cola tune", ARMS * TRIALS), ("cola confirm", ARMS * SEEDS)):
    print(f"  {name:<14} {n:4d} runs  {n*med/3600:6.1f} h")
PYEOF
