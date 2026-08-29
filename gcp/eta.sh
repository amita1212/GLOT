#!/usr/bin/env bash
# ETA for a running campaign: counts finished runs, measures the observed
# per-run cost, and projects the remaining wall time.
set -u
cd /home/t-amitalfasi/glot
CSV="${1:-results/campaign_stress_hard.csv}"
LOG="${2:-logs/campaign_stress_hard.log}"

echo "== process =="
pgrep -fa campaign.py | head -3 || echo "NOT RUNNING"
echo
echo "== last log lines =="
tail -4 "$LOG" 2>/dev/null
echo
~/glotenv/bin/python - "$CSV" <<'PY'
import csv, os, sys
from collections import defaultdict
p = sys.argv[1]
if not os.path.exists(p):
    print("no CSV yet"); raise SystemExit
rows = list(csv.DictReader(open(p)))
if not rows:
    print("CSV empty"); raise SystemExit

# Trial budget actually realised per arm = size of its sampled config set,
# capped by the space size. Recomputed here from campaign.py to stay in sync.
sys.path.insert(0, os.path.dirname(os.path.abspath(p)) + "/..")
sys.path.insert(0, "/home/t-amitalfasi/glot")
from campaign import ARMS, sample_configs

TUNE_SEED, N_TRIALS, N_CONFIRM = 42, 10, 3
planned_tune = {a: len(sample_configs(a, N_TRIALS, TUNE_SEED)) for a in ARMS}
total_planned = sum(planned_tune.values()) + len(ARMS) * N_CONFIRM

done_tune = defaultdict(int)
secs = defaultdict(list)
for r in rows:
    if r.get("stage") == "tune":
        done_tune[r["arm"]] += 1
    try:
        secs[r["arm"]].append(float(r["elapsed_sec"]))
    except (KeyError, ValueError):
        pass

done = len(rows)
allsec = [s for v in secs.values() for s in v]
avg = sum(allsec) / len(allsec)

# Project remaining using each arm's OWN observed cost where available -- the
# Stage C arms run noticeably slower than the cosine ones, so a single global
# average would understate the tail.
remaining = 0.0
for arm in ARMS:
    per = (sum(secs[arm]) / len(secs[arm])) if secs.get(arm) else avg
    left_tune = max(0, planned_tune[arm] - done_tune[arm])
    left_conf = N_CONFIRM if done_tune[arm] < planned_tune[arm] else max(
        0, N_CONFIRM - sum(1 for r in rows
                           if r["arm"] == arm and r.get("stage") == "confirm"))
    remaining += (left_tune + left_conf) * per

print(f"planned runs      : {total_planned}")
print(f"completed         : {done}")
print(f"observed avg/run  : {avg:.0f}s   (min {min(allsec):.0f}s, max {max(allsec):.0f}s)")
print(f"projected remain  : {remaining/3600:.2f} h")
print()
print(f"{'arm':<10} {'tune done/plan':>15} {'avg s/run':>10}")
for arm in ARMS:
    per = (sum(secs[arm]) / len(secs[arm])) if secs.get(arm) else float('nan')
    print(f"{arm:<10} {done_tune[arm]:>7}/{planned_tune[arm]:<7} {per:>10.0f}")
PY
