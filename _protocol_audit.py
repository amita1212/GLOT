"""Audit the actual protocol of every campaign: trials tuned, seeds confirmed.

The paper claims "40 trials each, 15 seeds". That is true of the BERT wide
campaigns. It is NOT true of every table in the paper, and the appendix only
partly says so. This prints the ground truth per file per arm so the protocol
claims can be checked against what ran.
"""
import csv
import glob
import os
from collections import defaultdict

os.chdir(os.path.expanduser("~/glot"))

rows_by_file = {}
for p in sorted(glob.glob("results/*.csv")):
    try:
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"{p}: UNREADABLE {e}")
        continue
    if not rows or "arm" not in rows[0]:
        continue
    rows_by_file[p] = rows

print(f"{'file':46s} {'arms':>4s} {'tune/arm':>9s} {'seeds':>6s}  models/tasks")
print("=" * 108)
for p, rows in rows_by_file.items():
    tune = defaultdict(int)
    conf = defaultdict(set)
    for r in rows:
        arm = r.get("arm", "?")
        st = r.get("stage", "")
        if st == "tune":
            tune[arm] += 1
        elif st == "confirm":
            s = r.get("seed")
            if s not in (None, ""):
                conf[arm].add(int(float(s)))
    arms = sorted(set(tune) | set(conf))
    if not arms:
        continue
    tvals = sorted({tune.get(a, 0) for a in arms})
    svals = sorted({len(conf.get(a, ())) for a in arms})
    models = sorted({r.get("model", "?") for r in rows})
    tasks = sorted({r.get("task", "?") for r in rows})
    tstr = ",".join(str(v) for v in tvals)
    sstr = ",".join(str(v) for v in svals)
    mt = f"{'/'.join(m.split('/')[-1] for m in models)}  {','.join(tasks)}"
    print(f"{os.path.basename(p):46s} {len(arms):>4d} {tstr:>9s} {sstr:>6s}  {mt}")

print()
print("PER-ARM DETAIL for the campaigns the paper's main tables rest on")
print("=" * 108)
KEY = ["campaign_wide_cola.csv", "campaign_wide_stsb.csv",
       "campaign_wide_mrpc.csv", "campaign_wide_rte.csv",
       "campaign_decoder_stsb.csv", "campaign_roberta_cola.csv",
       "campaign_roberta_stsb.csv"]
for name in KEY:
    p = f"results/{name}"
    if p not in rows_by_file:
        print(f"\n{name}: NOT FOUND")
        continue
    rows = rows_by_file[p]
    tune = defaultdict(int)
    conf = defaultdict(set)
    for r in rows:
        arm = r.get("arm", "?")
        if r.get("stage") == "tune":
            tune[arm] += 1
        elif r.get("stage") == "confirm" and r.get("seed") not in (None, ""):
            conf[arm].add(int(float(r["seed"])))
    print(f"\n{name}")
    for arm in sorted(set(tune) | set(conf)):
        seeds = sorted(conf.get(arm, ()))
        contiguous = seeds == list(range(1, len(seeds) + 1))
        flag = "" if contiguous else f"  <-- seeds {seeds}"
        print(f"   {arm:12s} tune={tune.get(arm, 0):3d}  confirm_seeds={len(seeds):3d}{flag}")
