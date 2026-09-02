"""Can the structural arms be reported as paired deltas, or only as absolutes?

main.tex reported them as tuning maxima and explicitly said they are "not
results". But results/campaign_struct_cola.csv also holds confirm rows. If that
campaign contains a baseline at the same confirm seeds, we can report proper
paired tests -- strictly better than tuning maxima. If it does not, the arms
fall under the same cross-campaign rule as the decoder's B and C.
"""
import csv
import collections
from scipy import stats

rows = list(csv.DictReader(open("results/campaign_struct_cola.csv", encoding="utf-8")))
print(f"rows {len(rows)}")

by = collections.defaultdict(lambda: collections.defaultdict(dict))
seen = set()
for r in rows:
    k = r.get("run_key")
    if k and k in seen:
        continue
    seen.add(k)
    try:
        by[r["stage"]][r["arm"]][int(r["seed"])] = float(r["score"])
    except (ValueError, KeyError):
        pass

for stage in sorted(by):
    print(f"\n=== stage={stage} ===")
    for arm in sorted(by[stage]):
        s = by[stage][arm]
        v = list(s.values())
        print(f"  {arm:<10} n={len(v):<3} seeds={sorted(s)[:16]}"
              f"  mean={sum(v)/len(v):6.2f}")

conf = by.get("confirm", {})
print("\n=== is there a baseline in the confirm stage of THIS campaign? ===")
if "baseline" in conf:
    print("  YES -> paired deltas are admissible")
    base = conf["baseline"]
    for arm in sorted(conf):
        if arm == "baseline":
            continue
        common = sorted(set(conf[arm]) & set(base))
        if len(common) < 3:
            print(f"  {arm:<10} only {len(common)} shared seeds")
            continue
        x = [conf[arm][s] for s in common]
        y = [base[s] for s in common]
        d = [a - b for a, b in zip(x, y)]
        m = sum(d) / len(d)
        t, pt = stats.ttest_rel(x, y)
        pos = sum(1 for z in d if z > 0)
        neg = sum(1 for z in d if z < 0)
        ps = min(1.0, 2 * stats.binom.cdf(min(pos, neg), pos + neg, 0.5)) if pos + neg else 1.0
        print(f"  {arm:<10} d={m:+6.2f}  t={t:+6.2f} p={pt:.4f}  sign p={ps:.4f}  "
              f"{pos}/{neg}  n={len(common)}")
else:
    print("  NO baseline in confirm stage.")
    print("  -> the confirm rows cannot be differenced within this campaign.")
    print("  -> report absolute means only, exactly as for the decoder B/C arms.")
    for arm in sorted(conf):
        v = list(conf[arm].values())
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
        print(f"     {arm:<10} n={len(v):<3} mean={mean:6.2f}  sd={sd:5.2f}")
