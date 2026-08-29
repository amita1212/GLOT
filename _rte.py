"""RTE wide campaign: paired deltas vs baseline and vs no_graph."""
import csv
from collections import defaultdict
from math import comb, sqrt

per = defaultdict(dict)
for r in csv.DictReader(open("results/campaign_wide_rte.csv", newline="",
                             encoding="utf-8")):
    if r.get("stage") != "confirm":
        continue
    try:
        per[r["arm"]][int(r["seed"])] = float(r["score"])
    except (ValueError, KeyError):
        pass

def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def paired(a, b):
    s = sorted(set(a) & set(b))
    d = [a[x] - b[x] for x in s]
    n = len(d)
    m = sum(d) / n
    sd = sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = m / (sd / sqrt(n)) if sd else float("inf")
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    return n, m, t, pos, neg, sign_p(pos, neg)

bl, ng = per["baseline"], per["no_graph"]
print(f"RTE (accuracy), n=15 paired seeds")
print(f"baseline mean = {sum(bl.values())/len(bl):.3f}")
print(f"no_graph mean = {sum(ng.values())/len(ng):.3f}")
print()
print(f"{'arm':10s} {'mean':>8s} {'vs base':>9s} {'t':>7s} {'p/n':>7s} "
      f"{'sign p':>8s} {'vs nograph':>11s}")
print("-" * 70)
out = []
for arm in per:
    if arm in ("baseline", "no_graph"):
        continue
    mean = sum(per[arm].values()) / len(per[arm])
    _, m, t, pos, neg, p = paired(per[arm], bl)
    _, mn, _, _, _, _ = paired(per[arm], ng)
    out.append((m, arm, mean, t, pos, neg, p, mn))
for m, arm, mean, t, pos, neg, p, mn in sorted(out, reverse=True):
    star = "  **" if p < 0.05 else ""
    print(f"{arm:10s} {mean:8.3f} {m:+9.2f} {t:7.2f} {pos:>3d}/{neg:<3d} "
          f"{p:8.4f} {mn:+11.2f}{star}")

_, m, t, pos, neg, p = paired(ng, bl)
print(f"{'no_graph':10s} {sum(ng.values())/len(ng):8.3f} {m:+9.2f} {t:7.2f} "
      f"{pos:>3d}/{neg:<3d} {p:8.4f}")

sds = []
for arm in ("baseline", "A", "C", "B", "AB", "ABC"):
    if arm in per:
        v = list(per[arm].values())
        mu = sum(v) / len(v)
        sds.append((arm, sqrt(sum((x-mu)**2 for x in v)/(len(v)-1))))
print()
print("seed sd:", "  ".join(f"{a}={s:.2f}" for a, s in sds))
