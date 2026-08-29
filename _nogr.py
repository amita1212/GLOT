"""Compare every arm against the no_graph control on shared seeds."""
import csv
from collections import defaultdict
from math import comb, sqrt

def load(path):
    per = defaultdict(dict)
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r.get("stage") != "confirm":
            continue
        try:
            per[r["arm"]][int(r["seed"])] = float(r["score"])
        except (ValueError, KeyError):
            pass
    return per

def sign_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)

def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    sd = sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = m / (sd / sqrt(n)) if sd > 0 else float("inf")
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    return n, m, t, pos, neg, sign_p(pos, neg)

for task, path in [("CoLA", "results/campaign_wide_cola.csv"),
                   ("STS-B", "results/campaign_wide_stsb.csv"),
                   ("MRPC", "results/campaign_wide_mrpc.csv")]:
    per = load(path)
    if "no_graph" not in per or "baseline" not in per:
        print(f"{task}: no_graph or baseline missing\n")
        continue
    ng, bl = per["no_graph"], per["baseline"]
    print("=" * 78)
    print(f"{task}   baseline mean={sum(bl.values())/len(bl):.3f}   "
          f"no_graph mean={sum(ng.values())/len(ng):.3f}")
    print("=" * 78)
    print(f"{'arm':10s} {'mean':>8s} {'vs base':>9s} {'vs NOGRAPH':>11s} "
          f"{'p/n':>7s} {'sign p':>9s}")
    for arm in sorted(per):
        if arm == "no_graph":
            continue
        mean = sum(per[arm].values()) / len(per[arm])
        vb = paired(per[arm], bl)
        vn = paired(per[arm], ng)
        if not vn:
            continue
        _, mn, _, pos, neg, p = vn
        vbs = f"{vb[1]:+.2f}" if vb else "  --"
        star = "  <-- fails: no better than having NO graph" if p > 0.20 else ""
        print(f"{arm:10s} {mean:8.3f} {vbs:>9s} {mn:+11.2f} "
              f"{pos:>3d}/{neg:<3d} {p:9.4f}{star}")
    print()
