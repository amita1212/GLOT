"""Stage A audit: does the Poincare graph help, on any task, by any comparison?"""
import csv
import os
from collections import defaultdict
from math import comb, sqrt

CAMPAIGNS = [
    ("CoLA  (BERT)",  "results/campaign_wide_cola.csv"),
    ("STS-B (BERT)",  "results/campaign_wide_stsb.csv"),
    ("MRPC  (BERT)",  "results/campaign_wide_mrpc.csv"),
    ("RTE   (BERT)",  "results/campaign_wide_rte.csv"),
    ("STS-B (TinyLlama)", "results/campaign_decoder_stsb.csv"),
    ("CoLA  (RoBERTa)",   "results/campaign_rob_cola.csv"),
    ("STS-B (RoBERTa)",   "results/campaign_rob_stsb.csv"),
]

def load(path):
    per = defaultdict(dict)
    if not os.path.exists(path):
        return per
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
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def paired(a, b):
    s = sorted(set(a) & set(b))
    if len(s) < 2:
        return None
    d = [a[x] - b[x] for x in s]
    n = len(d)
    m = sum(d) / n
    sd = sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    t = m / (sd / sqrt(n)) if sd else float("inf")
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    return m, t, pos, neg, sign_p(pos, neg)

print("=" * 92)
print("STAGE A: vs the tuned baseline, and vs deleting the graph entirely")
print("=" * 92)
print(f"{'setting':20s} {'A mean':>8s} {'vs baseline':>26s} {'vs no_graph':>26s}")
print("-" * 92)

vs_base_signs, vs_ng_signs = [], []
for label, path in CAMPAIGNS:
    per = load(path)
    if "A" not in per or "baseline" not in per:
        print(f"{label:20s}  (Stage A not run)")
        continue
    amean = sum(per["A"].values()) / len(per["A"])
    vb = paired(per["A"], per["baseline"])
    vn = paired(per["A"], per["no_graph"]) if "no_graph" in per else None
    sb = f"{vb[0]:+6.2f} {vb[2]:2d}/{vb[3]:<2d} p={vb[4]:.3f}" if vb else "--"
    sn = f"{vn[0]:+6.2f} {vn[2]:2d}/{vn[3]:<2d} p={vn[4]:.3f}" if vn else "  (no control run)"
    if vb:
        vs_base_signs.append(vb[0] > 0)
    if vn:
        vs_ng_signs.append(vn[0] > 0)
    star = ""
    if vb and vb[4] < 0.05 and abs(vb[1]) > 2.1:
        star = "  SIGNIF"
    print(f"{label:20s} {amean:8.2f} {sb:>26s} {sn:>26s}{star}")

print()
print("-" * 92)
p = sum(vs_base_signs)
n = len(vs_base_signs)
print(f"vs baseline : Stage A nominally positive in {p} of {n} settings; "
      f"sign test across settings p={sign_p(p, n - p):.3f}")
p2 = sum(vs_ng_signs)
n2 = len(vs_ng_signs)
print(f"vs no_graph : Stage A nominally positive in {p2} of {n2} settings; "
      f"sign test across settings p={sign_p(p2, n2 - p2):.3f}")
print()
print("Significant results for Stage A anywhere (both tests, alpha=0.05): "
      "see SIGNIF flags above.")
