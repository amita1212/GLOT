"""RoBERTa RTE, the first Table-1 gap cell to finish. Same protocol as the paper."""
import pandas as pd
from scipy import stats

d = pd.read_csv("results_vm/campaign_t1_rob_rte.csv")
c = d[d.stage == "confirm"]
print("confirm rows per arm:", dict(c.groupby("arm").size()))
print()

b = c[c.arm == "baseline"][["seed", "score"]]
hdr = f"{'arm':10s} {'mean':>7s} {'sd':>6s} {'delta':>7s} {'t':>7s} {'t_p':>9s} {'sign_p':>9s}  p/n"
print(hdr)
print("-" * len(hdr))

for arm in ["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC", "no_graph"]:
    g = c[c.arm == arm][["seed", "score"]]
    if len(g) == 0:
        continue
    m, s = g.score.mean(), g.score.std(ddof=1)
    if arm == "baseline":
        print(f"{arm:10s} {m:7.2f} {s:6.2f} {'ref':>7s}")
        continue
    mm = g.merge(b, on="seed", suffixes=("_a", "_b"))
    dd = mm.score_a.values - mm.score_b.values
    t, pt = stats.ttest_rel(mm.score_a, mm.score_b)
    pos, neg = int((dd > 0).sum()), int((dd < 0).sum())
    sp = stats.binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0
    print(f"{arm:10s} {m:7.2f} {s:6.2f} {dd.mean():+7.2f} {t:7.2f} {pt:9.4f} {sp:9.4f}  {pos}/{neg}")

# minimum detectable effect, same recipe the paper uses elsewhere
diffs = []
for arm in ["A", "B", "C", "AB", "AC", "BC", "ABC", "no_graph"]:
    g = c[c.arm == arm][["seed", "score"]]
    mm = g.merge(b, on="seed", suffixes=("_a", "_b"))
    diffs.append((mm.score_a.values - mm.score_b.values).std(ddof=1))
import statistics
med_sd = statistics.median(diffs)
print()
print(f"median paired sd across arms : {med_sd:.3f}")
print(f"MDE at n=15, 80% power, a=.05: {2.98 * med_sd / (15 ** 0.5):.3f}")
print(f"baseline seed sd             : {b.score.std(ddof=1):.3f}")
