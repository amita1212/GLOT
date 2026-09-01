"""The two experiments the paper was openly waiting on.

(1) Stage A on CoLA at n=50. The paper calls Stage A "unresolved": +0.70 MCC at
    n=15 with sign p=0.118, ahead in 5 of 7 settings. This run was powered in
    ADVANCE, so whichever way it goes is the answer -- re-running at a different
    n after seeing it would be the practice the paper criticises.

(2) Decoder STS-B, arms B and C ALONE. These are the \\PEND cells of
    tab:absolute. Until now the paper claimed a decoder replication of Stage B
    without ever having run B by itself on a decoder.

Both are paired by seed against their own campaign's baseline.
"""
import csv
import collections
from scipy import stats


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def confirm(rows):
    """arm -> {seed: score}, de-duplicated by run_key (shard/merged overlap)."""
    seen, out = set(), collections.defaultdict(dict)
    for r in rows:
        if r.get("stage") != "confirm":
            continue
        k = r.get("run_key")
        if k in seen:
            continue
        seen.add(k)
        out[r["arm"]][int(r["seed"])] = float(r["score"])
    return out


def paired(a, b, label, alpha):
    """a - b over shared seeds."""
    common = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in common]
    n = len(d)
    mean = sum(d) / n
    t, p_t = stats.ttest_rel([a[s] for s in common], [b[s] for s in common])
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    # exact two-sided sign test over non-zero differences
    nz = pos + neg
    p_sign = min(1.0, 2 * stats.binom.cdf(min(pos, neg), nz, 0.5)) if nz else 1.0
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / n ** 0.5)
    both = (p_t < alpha) and (p_sign < alpha)
    print(f"  {label:<26} d={mean:+7.3f}  95%CI [{ci[0]:+6.3f},{ci[1]:+6.3f}]  "
          f"t={t:+6.2f} p={p_t:.4f}  sign p={p_sign:.4f}  {pos}/{neg}  n={n}  "
          f"{'SIGNIFICANT(both)' if both else 'ns'}")
    return mean, p_t, p_sign


print("=" * 78)
print("(1) STAGE A on CoLA, n=50 -- powered in advance to settle 'unresolved'")
print("=" * 78)
d = confirm(load("results/stageA_n50_cola.csv"))
for arm in ("baseline", "A"):
    v = d[arm]
    print(f"  {arm:<9} n={len(v):<3} mean={sum(v.values())/len(v):7.3f}")
print("\n  Bonferroni: 1 planned comparison in this run -> alpha=0.05")
paired(d["A"], d["baseline"], "A - baseline (n=50)", 0.05)

print("\n  For reference, the same contrast restricted to the original 15 seeds:")
a15 = {s: v for s, v in d["A"].items() if s <= 15}
b15 = {s: v for s, v in d["baseline"].items() if s <= 15}
paired(a15, b15, "A - baseline (seeds 1-15)", 0.05)

print("\n" + "=" * 78)
print("(2) DECODER STS-B, arms B and C ALONE (TinyLlama-1.1B)")
print("=" * 78)
dd = confirm(load("results/campaign_decoder_stsb_BC.csv"))
print("  arms present:", sorted(dd))
for arm in sorted(dd):
    v = dd[arm]
    print(f"  {arm:<9} n={len(v):<3} mean={sum(v.values())/len(v):7.3f}  "
          f"sd={ (sum((x-sum(v.values())/len(v))**2 for x in v.values())/(len(v)-1))**0.5 :.3f}")
print("\n  NOTE: this campaign has no baseline arm of its own -- B and C were run")
print("  alone. The paper's decoder baseline (79.95 +/- 0.49, n=15, seeds 1-15)")
print("  comes from the earlier decoder campaign, so differencing against it is")
print("  a CROSS-CAMPAIGN splice, which app:determinism shows is unsafe.")
print("  Reporting absolute means here and flagging the comparison as such.")
