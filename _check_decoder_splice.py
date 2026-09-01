"""Are the paper's new decoder B and C deltas computed ACROSS campaigns?

short.tex now reports, for TinyLlama STS-B:
    B alone  -1.163  t=-3.66  p=0.0026  sign p=0.0074  2/13
    C alone  +0.291  t= 1.52  p=0.302              10/5

But results/campaign_decoder_stsb_BC.csv contains ONLY arms B and C -- it has
no baseline of its own. Any delta therefore has to subtract the baseline from
results/campaign_decoder_stsb.csv, a DIFFERENT campaign run four days earlier.

The paper's own app:determinism concludes "no table may splice cells across
campaigns", and app:audit lists the previous splice as its most consequential
error. So either these numbers are not a splice, or the paper contradicts its
own rule. This script settles which by reproducing them.
"""
import csv
import collections
from scipy import stats


def confirm(path):
    seen, out = set(), collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("stage") != "confirm":
            continue
        k = r.get("run_key")
        if k in seen:
            continue
        seen.add(k)
        out[r["arm"]][int(r["seed"])] = float(r["score"])
    return out


old = confirm("results/campaign_decoder_stsb.csv")
new = confirm("results/campaign_decoder_stsb_BC.csv")

print("=== arms in each file ===")
print("  campaign_decoder_stsb.csv    (Aug 27):", sorted(old))
print("  campaign_decoder_stsb_BC.csv (Aug 31):", sorted(new))
print("  baseline present in the B/C campaign?",
      "YES" if "baseline" in new else "NO  <-- so any delta must be spliced")

base = old.get("baseline", {})
print(f"\n  baseline (old campaign) n={len(base)} mean={sum(base.values())/len(base):.3f}")


def paired(a, b, label):
    common = sorted(set(a) & set(b))
    da = [a[s] for s in common]
    db = [b[s] for s in common]
    d = [x - y for x, y in zip(da, db)]
    n = len(d)
    mean = sum(d) / n
    t, p_t = stats.ttest_rel(da, db)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    p_sign = min(1.0, 2 * stats.binom.cdf(min(pos, neg), pos + neg, 0.5))
    print(f"  {label:<38} d={mean:+7.3f}  t={t:+6.2f}  p={p_t:.4f}  "
          f"sign p={p_sign:.4f}  {pos}/{neg}  n={n}")


print("\n=== SPLICED: new B/C campaign minus OLD campaign's baseline ===")
paired(new["B"], base, "B (Aug 31) - baseline (Aug 27)")
paired(new["C"], base, "C (Aug 31) - baseline (Aug 27)")
print("  paper reports: B -1.163 t=-3.66 p=0.0026 sign 0.0074 2/13 ;"
      " C +0.291 t=1.52 p=0.302 10/5")

print("\n=== for contrast, WITHIN the old campaign (no splice) ===")
for arm in sorted(old):
    if arm != "baseline":
        paired(old[arm], base, f"{arm} - baseline (both Aug 27)")

print("\n=== how far apart are the two campaigns on a SHARED arm? ===")
shared = (set(old) & set(new)) - {"baseline"}
if shared:
    for arm in sorted(shared):
        paired(new[arm], old[arm], f"{arm}: Aug 31 run - Aug 27 run")
else:
    print("  No arm was run in BOTH campaigns, so the re-run drift between them")
    print("  cannot be measured at all -- there is no shared cell to compare.")
    print("  That is precisely the situation app:determinism warns about: the")
    print("  splice is unquantifiable rather than merely small.")
