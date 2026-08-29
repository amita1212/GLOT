"""Paired per-seed comparison of each arm against the cosine baseline.

WHY PAIRED
----------
`analyze_campaign.py` reports mean +/- std over the confirmation seeds and
compares the gap to a pooled noise floor. With n=3 that is badly underpowered:
on CoLA L8 the per-arm std is ~1.1-1.9, so the SE on a difference of independent
means is ~1.3 MCC and nothing below ~2.5 MCC can be resolved.

But the arms are NOT independent samples -- every arm was run on the SAME seeds
(1, 2, 3). Seed controls classifier init and batch order, which is the dominant
variance component and is SHARED between an arm and the baseline. So the right
statistic is the per-seed difference d_s = arm_s - baseline_s. Pairing cancels
the shared term and typically shrinks the standard error several-fold.

Reports, per arm: the three paired differences, their mean, the SE of that mean,
and a paired t statistic. With n=3 (df=2) the two-sided 95% critical value is
4.303 -- brutal, but honest. Also prints the minimum detectable effect so an
underpowered "no difference" is never mistaken for "no effect".
"""
import csv
import glob
import os
import sys
from collections import defaultdict

# Two-sided 95% critical values of Student's t. The gate MUST depend on the
# actual degrees of freedom: an earlier version hardcoded the df=2 value AND
# required n == 3, which silently made it impossible for any larger run to be
# reported as significant -- it hid a genuine 15-seed effect (STS-B arm A,
# t = 6.20) behind an "n.s." label.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042}


def t_crit(n):
    """Two-sided 95% critical value for n paired observations (df = n-1)."""
    df = max(1, n - 1)
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return T_CRIT_95[min(T_CRIT_95, key=lambda k: abs(k - df))]


def sign_test_p(diffs):
    """Exact two-sided sign test.

    Distribution-free, so it cannot be fooled by the discrete-metric artefacts
    that make MRPC's parametric SE misleadingly small (408 dev examples, so
    accuracy is quantised at 0.245 and one example moves the score).
    """
    from math import comb
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def load(path):
    """arm -> {seed: score} for confirmation rows only."""
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r.get("stage") != "confirm":
            continue
        try:
            out[r["arm"]][int(r["seed"])] = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def analyse(path):
    arms = load(path)
    if "baseline" not in arms:
        print(f"  no baseline confirmation rows in {path}")
        return {}
    base = arms["baseline"]
    seeds = sorted(base)
    if not seeds:
        return {}

    print(f"  seeds = {seeds}")
    print(f"  {'arm':<10} {'n':>3} {'mean':>8} {'SE':>6} {'t':>7} {'signP':>9} "
          f"{'pos/neg':>9}  verdict")
    rows = []
    for arm, byseed in arms.items():
        if arm == "baseline":
            continue
        common = [s for s in seeds if s in byseed]
        if len(common) < 2:
            continue
        diffs = [byseed[s] - base[s] for s in common]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
        se = (var / n) ** 0.5
        t = mean / se if se > 0 else (float("inf") if mean else 0.0)
        pos = sum(1 for d in diffs if d > 0)
        neg = sum(1 for d in diffs if d < 0)
        sig = abs(t) > t_crit(n)
        rows.append((mean, arm, diffs, se, t, sig, n, sign_test_p(diffs), pos, neg))

    rows.sort(reverse=True)

    # MULTIPLE COMPARISONS. One campaign tests every arm against the same
    # baseline, so the family-wise error rate is what matters, not the per-arm
    # rate. With 8 arms at alpha=0.05 the chance of at least one spurious
    # "significant" is 1-0.95^8 = 34%. Bonferroni is conservative but it is the
    # assumption-free choice, and this project has already published one false
    # positive (MRPC at n=3) by reading a per-test p-value as if it stood alone.
    n_tests = len(rows)
    alpha_fw = 0.05 / n_tests if n_tests else 0.05

    for mean, arm, diffs, se, t, sig, n, sp, pos, neg in rows:
        # The verdict now requires BOTH tests to agree. The sign test was added
        # precisely because the parametric SE is untrustworthy here (task metrics
        # are quantised, and a single outlier seed distorts the t statistic);
        # letting the t-test alone drive the verdict threw that safeguard away and
        # produced self-contradictory rows -- e.g. an arm with signP=0.035 printed
        # "n.s." while one with signP=0.118 printed "SIGNIFICANT".
        sig_sign = sp < 0.05
        if sig and sig_sign:
            verdict = "** SIGNIFICANT **"
            if sp < alpha_fw:
                verdict += f"  (survives Bonferroni, a={alpha_fw:.4f})"
            else:
                verdict += "  (NOT after Bonferroni)"
        elif sig or sig_sign:
            which = "t only" if sig else "sign only"
            verdict = f"borderline ({which}) -- do not report as significant"
        else:
            verdict = "n.s."
        if (sig or sig_sign) and n < 5:
            verdict += "  (n tiny -- likely a false positive)"
        print(f"  {arm:<10} {n:>3} {mean:>+8.3f} {se:>6.3f} {t:>7.2f} {sp:>9.5f} "
              f"{pos:>4}/{neg:<4}  {verdict}")


    # Minimum detectable effect: what mean difference WOULD have been significant,
    # given the paired scatter actually observed. Guards against reading an
    # underpowered null as evidence of no effect.
    ses = [r[3] for r in rows if r[3] > 0]
    ns = [r[6] for r in rows]
    if ses:
        med = sorted(ses)[len(ses) // 2]
        nmax = max(ns)
        print(f"  -> median paired SE = {med:.3f}; minimum detectable effect "
              f"at n={nmax} is {t_crit(nmax) * med:.3f}")
    return {r[1]: r[0] for r in rows}


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("results/campaign_glue_*.csv"))
    per_task = {}
    for p in paths:
        if "ABORTED" in p:
            continue
        print(f"\n===== {os.path.basename(p)} =====")
        per_task[os.path.basename(p)] = analyse(p)

    # Cross-task summary: does any arm help CONSISTENTLY? A small effect that is
    # invisible per-task can still show up as a consistent sign across tasks.
    print("\n===== mean paired delta vs baseline, per task =====")
    tasks = list(per_task)
    allarms = sorted({a for d in per_task.values() for a in d})
    print(f"{'arm':<10}" + "".join(f"{t.split('_')[-1][:9]:>11}" for t in tasks)
          + f"{'mean':>9}{'wins':>7}")
    for a in allarms:
        vals = [per_task[t].get(a) for t in tasks]
        got = [v for v in vals if v is not None]
        if not got:
            continue
        cells = "".join(f"{v:>+11.2f}" if v is not None else f"{'-':>11}" for v in vals)
        print(f"{a:<10}{cells}{sum(got) / len(got):>+9.2f}"
              f"{sum(1 for v in got if v > 0):>4}/{len(got)}")
