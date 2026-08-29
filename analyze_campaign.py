#!/usr/bin/env python
"""
Analysis for a campaign.py run (all arms, equal-budget random search).

Two tables, deliberately kept separate:

  TUNING (seed 42)  -- the maximum over each arm's trials. This number is
                       BIASED UPWARD and must never be reported as a result.
                       With k trials drawn from a distribution of width sigma,
                       the expected maximum exceeds the true mean by roughly
                       sigma * E[max of k standard normals]; for k=10 that is
                       about 1.54 sigma even if the arm is identical to the
                       baseline. It is shown only to document what was searched.

  CONFIRMATION      -- the same config re-run on seeds the tuner never saw.
                       This is the only table a claim may be based on. Deltas
                       are quoted against the baseline in units of the pooled
                       across-seed standard deviation.

Also reported, because each has previously produced a false positive here:
  * mean edge density per arm -- a "win" that comes with a very different
    density is a sparsity effect, not a geometry effect.
  * the `no_graph` control -- if an arm does not also beat "no token graph at
    all", it is not evidence for the graph, let alone for hyperbolic geometry.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics as st
from collections import defaultdict

# E[max of k iid standard normals]: the selection-bias inflation of a
# max-over-k-trials tuning score, in units of sigma.
EMAX = {1: 0.00, 2: 0.56, 3: 0.85, 4: 1.03, 5: 1.16, 6: 1.27,
        7: 1.35, 8: 1.42, 9: 1.49, 10: 1.54, 12: 1.63, 15: 1.74, 20: 1.87}


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(r, k, d=float("nan")):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--baseline", default="baseline")
    args = ap.parse_args()

    rows = []
    for p in args.csvs:
        if os.path.exists(p):
            rows += load(p)
    if not rows:
        print("no rows")
        return

    for setting in sorted({r.get("setting", "") for r in rows}):
        srows = [r for r in rows if r.get("setting") == setting]
        model = srows[0].get("model", "")
        metric = srows[0].get("metric", "")
        print("=" * 94)
        print(f"SETTING {setting}   model={model}   metric={metric}")
        print("=" * 94)

        tune = [r for r in srows if r.get("stage") == "tune"]
        conf = [r for r in srows if r.get("stage") == "confirm"]

        # ---------- tuning (biased, documentation only) ----------
        best = {}
        ntrials = defaultdict(int)
        for r in tune:
            a = r["arm"]
            ntrials[a] += 1
            if a not in best or fnum(r, "score") > fnum(best[a], "score"):
                best[a] = r
        if best:
            print("\nTUNING max over trials, seed 42 -- BIASED, not a result")
            print(f"{'arm':<10} {'trials':>6} {'best':>7} {'density':>8}  config")
            print("-" * 94)
            for a, r in sorted(best.items(), key=lambda kv: -fnum(kv[1], "score")):
                d = fnum(r, "mean_density")
                print(f"{a:<10} {ntrials[a]:>6} {fnum(r,'score'):>7.2f} "
                      f"{d:>8.4f}  {r.get('detail','')[:52]}")

        # ---------- confirmation (the real comparison) ----------
        if not conf:
            print("\n(confirmation stage not finished yet)")
            continue

        by_arm = defaultdict(list)
        dens = defaultdict(list)
        for r in conf:
            by_arm[r["arm"]].append(fnum(r, "score"))
            d = fnum(r, "mean_density")
            if not math.isnan(d):
                dens[r["arm"]].append(d)

        # Noise floor. A pooled std over ALL arms is the wrong estimator here:
        # the Stage B/C arms are genuinely unstable (seed std up to ~10 acc
        # points), so pooling lets their instability inflate sigma until every
        # comparison is trivially "within noise". The reference scale for
        # "is this arm better than the baseline" is the variability of STABLE
        # runs, so we use the MEDIAN per-arm std, and report the pooled value
        # and the worst arm separately because the instability is itself a
        # finding rather than a nuisance.
        spreads = sorted(st.stdev(v) for v in by_arm.values() if len(v) > 1)
        sigma = st.median(spreads) if spreads else float("nan")
        pooled = ((sum(s * s for s in spreads) / len(spreads)) ** 0.5
                  if spreads else float("nan"))

        base = st.mean(by_arm[args.baseline]) if args.baseline in by_arm else float("nan")
        nog = st.mean(by_arm["no_graph"]) if "no_graph" in by_arm else float("nan")

        print(f"\nCONFIRMATION on held-out seeds")
        print(f"  noise floor (median per-arm std) = {sigma:.2f}")
        print(f"  pooled std over all arms         = {pooled:.2f} "
              f"(inflated by the unstable hyperbolic arms)")
        print(f"  worst arm std                    = {spreads[-1]:.2f}" if spreads else "")
        print(f"{'arm':<10} {'n':>2} {'mean':>7} {'std':>6} {'density':>8} "
              f"{'d-base':>7} {'sigma':>6} {'d-nograph':>10}  verdict")
        print("-" * 94)
        for a, v in sorted(by_arm.items(), key=lambda kv: -st.mean(kv[1])):
            m = st.mean(v)
            sd = st.stdev(v) if len(v) > 1 else float("nan")
            dd = m - base
            z = dd / sigma if sigma and not math.isnan(sigma) else float("nan")
            dn = m - nog
            dstr = f"{st.mean(dens[a]):.4f}" if dens.get(a) else "-"
            if a == args.baseline:
                verdict = "(reference)"
            elif len(v) < 3:
                verdict = "too few seeds"
            elif z > 1.5 and dn > 0:
                verdict = "*** BEATS BASELINE AND no_graph ***"
            elif z > 1.5:
                verdict = "beats baseline but NOT no_graph"
            elif abs(z) <= 1.5:
                verdict = "within noise"
            else:
                verdict = "worse"
            print(f"{a:<10} {len(v):>2} {m:>7.2f} {sd:>6.2f} {dstr:>8} "
                  f"{dd:>+7.2f} {z:>+6.2f} {dn:>+10.2f}  {verdict}")

        # ---------- selection-bias sanity note ----------
        k = max(ntrials.values()) if ntrials else 0
        if k and not math.isnan(sigma):
            infl = EMAX.get(k, 1.5) * sigma
            print(f"\nSelection-bias check: with {k} trials and sigma={sigma:.2f}, the "
                  f"expected max-over-trials\nis inflated by ~{infl:.2f} even for an arm "
                  f"identical to the baseline. Any tuning-stage\nlead smaller than that "
                  f"is fully explained by the search itself.")

        # ---------- density audit ----------
        if dens:
            lo = min(st.mean(v) for v in dens.values())
            hi = max(st.mean(v) for v in dens.values())
            if hi > 0 and (hi / max(lo, 1e-9)) > 2.0:
                print(f"\n!! density spans {lo:.4f}..{hi:.4f} across arms ({hi/max(lo,1e-9):.1f}x). "
                      f"Differences may be SPARSITY, not geometry.")
        print()


if __name__ == "__main__":
    main()
