#!/usr/bin/env python
"""
Analysis for the HyperGLOT geometry experiments.

Reporting rules enforced here (they exist because earlier claims died to each):
  * The comparison unit is the ARM's BEST config, not a cherry-picked row.
  * Deltas are quoted in units of the measured seed noise floor, never in raw
    points. Anything under ~1.5 sigma on CoLA is indistinguishable from noise.
  * Single-seed rows are labelled as such. Same-seed GPU nondeterminism alone is
    0.2-0.5 MCC (scatter_add atomics), so a single-seed win is never a result.
  * `mean_density` and `empty_frac` are printed for every arm: an empty graph is
    the failure mode that invalidated an entire generation of this project.
  * A knob whose scores are constant across its grid is flagged, because that
    means it is disconnected (how the c=1.0 saturation went unnoticed).
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
from collections import defaultdict

# Measured seed noise floors (std across seeds 1-5, warm cache, paper recipe).
NOISE = {"cola": 0.81, "stsb": 0.53, "rte": 1.40}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(r, k, default=float("nan")):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--baseline_arm", default="cosine")
    args = ap.parse_args()

    rows = []
    for p in args.csvs:
        if os.path.exists(p):
            rows += load(p)
    if not rows:
        print("no rows")
        return

    for task in sorted({r["task"] for r in rows}):
        trows = [r for r in rows if r["task"] == task]
        metric = trows[0]["metric"]
        sigma = NOISE.get(task, 1.0)
        print("=" * 92)
        print(f"TASK {task}  ({metric})   seed-noise sigma = {sigma}")
        print("=" * 92)

        # ---- every config, grouped by arm ----
        by_arm = defaultdict(list)
        for r in trows:
            by_arm[r.get("arm", "?")].append(r)

        print(f"\n{'arm':<18} {'detail':<20} {'n':>2} {'mean':>7} {'std':>6} "
              f"{'density':>8}")
        print("-" * 70)
        arm_best = {}
        for arm in sorted(by_arm):
            by_cfg = defaultdict(list)
            for r in by_arm[arm]:
                by_cfg[r.get("detail", "")].append(fnum(r, "score"))
            cfg_means = []
            for detail in sorted(by_cfg, key=lambda d: -st.mean(by_cfg[d])):
                v = by_cfg[detail]
                dens = [fnum(r, "mean_density") for r in by_arm[arm]
                        if r.get("detail") == detail and r.get("mean_density")]
                d = f"{st.mean(dens):.4f}" if dens else "-"
                sd = f"{st.stdev(v):.2f}" if len(v) > 1 else "-"
                print(f"{arm:<18} {detail:<20} {len(v):>2} {st.mean(v):>7.2f} "
                      f"{sd:>6} {d:>8}")
                cfg_means.append((st.mean(v), detail, len(v)))
            if cfg_means:
                arm_best[arm] = max(cfg_means)

            # disconnected-knob detector
            allv = [fnum(r, "score") for r in by_arm[arm]]
            if len(set(round(v, 3) for v in allv)) == 1 and len(allv) > 2:
                print(f"  !! {arm}: score is CONSTANT across its whole grid -> "
                      f"the knob is disconnected. Investigate before reporting.")

        # ---- best-per-arm league table vs the baseline arm ----
        if args.baseline_arm in arm_best:
            base, base_detail, base_n = arm_best[args.baseline_arm]
            print(f"\nBEST PER ARM vs {args.baseline_arm} ({base:.2f}, {base_detail})")
            print(f"{'arm':<18} {'best config':<20} {'score':>7} {'delta':>7} "
                  f"{'sigma':>7} {'n':>3}  verdict")
            print("-" * 82)
            for arm, (sc, detail, n) in sorted(arm_best.items(), key=lambda kv: -kv[1][0]):
                d = sc - base
                z = d / sigma
                if n < 3:
                    verdict = "single-seed, NOT a result"
                elif abs(z) < 1.5:
                    verdict = "within noise"
                elif z > 0:
                    verdict = "*** BEATS BASELINE ***"
                else:
                    verdict = "worse"
                print(f"{arm:<18} {detail:<20} {sc:>7.2f} {d:>+7.2f} {z:>+7.2f} "
                      f"{n:>3}  {verdict}")

        # ---- empty-graph guard ----
        bad = [r for r in trows if r.get("empty_frac") not in (None, "")
               and fnum(r, "empty_frac", 0) > 0.01]
        if bad:
            print(f"\n!! {len(bad)} run(s) had >1% EMPTY graphs -- results invalid:")
            for r in bad[:10]:
                print(f"   {r.get('run_key')}  empty_frac={r['empty_frac']}")
        print()


if __name__ == "__main__":
    main()
