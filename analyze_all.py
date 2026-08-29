#!/usr/bin/env python
"""Final analysis across every result file the pipeline produced.

Reports, in order:
  1. Fair ablation  -- best-per-arm, multi-seed, with deltas in units of the
     pooled seed std. This is the answer to "do our methods work?".
  2. Diagnostic stress test vs the paper's Table 7.
  3. Table 6 sweeps -- does ANY config in the paper's declared search space
     reproduce the Table 1 number, and how much of that is selection bias?
"""

from __future__ import annotations

import csv
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

PAPER_TABLE1_BERT = {
    "cola": 47.49, "sst2": 90.25, "stsb": 83.86, "mrpc": 82.58, "qqp": 62.19,
    "mnli": 54.39, "qnli": 61.08, "rte": 59.21, "wnli": 54.93,
}
# Paper Table 7, BERT / GLOT row: accuracy by distractor ratio
PAPER_STRESS_BERT = {"0.2": 97.2, "0.5": 97.0, "0.8": 97.8, "0.9": 98.8}


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def expected_max_of_n(n: int) -> float:
    """E[max of n standard normals] (Blom)."""
    if n < 2:
        return 0.0
    p = (n - 0.375) / (n + 0.25)
    y = 2.0 * p - 1.0
    a = 0.147
    ln = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    return math.sqrt(2.0) * math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)


# --------------------------------------------------------------------------- #
def ablation():
    rows = [r for r in load(os.path.join(R, "ablation_fair.csv")) if r["stage"] == "final"]
    if not rows:
        print("no ablation rows")
        return
    bar("1. FAIR ABLATION  (best config per arm, tuned on equal budget, 3 seeds)")
    for task in ["cola", "stsb", "rte"]:
        trows = [r for r in rows if r["task"] == task]
        if not trows:
            continue
        metric = trows[0]["metric"]
        by_arm = {}
        for r in trows:
            by_arm.setdefault(r["arm"], {"vals": [], "knob": f"{r['knob']}={r['knob_value']}"})
            by_arm[r["arm"]]["vals"].append(float(r["score"]))

        stds = [statistics.stdev(v["vals"]) for v in by_arm.values() if len(v["vals"]) > 1]
        pooled = statistics.mean(stds) if stds else float("nan")
        base = by_arm.get("baseline", {}).get("vals", [])
        base_mean = statistics.mean(base) if base else None

        print(f"\n{task.upper()}  ({metric})   paper Table 1 = {PAPER_TABLE1_BERT.get(task, '?')}")
        print(f"  {'arm':12s} {'best knob':>22s} {'n':>2s} {'mean':>7s} {'std':>6s} "
              f"{'vs base':>8s} {'sigmas':>7s}  verdict")
        print("  " + "-" * 86)
        order = sorted(by_arm, key=lambda a: -statistics.mean(by_arm[a]["vals"]))
        for arm in order:
            v = by_arm[arm]["vals"]
            m, s = statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
            if arm == "baseline" or base_mean is None:
                d = sg = ""
                verdict = "(control)" if arm == "baseline" else ""
            else:
                delta = m - base_mean
                sig = delta / pooled if pooled == pooled and pooled > 0 else 0.0
                d, sg = f"{delta:+.2f}", f"{sig:+.1f}"
                verdict = "SIGNIFICANT" if abs(sig) >= 2 else "not established"
                if abs(sig) >= 2 and delta < 0:
                    verdict = "SIGNIFICANTLY WORSE"
            print(f"  {arm:12s} {by_arm[arm]['knob']:>22s} {len(v):>2d} {m:>7.2f} {s:>6.2f} "
                  f"{d:>8s} {sg:>7s}  {verdict}")
        print(f"  pooled seed std = {pooled:.2f}   (|delta| < 2 sigma => not established)")


# --------------------------------------------------------------------------- #
def stress():
    rows = load(os.path.join(R, "stress_warm.csv"))
    if not rows:
        print("\nno stress rows")
        return
    bar("2. DIAGNOSTIC STRESS TEST  vs paper Table 7 (BERT / GLOT)")
    acc_key = next((k for k in ("acc", "score") if k in rows[0]), None)
    ratio_key = next((k for k in ("distractor_ratio", "ratio") if k in rows[0]), None)
    if not acc_key or not ratio_key:
        print("  columns:", list(rows[0])[:14])
        return
    arms = sorted({r["arm"] for r in rows})
    ratios = ["0.2", "0.5", "0.8", "0.9"]
    print(f"  {'arm':16s} " + "".join(f"{r:>9s}" for r in ratios))
    print("  " + "-" * 60)
    print(f"  {'PAPER GLOT':16s} " + "".join(f"{PAPER_STRESS_BERT[r]:>9.1f}" for r in ratios))
    for arm in arms:
        cells = []
        for ratio in ratios:
            v = [float(r[acc_key]) * (100 if float(r[acc_key]) <= 1 else 1)
                 for r in rows if r["arm"] == arm and str(float(r[ratio_key])) == str(float(ratio))]
            cells.append(f"{statistics.mean(v):>9.1f}" if v else f"{'-':>9s}")
        print(f"  {arm:16s} " + "".join(cells))


# --------------------------------------------------------------------------- #
def sweeps():
    bar("3. PAPER TABLE 6 SWEEPS  (does any declared config reproduce Table 1?)")
    # seed-noise std measured earlier on CoLA, used for the selection-bias check
    NOISE = {"cola": 0.81, "stsb": 0.53, "rte": 1.40}
    for task in ["cola", "stsb", "rte"]:
        rows = load(os.path.join(R, f"sweep_{task}.csv"))
        if not rows:
            continue
        scores = [float(r["score"]) for r in rows]
        best = max(rows, key=lambda r: float(r["score"]))
        n = len(scores)
        paper = PAPER_TABLE1_BERT[task]
        mean = statistics.mean(scores)
        bias = NOISE[task] * expected_max_of_n(n)
        corrected = float(best["score"]) - bias
        n_beat = sum(1 for s in scores if s >= paper)
        print(f"\n{task.upper()}  ({best['metric']})  trials = {n}")
        print(f"  paper Table 1        {paper:.2f}")
        print(f"  our best             {float(best['score']):.2f}   "
              f"(delta {float(best['score']) - paper:+.2f})")
        print(f"  our mean over trials {mean:.2f}")
        print(f"  trials >= paper      {n_beat}/{n}")
        print(f"  best config          lr={best['lr']} wd={best['weight_decay']} "
              f"K={best['num_layers']} h={best['gat_hidden_dim']} jk={best['jk_mode']} "
              f"proj={best['proj_dim']} tau={best['tau']}")
        print(f"  selection bias       +{bias:.2f} expected from noise alone (N={n}, "
              f"sigma={NOISE[task]})")
        print(f"  noise-corrected best {corrected:.2f}  -> "
              f"{'REPRODUCED' if corrected >= paper else 'NOT reproduced'}")


if __name__ == "__main__":
    ablation()
    stress()
    sweeps()
    print()
