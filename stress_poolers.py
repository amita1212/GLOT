#!/usr/bin/env python
"""Competing poolers on the distractor stress test.

WHY THIS EXISTS
---------------
GLOT's Table 7 / Figure 3 is the paper's headline robustness claim: GLOT holds
97.2/97.0/97.8/98.8 accuracy at 20/50/80/90% distractors while (they report)
other poolers collapse. Our replication (results/stress_followup.csv, 8 arms x
4 ratios x 5 seeds) gets 96.0/90.2/88.1/79.8 for the GLOT baseline -- 19 points
below their 90% number.

We currently CANNOT interpret that gap, because we never ran the poolers GLOT is
being compared AGAINST. If [CLS]/mean/max/AdaPool collapse the way their figure
shows, our absolute numbers are just a harder task instantiation and their
qualitative claim survives. If those poolers do FINE on our data, then either
our stress generator differs from theirs or the claim does not replicate.
Either way the paper cannot say anything honest about Table 7 without this.

Their Appendix B.4 underspecifies the vocabulary source and the sentence
templates, so an exact reproduction is not possible; this measures the RELATIVE
ordering, which is what their claim is actually about.

Graph flags are still passed for non-glot poolers; they are inert there, and
argparse takes the last occurrence so --pooling_method in the arm flags wins.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS = os.path.join(HERE, "hyperglot", "diagnostic_stress_test.py")
if not os.path.exists(STRESS):
    STRESS = os.path.join(HERE, "diagnostic_stress_test.py")

# name -> extra CLI flags. Mirrors stress_sweep.py's baseline exactly except for
# the pooler, so the comparison is pooler-only.
ARMS = {
    "glot":    ["--pooling_method=glot", "--graph_metric=cosine",
                "--graph_adj=threshold", "--hyperbolic_gnn=0",
                "--hyperbolic_readout=0"],
    "cls":     ["--pooling_method=cls"],
    "mean":    ["--pooling_method=mean"],
    "max":     ["--pooling_method=max"],
    "adapool": ["--pooling_method=adapool"],
    # GLOT with K=0: same adaptive scorer, no message passing at all. This is
    # the parameter-matched control for "is the benefit relational or capacity?"
    # (their Table 15 asks the same question with an MLP).
    "glot_K0": ["--pooling_method=glot", "--graph_metric=cosine",
                "--graph_adj=threshold", "--hyperbolic_gnn=0",
                "--hyperbolic_readout=0", "--num_layers=0"],
}


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r.get("arm", ""), str(float(r.get("distractor_ratio", -1))),
                 str(r.get("seed", ""))) for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.5, 0.8, 0.9])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=os.path.join(HERE, "results", "stress_poolers.csv"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = done_keys(args.out)

    if args.smoke:
        combos = [(a, 0.2, args.seeds[0]) for a in args.arms]
    else:
        combos = list(itertools.product(args.arms, args.ratios, args.seeds))
    print(f"[poolers] {len(combos)} runs; {len(done)} already present", flush=True)

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"

    fails = 0
    for i, (arm, ratio, seed) in enumerate(combos, 1):
        key = (arm, str(float(ratio)), str(seed))
        if key in done and not args.smoke:
            print(f"[{i}/{len(combos)}] SKIP {arm} r={ratio} s={seed}", flush=True)
            continue
        cmd = [
            sys.executable, STRESS,
            "--model_name_or_path=bert-base-uncased",
            "--decoder_cls_last_token=0",
            f"--distractor_ratio={ratio}",
            "--epochs=3", "--batch_size=32", "--eval_batch_size=32",
            "--gat_hidden_dim=64", "--scorer_hidden=256", "--num_layers=4",
            "--tau=0.6", "--lr=1e-4", f"--seed={seed}",
            f"--arm={arm}",
            f"--results_csv={args.out}",
            "--run_tag=stress_poolers",
        ] + ARMS[arm]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
        el = time.time() - t0
        if proc.returncode != 0:
            fails += 1
            print(f"[{i}/{len(combos)}] FAIL {arm} r={ratio} s={seed} "
                  f"rc={proc.returncode}", flush=True)
            for ln in (proc.stdout + proc.stderr).strip().splitlines()[-8:]:
                print("    | " + ln, flush=True)
        else:
            acc = ""
            for ln in proc.stdout.splitlines():
                if "acc" in ln.lower():
                    acc = ln.strip()[:80]
            print(f"[{i}/{len(combos)}] OK   {arm:9s} r={ratio} s={seed} "
                  f"[{el:.0f}s]  {acc}", flush=True)

    print(f"[poolers] DONE fails={fails}", flush=True)
    return 1 if (args.smoke and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
