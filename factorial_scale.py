#!/usr/bin/env python
"""Density x scale factorial: is GLOT's calibration failure about DENSITY or SCALE?

WHY THIS EXISTS
---------------
The paper currently claims a "two-part correction" (density-matching + median
rescaling) that recovers +5.44 MCC on ModernBERT/CoLA. The RoBERTa campaign
(results/campaign_rob_{cola,stsb}.csv, 15 paired seeds) refutes the density half
outright:

    density_fix vs baseline   CoLA -1.408  (2/13, p=0.0074, survives Bonferroni)
    density_fix vs paper_tau  CoLA -1.265  (3/12)
    density_fix vs paper_tau  STS-B -0.475 (2/13, p=0.0074)

i.e. density-matching is SIGNIFICANTLY HARMFUL, and the near-complete graph the
paper calls "degenerate" is the best arm on STS-B. Consistent with ModernBERT,
where density-matching ALONE scored 9.16 vs the published 21.71 -- also harmful.

That leaves median rescaling as the only candidate for the +5.44, but the two
were never separated: the cell (absolute tau=0.6) x (median) was never run.
This script runs the full 2x3 factorial so each factor is attributable:

    density in {abs06 = --tau=0.6, q05 = --tau_quantile=0.05}
    scale   in {none, median, rms}

across four backbone/layer combinations, 5 seeds each. BERT is the control: its
natural density at tau=0.6 is ~0.10, so abs06 and q05 are nearly the same graph
and BERT should barely move. If BERT moves a lot, the knob does something
unintended and the whole section is wrong.

ModernBERT is run at BOTH L12 (where the +5.44 was measured, and where the token
norm mean/median ratio is 7.66) and at the final layer (where the independently
re-measured density is 0.599, not the 0.996 the paper reported -- that number
was L12, not final).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_runner import run_once  # noqa: E402

# (tag, model, hidden_layer)
BACKBONES = [
    ("bert_final",     "bert-base-uncased",           -1),
    ("mbert_L12",      "answerdotai/ModernBERT-base",  12),
    ("mbert_final",    "answerdotai/ModernBERT-base",  -1),
    ("roberta_final",  "roberta-base",                -1),
]

# The two halves of the "two-part correction", crossed.
DENSITY = {
    "abs06": {"tau": 0.6},                 # GLOT as published
    "q05":   {"tau_quantile": 0.05},       # density-matched
}
SCALE = {
    "none":   {},                          # upstream GLOT
    "median": {"input_scale_norm": "median", "input_scale_target": 15.0},
    "rms":    {"input_scale_norm": "rms",    "input_scale_target": 15.0},
}

FIELDS = ["run_key", "backbone", "model", "hidden_layer", "task", "density",
          "scale", "seed", "metric", "score", "mean_density", "elapsed_sec"]


def load_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["run_key"] for r in csv.DictReader(f) if r.get("run_key")}


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cola")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--backbones", nargs="+", default=[b[0] for b in BACKBONES])
    p.add_argument("--out", default=os.path.join(HERE, "results", "factorial_scale.csv"))
    p.add_argument("--smoke", action="store_true",
                   help="one run per backbone, seed 1, abs06/none; verifies the "
                        "flags are accepted before committing to the full grid")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = load_done(args.out)

    bks = [b for b in BACKBONES if b[0] in args.backbones]
    if args.smoke:
        combos = [(b, "abs06", "none", args.seeds[0]) for b in bks]
    else:
        combos = list(itertools.product(bks, DENSITY, SCALE, args.seeds))

    print(f"[factorial] {len(combos)} runs, {len(done)} already done", flush=True)
    ok = fail = 0
    for i, (bk, dens, sc, seed) in enumerate(combos, 1):
        tag, model, layer = bk
        key = f"{tag}|{args.task}|{dens}|{sc}|s{seed}"
        if key in done:
            print(f"[{i}/{len(combos)}] SKIP {key}", flush=True)
            continue

        cfg = {
            "model_name_or_path": model,
            "hidden_layer": layer,
            "task": args.task,
            "seed": seed,
            # GLOT paper recipe, held fixed so only density/scale vary.
            "pooling_method": "glot", "gnn_type": "gat",
            "scorer_hidden": 128, "gat_hidden_dim": 128, "num_layers": 2,
            "jk_mode": "cat", "graph_adj": "threshold", "proj_dim": 256,
            "lr": 2e-4, "weight_decay": 0.0, "epochs": 2, "batch_size": 32,
            # MUST stay 1: exp_runner.run_once recovers the score by regexing the
            # per-epoch "epoch N loss .. acc .. mcc .." line out of stdout, and
            # verbose=0 suppresses exactly that line. With verbose=0 every run
            # succeeds, prints RESULT_JSON, and is still recorded as a failure.
            "verbose": 1,
        }
        cfg.update(DENSITY[dens])
        cfg.update(SCALE[sc])

        r = run_once(cfg)
        if not r:
            fail += 1
            print(f"[{i}/{len(combos)}] FAIL {key}", flush=True)
            continue
        ok += 1
        # Smoke runs are deliberately NOT recorded. They execute on a cold
        # feature cache, and cold-vs-warm cache is a known confound in this
        # codebase (shuffled DataLoader consumes torch.randperm only on a cache
        # miss: 40.37 vs 45.54 MCC on CoLA at the same seed). Their real value
        # is that they WARM the cache for the graded runs that follow.
        if args.smoke:
            print(f"[{i}/{len(combos)}] SMOKE-OK {key} {r.get('score')} (not recorded)",
                  flush=True)
            continue
        append(args.out, dict(
            run_key=key, backbone=tag, model=model, hidden_layer=layer,
            task=args.task, density=dens, scale=sc, seed=seed,
            metric=r.get("metric"), score=r.get("score"),
            mean_density=r.get("mean_density"), elapsed_sec=r.get("elapsed_sec"),
        ))
        print(f"[{i}/{len(combos)}] OK  {key:44s} {r.get('score')}  "
              f"density={r.get('mean_density')}", flush=True)

    print(f"[factorial] DONE ok={ok} fail={fail}", flush=True)
    return 1 if (args.smoke and fail) else 0


if __name__ == "__main__":
    sys.exit(main())
