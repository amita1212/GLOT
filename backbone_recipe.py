#!/usr/bin/env python
"""Two controls GLOT ran that we never did, plus the last reproduction gap.

BLOCK `backbone` -- their Table 11 equivalent.
    Every one of our ~2,000 runs pinned --gnn_type=gat. GLOT itself reports a
    GNN-backbone ablation, so a reviewer will reasonably ask whether "the graph
    carries no information" is really "GAT extracts no information". GAT is the
    only one of the three that consumes edge attributes, so if the graph mattered
    at all we would expect the backbones to separate. Runs {gat, gcn, gin} on
    CoLA and STS-B, 5 seeds, everything else at the paper recipe.

BLOCK `recipe` -- the last unexplained reproduction gap.
    At seed 42 over the Table 6 grid we reach or beat the published CoLA (50.52
    vs 47.49) and RTE (59.57 vs 59.21), but STS-B lands at 82.99 vs a published
    83.86 -- a residual of -0.87 that no estimator argument explains. The prime
    suspect is that the released README trains differently from the paper's
    Appendix B.2: README says 3 epochs / scorer_hidden 256 / tau 0.8, the paper
    says 2 / 128 / 0.6. This runs the 2x2x2 decomposition at THEIR seed (42) so
    the residual is either explained or definitively ruled out, and 5 seeds on
    the two endpoints so the comparison is not itself a single-seed artifact.
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

FIELDS = ["run_key", "block", "task", "variant", "seed", "metric", "score",
          "mean_density", "elapsed_sec", "detail"]

PAPER = {  # Appendix B.2 recipe, the baseline everything else perturbs
    "pooling_method": "glot", "gnn_type": "gat", "graph_adj": "threshold",
    "scorer_hidden": 128, "gat_hidden_dim": 128, "num_layers": 2,
    "jk_mode": "cat", "proj_dim": 256, "tau": 0.6,
    "lr": 2e-4, "weight_decay": 0.0, "epochs": 2, "batch_size": 32,
    "model_name_or_path": "bert-base-uncased",
    # MUST stay 1 -- run_once parses the per-epoch stdout line for the score.
    "verbose": 1,
}


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


def jobs_backbone(seeds, smoke):
    gnns = ["gat", "gcn", "gin"]
    tasks = ["cola", "stsb"]
    if smoke:
        return [("backbone", "cola", g, seeds[0], {"gnn_type": g}) for g in gnns]
    return [("backbone", t, g, s, {"gnn_type": g})
            for t, g, s in itertools.product(tasks, gnns, seeds)]


def jobs_recipe(seeds, smoke):
    out = []
    grid = list(itertools.product([2, 3], [128, 256], [0.6, 0.8]))
    if smoke:
        grid = grid[:1]
        return [("recipe", "stsb", "e2_h128_t0.6", 42,
                 {"epochs": 2, "scorer_hidden": 128, "tau": 0.6})]
    # their seed, full 2x2x2
    for ep, sh, tau in grid:
        out.append(("recipe", "stsb", f"e{ep}_h{sh}_t{tau}", 42,
                    {"epochs": ep, "scorer_hidden": sh, "tau": tau}))
    # endpoints across seeds, so the comparison is not single-seed
    for s in seeds:
        out.append(("recipe", "stsb", "paper_e2_h128_t0.6", s,
                    {"epochs": 2, "scorer_hidden": 128, "tau": 0.6}))
        out.append(("recipe", "stsb", "readme_e3_h256_t0.8", s,
                    {"epochs": 3, "scorer_hidden": 256, "tau": 0.8}))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", nargs="+", default=["backbone", "recipe"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--out", default=os.path.join(HERE, "results", "backbone_recipe.csv"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = load_done(args.out)

    combos = []
    if "backbone" in args.blocks:
        combos += jobs_backbone(args.seeds, args.smoke)
    if "recipe" in args.blocks:
        combos += jobs_recipe(args.seeds, args.smoke)

    print(f"[bkrec] {len(combos)} runs, {len(done)} already done", flush=True)
    fails = 0
    for i, (block, task, variant, seed, over) in enumerate(combos, 1):
        key = f"{block}|{task}|{variant}|s{seed}"
        if key in done and not args.smoke:
            print(f"[{i}/{len(combos)}] SKIP {key}", flush=True)
            continue
        cfg = dict(PAPER)
        cfg["task"] = task
        cfg["seed"] = seed
        cfg.update(over)

        r = run_once(cfg)
        if not r:
            fails += 1
            print(f"[{i}/{len(combos)}] FAIL {key}", flush=True)
            continue
        if args.smoke:  # cold cache -- see factorial_scale.py
            print(f"[{i}/{len(combos)}] SMOKE-OK {key} {r.get('score')}", flush=True)
            continue
        append(args.out, dict(
            run_key=key, block=block, task=task, variant=variant, seed=seed,
            metric=r.get("metric"), score=r.get("score"),
            mean_density=r.get("mean_density"), elapsed_sec=r.get("elapsed_sec"),
            detail=";".join(f"{k}={v}" for k, v in sorted(over.items())),
        ))
        print(f"[{i}/{len(combos)}] OK  {key:38s} {r.get('score')}", flush=True)

    print(f"[bkrec] DONE fails={fails}", flush=True)
    return 1 if (args.smoke and fails) else 0


if __name__ == "__main__":
    sys.exit(main())
