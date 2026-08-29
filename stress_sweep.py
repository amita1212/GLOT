#!/usr/bin/env python
"""Stress-test follow-up: 8 arms x 4 distractor ratios x 5 seeds, resumable.

Why this rerun exists
---------------------
The first stress sweep passed no --rho, so `poincare + threshold` fell back to
rho=1.0. Measured on real BERT features, every rho below ~9 yields ZERO edges,
so what was labelled "A_thresh" was in fact a NO-GRAPH control. It scored
97.8/97.2/95.6/94.8 vs the cosine baseline's 95.8/92.6/89.8/88.4 -- i.e. the
apparent "Stage A wins" was really "removing the graph wins".

That is still an interesting result, so instead of discarding it we promote it
to a first-class, honestly-named arm (`no_graph`) and add the genuine Stage A
with a density-matched quantile threshold. The design now separates three
distinct hypotheses:

    baseline   cosine graph at tau=0.6            (GLOT as published)
    no_graph   empty graph (rho=1.0, 0 edges)     (does the graph help at all?)
    A_thresh   Poincare graph, density-matched    (does GEOMETRY help?)

`A_thresh` uses rho_quantile=0.10, matched to the measured density of cosine
tau=0.6 (0.0989 vs 0.0998), so baseline and A_thresh differ ONLY in geometry.

Config follows the paper's stress-test command (README): 4 GNN layers,
hidden 64, scorer 256, tau 0.6, lr 1e-4, 3 epochs.
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

# name -> extra CLI flags identifying the arm
ARMS = {
    "baseline":   ["--graph_metric=cosine",   "--graph_adj=threshold",
                   "--hyperbolic_gnn=0", "--hyperbolic_readout=0"],
    "no_graph":   ["--graph_metric=poincare", "--graph_adj=threshold",
                   "--hyperbolic_gnn=0", "--hyperbolic_readout=0",
                   "--rho=1.0"],                      # deliberately empty graph
    "A_thresh":   ["--graph_metric=poincare", "--graph_adj=threshold",
                   "--hyperbolic_gnn=0", "--hyperbolic_readout=0",
                   "--rho_quantile=0.10"],            # density-matched to tau=0.6
    "A_knn":      ["--graph_metric=poincare", "--graph_adj=knn", "--knn_k=8",
                   "--hyperbolic_gnn=0", "--hyperbolic_readout=0"],
    "B_thresh":   ["--graph_metric=poincare", "--graph_adj=threshold",
                   "--hyperbolic_gnn=0", "--hyperbolic_readout=1",
                   "--rho_quantile=0.10", "--readout_clip=0.7",
                   "--readout_scale=1", "--learnable_curvature=1"],
    "C_thresh":   ["--graph_metric=cosine",   "--graph_adj=threshold",
                   "--hyperbolic_gnn=1", "--hyperbolic_readout=0",
                   "--hyp_gnn_type=gat", "--gnn_input_clip=0.7", "--gnn_input_scale=1"],
    "AC_thresh":  ["--graph_metric=poincare", "--graph_adj=threshold",
                   "--hyperbolic_gnn=1", "--hyperbolic_readout=0",
                   "--rho_quantile=0.10", "--hyp_gnn_type=gat",
                   "--gnn_input_clip=0.7", "--gnn_input_scale=1"],
    "ABC_thresh": ["--graph_metric=poincare", "--graph_adj=threshold",
                   "--hyperbolic_gnn=1", "--hyperbolic_readout=1",
                   "--rho_quantile=0.10", "--hyp_gnn_type=gat",
                   "--gnn_input_clip=0.7", "--gnn_input_scale=1",
                   "--readout_clip=0.7", "--readout_scale=1", "--learnable_curvature=1"],
}


def done_keys(path):
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.add((r.get("arm", ""), str(float(r.get("distractor_ratio", -1))),
                     str(r.get("seed", ""))))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--ratios", nargs="+", type=float, default=[0.2, 0.5, 0.8, 0.9])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--out", default=os.path.join(HERE, "results", "stress_followup.csv"))
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = done_keys(args.out)

    combos = list(itertools.product(args.arms, args.ratios, args.seeds))
    print(f"total {len(combos)} runs; {len(done)} already present", flush=True)

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"

    for i, (arm, ratio, seed) in enumerate(combos, 1):
        key = (arm, str(float(ratio)), str(seed))
        if key in done:
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
            "--pooling_method=glot",
            f"--arm={arm}",
            f"--results_csv={args.out}",
            "--run_tag=stress_followup",
        ] + ARMS[arm]

        t0 = time.time()
        proc = subprocess.run(cmd, cwd=HERE, env=env,
                              capture_output=True, text=True)
        el = time.time() - t0
        if proc.returncode != 0:
            print(f"[{i}/{len(combos)}] FAIL {arm} r={ratio} s={seed} rc={proc.returncode}",
                  flush=True)
            for ln in (proc.stdout + proc.stderr).strip().splitlines()[-6:]:
                print("    | " + ln, flush=True)
        else:
            acc = ""
            for ln in proc.stdout.splitlines():
                if "acc" in ln.lower():
                    acc = ln.strip()[:80]
            print(f"[{i}/{len(combos)}] OK   {arm:11s} r={ratio} s={seed} "
                  f"[{el:.0f}s]  {acc}", flush=True)


if __name__ == "__main__":
    main()
