#!/usr/bin/env python
"""Fair HyperGLOT ablation: tune every arm's graph-sparsity knob on an equal
budget, then compare best-vs-best across multiple seeds.

Why this replaces the old ablation
----------------------------------
The July 2026 ablation had TWO biases, both favouring the hyperbolic arms:

  1. Cache confound. `baseline` was the first row for every task, so it built
     the hidden-state cache (COLD) while every hyperbolic arm ran WARM. Cold
     runs score ~4.9 MCC lower on CoLA (40.36 vs 45.30, pooled std ~0.37).
     Fixed here by requiring a pre-warmed cache (see gcp/prewarm_caches.sh).

  2. Asymmetric tuning. The cosine baseline was pinned at a single tau=0.8
     while each hyperbolic arm got its own rho / knn_k. Paper Table 8 shows
     tau alone moves CoLA by 7.9 points, so this silently detuned the control.
     Fixed here: every arm sweeps its OWN sparsity knob over a grid of the
     same size, and we compare each arm's best against the others' best.

Protocol
--------
  Stage 1 (tune)    : sweep each arm's sparsity knob at seed 42; pick the best.
  Stage 2 (confirm) : re-run each arm's best config over several seeds.
  Report            : mean +/- std per arm, and the delta vs the baseline arm
                      expressed in units of the pooled seed std.

Caveat, stated plainly: Stage 1 tunes on the same split we report, which is
optimistic in absolute terms. It is applied identically to every arm, so the
*comparison between arms* stays fair -- which is the only claim we make.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "hyperglot", "main.py")

RE_STS = re.compile(r"epoch (\d+) MSE [\d.]+ Spearman ([-\d.]+) Pearson ([-\d.]+)")
RE_PAIR = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) f1 ([-\d.]+)")
RE_SINGLE = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) mcc ([-\d.]+)")

PAPER_METRIC = {
    "cola": "mcc", "sst2": "acc", "stsb": "spearman", "mrpc": "f1",
    "qqp": "f1", "mnli": "acc", "qnli": "acc", "rte": "acc", "wnli": "acc",
    "imdb": "acc",
}

# Each arm: the three orthogonal HyperGLOT switches plus which knob controls
# graph sparsity. Every arm gets a 5-point grid so the tuning budget is equal.
#
# The Poincare threshold arms use `rho_quantile`, NOT absolute `rho`. Measured on
# real CoLA features (see diagnose_graph_density.py): BERT token norms ~14.7
# saturate expmap0 onto the ball boundary, real pairwise distances land in
# [8.85, 11.76], and the old grid rho in {0.5..3.0} produced ZERO edges on 16/16
# sentences -- every threshold arm was silently training on an EMPTY graph, which
# is exactly why their scores were bit-identical across the whole rho grid.
#
# The quantile grid is chosen to DENSITY-MATCH the cosine tau grid, so the
# comparison isolates geometry from sparsity:
#     cosine tau  0.0   0.2   0.4   0.6    0.8
#     density     0.97  0.69  0.38  0.099  0.025
QUANTILE_GRID = [0.025, 0.10, 0.38, 0.69, 0.97]
TAU_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
KNN_GRID = [1, 2, 4, 8, 16]

ARMS = {
    "baseline":  dict(graph_metric="cosine",   graph_adj="threshold", hgnn=0, hread=0,
                      knob="tau",   grid=TAU_GRID),
    "A_thresh":  dict(graph_metric="poincare", graph_adj="threshold", hgnn=0, hread=0,
                      knob="rho_quantile", grid=QUANTILE_GRID),
    "A_knn":     dict(graph_metric="poincare", graph_adj="knn",       hgnn=0, hread=0,
                      knob="knn_k", grid=KNN_GRID),
    "B_thresh":  dict(graph_metric="poincare", graph_adj="threshold", hgnn=0, hread=1,
                      knob="rho_quantile", grid=QUANTILE_GRID,
                      extra={"readout_clip": 0.7, "readout_scale": 1, "learnable_curvature": 1}),
    "C_thresh":  dict(graph_metric="cosine",   graph_adj="threshold", hgnn=1, hread=0,
                      knob="tau",   grid=TAU_GRID,
                      extra={"hyp_gnn_type": "gat", "gnn_input_clip": 0.7, "gnn_input_scale": 1}),
    "AC_thresh": dict(graph_metric="poincare", graph_adj="threshold", hgnn=1, hread=0,
                      knob="rho_quantile", grid=QUANTILE_GRID,
                      extra={"hyp_gnn_type": "gat", "gnn_input_clip": 0.7, "gnn_input_scale": 1}),
    "ABC_thresh": dict(graph_metric="poincare", graph_adj="threshold", hgnn=1, hread=1,
                       knob="rho_quantile", grid=QUANTILE_GRID,
                       extra={"hyp_gnn_type": "gat", "gnn_input_clip": 0.7, "gnn_input_scale": 1,
                              "readout_clip": 0.7, "readout_scale": 1, "learnable_curvature": 1}),
}

CSV_FIELDS = [
    "timestamp", "stage", "task", "arm", "knob", "knob_value", "seed",
    "metric", "score", "graph_metric", "graph_adj", "hyperbolic_gnn",
    "hyperbolic_readout", "epochs", "gat_hidden_dim", "num_layers", "proj_dim",
    "scorer_hidden", "lr", "batch_size", "elapsed_sec",
]


def run_once(task, arm_name, spec, knob_value, seed, stage, args):
    cfg = dict(spec)
    extra = cfg.get("extra", {})
    max_length = "512" if task == "imdb" else "128"

    # sparsity knob defaults; the tuned one is overwritten below
    knobs = {"tau": 0.6, "rho": 1.0, "rho_quantile": -1.0, "knn_k": 8}
    knobs[cfg["knob"]] = knob_value

    cmd = [
        sys.executable, MAIN,
        f"--model_name_or_path={args.model}",
        "--decoder_cls_last_token=0",
        f"--task={task}",
        f"--max_length={max_length}",
        "--adaptive_length=0",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        "--eval_batch_size=64",
        f"--lr={args.lr}",
        f"--weight_decay={args.weight_decay}",
        f"--seed={seed}",
        "--verbose=1",
        "--pooling_method=glot",
        "--gnn_type=gat",
        f"--scorer_hidden={args.scorer_hidden}",
        f"--gat_hidden_dim={args.gat_hidden_dim}",
        f"--num_layers={args.num_layers}",
        "--jk_mode=cat",
        f"--graph_metric={cfg['graph_metric']}",
        f"--graph_adj={cfg['graph_adj']}",
        f"--hyperbolic_gnn={cfg['hgnn']}",
        f"--hyperbolic_readout={cfg['hread']}",
        f"--tau={knobs['tau']}",
        f"--rho={knobs['rho']}",
        f"--rho_quantile={knobs['rho_quantile']}",
        f"--knn_k={knobs['knn_k']}",
        f"--curvature={args.curvature}",
        f"--proj_dim={args.proj_dim}",
        "--precompute_hidden_states=1",
        "--override_precompute=0",          # caches MUST be pre-warmed
        "--finetune_backbone=0",
        f"--arm={arm_name}",
    ]
    for k, v in extra.items():
        cmd.append(f"--{k}={v}")

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    per_epoch = []
    assert proc.stdout is not None
    tail = []
    for line in proc.stdout:
        tail.append(line)
        if len(tail) > 25:
            tail.pop(0)
        for rx, keys in ((RE_STS, ("spearman", "pearson")),
                         (RE_PAIR, ("acc", "f1")),
                         (RE_SINGLE, ("acc", "mcc"))):
            m = rx.search(line)
            if m:
                per_epoch.append({keys[0]: float(m.group(2)), keys[1]: float(m.group(3))})
                break
    proc.wait()
    elapsed = round(time.time() - t0, 2)

    if proc.returncode != 0 or not per_epoch:
        print(f"  [FAIL] {arm_name} {cfg['knob']}={knob_value} seed={seed} "
              f"rc={proc.returncode}", flush=True)
        for ln in tail:
            print("    | " + ln.rstrip(), flush=True)
        return None

    key = PAPER_METRIC[task]
    best = max(per_epoch, key=lambda e: e.get(key, float("-inf")))
    score = best.get(key, float("nan")) * 100.0

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage,
        "task": task, "arm": arm_name, "knob": cfg["knob"],
        "knob_value": knob_value, "seed": seed, "metric": key,
        "score": round(score, 2),
        "graph_metric": cfg["graph_metric"], "graph_adj": cfg["graph_adj"],
        "hyperbolic_gnn": cfg["hgnn"], "hyperbolic_readout": cfg["hread"],
        "epochs": args.epochs, "gat_hidden_dim": args.gat_hidden_dim,
        "num_layers": args.num_layers, "proj_dim": args.proj_dim,
        "scorer_hidden": args.scorer_hidden, "lr": args.lr,
        "batch_size": args.batch_size, "elapsed_sec": elapsed,
    }
    print(f"  [{stage}] {arm_name:11s} {cfg['knob']}={knob_value:<5} seed={seed:<3} "
          f"{key}={score:6.2f}  [{elapsed}s]", flush=True)
    return row


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["cola", "stsb", "rte"])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2],
                   help="Stage-2 confirmation seeds (stage 1 always uses the first).")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gat_hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--scorer_hidden", type=int, default=128)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--out", default=os.path.join(HERE, "results", "ablation_fair.csv"))
    args = p.parse_args()

    tune_seed = args.seeds[0]
    done = {(r["stage"], r["task"], r["arm"], r["knob_value"], r["seed"])
            for r in load_rows(args.out)}

    for task in args.tasks:
        print(f"\n{'=' * 86}\nTASK {task}\n{'=' * 86}", flush=True)

        # ---- Stage 1: tune each arm's sparsity knob, equal budget ----
        print(f"-- stage 1: tune sparsity knob (seed {tune_seed}) --", flush=True)
        for arm in args.arms:
            spec = ARMS[arm]
            for val in spec["grid"]:
                key = ("tune", task, arm, str(val), str(tune_seed))
                if key in done:
                    continue
                row = run_once(task, arm, spec, val, tune_seed, "tune", args)
                if row:
                    append_csv(args.out, row)

        # pick best knob per arm
        rows = [r for r in load_rows(args.out)
                if r["stage"] == "tune" and r["task"] == task]
        best_knob = {}
        for arm in args.arms:
            cand = [r for r in rows if r["arm"] == arm]
            if cand:
                b = max(cand, key=lambda r: float(r["score"]))
                best_knob[arm] = b["knob_value"]
                print(f"   best {arm:11s} {b['knob']}={b['knob_value']:<6} "
                      f"{b['metric']}={float(b['score']):.2f}", flush=True)

        # ---- Stage 2: confirm best config over the remaining seeds ----
        print(f"-- stage 2: confirm best config over seeds {args.seeds} --", flush=True)
        for arm in args.arms:
            if arm not in best_knob:
                continue
            spec = ARMS[arm]
            raw = best_knob[arm]
            val = int(raw) if spec["knob"] == "knn_k" else float(raw)
            for seed in args.seeds:
                key = ("final", task, arm, str(val), str(seed))
                if key in done:
                    continue
                row = run_once(task, arm, spec, val, seed, "final", args)
                if row:
                    append_csv(args.out, row)

    report(args)


def report(args):
    rows = [r for r in load_rows(args.out) if r["stage"] == "final"]
    if not rows:
        return
    print("\n" + "=" * 96)
    print("FAIR ABLATION -- best config per arm, averaged over seeds")
    print("=" * 96)
    for task in sorted({r["task"] for r in rows}):
        trows = [r for r in rows if r["task"] == task]
        metric = trows[0]["metric"]
        print(f"\n{task}  ({metric})")
        print(f"  {'arm':12s} {'knob':>12s} {'n':>3s} {'mean':>8s} {'std':>7s} "
              f"{'vs base':>9s} {'sigmas':>8s}")
        print("  " + "-" * 76)
        by_arm = {}
        for arm in sorted({r["arm"] for r in trows}):
            vals = [float(r["score"]) for r in trows if r["arm"] == arm]
            knob = next(f"{r['knob']}={r['knob_value']}" for r in trows if r["arm"] == arm)
            by_arm[arm] = (vals, knob)
        base_vals = by_arm.get("baseline", ([], ""))[0]
        base_mean = statistics.mean(base_vals) if base_vals else None
        # pooled std across arms = our noise yardstick
        pooled = [statistics.stdev(v) for v, _ in by_arm.values() if len(v) > 1]
        pooled_std = statistics.mean(pooled) if pooled else float("nan")
        for arm, (vals, knob) in by_arm.items():
            if not vals:
                continue
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if base_mean is None or arm == "baseline":
                d, sig = "", ""
            else:
                delta = m - base_mean
                d = f"{delta:+.2f}"
                sig = f"{delta / pooled_std:+.1f}" if pooled_std and pooled_std == pooled_std else "n/a"
            print(f"  {arm:12s} {knob:>12s} {len(vals):>3d} {m:>8.2f} {s:>7.2f} "
                  f"{d:>9s} {sig:>8s}")
        print(f"  pooled seed std = {pooled_std:.2f}  "
              f"(treat |delta| < 2*std as not established)")
    print("=" * 96)


if __name__ == "__main__":
    main()
