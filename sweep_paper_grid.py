#!/usr/bin/env python
"""Reproduce the paper's numbers the way the paper actually produced them:
a hyperparameter search over the GLOT paper's own Table 6 grid.

Why this script exists
----------------------
The paper's Table 1 / Table 8 numbers are NOT single fixed-config runs. Appendix
B.1 says the final configuration "was determined via a grid search over these
values ... performed consistently across all backbone models and datasets", run
with Weights and Biases. Table 8 then shows the reported CoLA/STS-B/RTE numbers
each come from a DIFFERENT tau -- i.e. Table 1 reports the best cell of a search.

So to reproduce Table 1 we must reproduce the *search*, not a single config.

Table 6 search space (paper, verbatim)
--------------------------------------
    Learning Rate              {1e-3, 2e-4, 2e-5}
    Weight Decay               {0.0, 1e-5, 5e-5}
    GNN Layers (K)             {2, 4}
    GNN Hidden Dimension       {64, 128, 256}
    Jumping Knowledge          {cat, max, mean, none}
    Input Projection Dimension {128, 256, 512}
    Similarity Threshold (tau) {0.1, 0.3, 0.6}

Full grid = 3*3*2*3*4*3*3 = 1944 configs. We support exhaustive enumeration and
random search (`--n_trials`).

IMPORTANT caveat this script measures for you
---------------------------------------------
"Best over N trials" on the *same* split you report is upward-biased whenever
runs are noisy. Given a per-run noise std `sigma`, the expected best-of-N is
roughly `mean + sigma * E[max of N standard normals]`, which for N=100 is about
`mean + 2.5*sigma`. Pass `--noise_std` (measured empirically with
`repro_paper.py --seeds 42 42 42 ...`) and the summary will tell you how much of
any apparent "match" is just selection bias.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "glot_original", "main.py")

RE_STS = re.compile(r"epoch (\d+) MSE [\d.]+ Spearman ([-\d.]+) Pearson ([-\d.]+)")
RE_PAIR = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) f1 ([-\d.]+)")
RE_SINGLE = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) mcc ([-\d.]+)")

PAPER_METRIC = {
    "cola": "mcc", "sst2": "acc", "stsb": "spearman", "mrpc": "f1",
    "qqp": "f1", "mnli": "acc", "qnli": "acc", "rte": "acc", "wnli": "acc",
    "imdb": "acc",
}

PAPER_TABLE1_BERT = {
    "cola": 47.49, "sst2": 90.25, "stsb": 83.86, "mrpc": 82.58, "qqp": 62.19,
    "mnli": 54.39, "qnli": 61.08, "rte": 59.21, "wnli": 54.93,
}

# Paper Table 6, verbatim -- EXCEPT jk_mode.
#
# The paper lists Jumping Knowledge {cat, max, mean, none}, but the released
# code only accepts choices=["cat", "lstm", "max"] (see main.py --jk_mode).
# Passing "mean" or "none" makes argparse exit with rc=2, which silently killed
# ~25% of the sweep trials. We therefore search the values the code actually
# implements, and note the paper/code discrepancy here.
SEARCH_SPACE = {
    "lr": [1e-3, 2e-4, 2e-5],
    "weight_decay": [0.0, 1e-5, 5e-5],
    "num_layers": [2, 4],
    "gat_hidden_dim": [64, 128, 256],
    "jk_mode": ["cat", "max", "lstm"],
    "proj_dim": [128, 256, 512],
    "tau": [0.1, 0.3, 0.6],
}

CSV_FIELDS = [
    "timestamp", "trial", "task", "metric", "score", "paper", "delta",
    "acc", "f1", "mcc", "spearman", "pearson",
    "lr", "weight_decay", "num_layers", "gat_hidden_dim", "jk_mode",
    "proj_dim", "tau", "scorer_hidden", "epochs", "batch_size", "seed",
    "max_length", "elapsed_sec",
]


def enumerate_configs(space: dict, n_trials: int | None, seed: int) -> list[dict]:
    keys = list(space)
    all_cfgs = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
    if n_trials is None or n_trials >= len(all_cfgs):
        return all_cfgs
    rng = random.Random(seed)
    return rng.sample(all_cfgs, n_trials)


def run_trial(trial: int, task: str, cfg: dict, args) -> dict:
    max_length = "512" if task == "imdb" else "128"
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
        f"--lr={cfg['lr']}",
        f"--weight_decay={cfg['weight_decay']}",
        f"--seed={args.seed}",
        "--verbose=1",
        "--pooling_method=glot",
        "--gnn_type=gat",
        f"--scorer_hidden={args.scorer_hidden}",
        f"--gat_hidden_dim={cfg['gat_hidden_dim']}",
        f"--num_layers={cfg['num_layers']}",
        f"--jk_mode={cfg['jk_mode']}",
        "--graph_adj=threshold",
        f"--tau={cfg['tau']}",
        f"--proj_dim={cfg['proj_dim']}",
        "--precompute_hidden_states=1",
        "--override_precompute=0",
        "--finetune_backbone=0",
    ]

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    per_epoch: list[dict] = []
    assert proc.stdout is not None
    for line in proc.stdout:
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
        print(f"[FAIL {trial:>4d}] rc={proc.returncode} cfg={json.dumps(cfg)}", flush=True)
        return {}

    key = PAPER_METRIC[task]
    best = max(per_epoch, key=lambda e: e.get(key, float("-inf")))
    score = best.get(key, float("nan")) * 100.0
    paper = PAPER_TABLE1_BERT.get(task)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trial": trial, "task": task, "metric": key, "score": round(score, 2),
        "paper": paper if paper is not None else "",
        "delta": round(score - paper, 2) if paper is not None else "",
        "acc": round(best["acc"] * 100, 2) if "acc" in best else "",
        "f1": round(best["f1"] * 100, 2) if "f1" in best else "",
        "mcc": round(best["mcc"] * 100, 2) if "mcc" in best else "",
        "spearman": round(best["spearman"] * 100, 2) if "spearman" in best else "",
        "pearson": round(best["pearson"] * 100, 2) if "pearson" in best else "",
        "scorer_hidden": args.scorer_hidden, "epochs": args.epochs,
        "batch_size": args.batch_size, "seed": args.seed,
        "max_length": max_length, "elapsed_sec": elapsed,
        **cfg,
    }
    flag = ""
    if paper is not None:
        flag = "  <-- MATCHES/BEATS PAPER" if score >= paper else ""
    print(f"[{trial:>4d}] {key}={score:6.2f}  lr={cfg['lr']:<7} wd={cfg['weight_decay']:<7} "
          f"K={cfg['num_layers']} h={cfg['gat_hidden_dim']:<4} jk={cfg['jk_mode']:<5} "
          f"proj={cfg['proj_dim']:<4} tau={cfg['tau']}  [{elapsed}s]{flag}", flush=True)
    return row


def append_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def expected_max_of_n(n: int) -> float:
    """E[max of n standard normals], Blom's approximation."""
    if n < 2:
        return 0.0
    # inverse normal CDF via Acklam-free approximation using erfinv
    p = (n - 0.375) / (n + 0.25)
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(y: float) -> float:
    a = 0.147
    ln = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(t1 * t1 - ln / a) - t1), y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cola")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--n_trials", type=int, default=None,
                   help="Random-search budget. Omit for the full 1944-config grid.")
    p.add_argument("--sample_seed", type=int, default=0, help="RNG seed for trial sampling.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--scorer_hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42, help="Training seed (paper uses 42).")
    p.add_argument("--noise_std", type=float, default=None,
                   help="Empirical per-run std of the metric; enables selection-bias analysis.")
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    out = args.out or os.path.join(HERE, "results", f"sweep_paper_grid_{args.task}.csv")

    done_trials = set()
    if args.resume and os.path.exists(out):
        with open(out, newline="", encoding="utf-8") as f:
            done_trials = {int(r["trial"]) for r in csv.DictReader(f)}

    configs = enumerate_configs(SEARCH_SPACE, args.n_trials, args.sample_seed)
    print(f"Task={args.task}  trials={len(configs)}  "
          f"(full grid = {math.prod(len(v) for v in SEARCH_SPACE.values())})", flush=True)

    rows = []
    for i, cfg in enumerate(configs):
        if i in done_trials:
            continue
        row = run_trial(i, args.task, cfg, args)
        if row:
            append_csv(out, row)
            rows.append(row)

    # Re-read everything so --resume runs still get a full summary.
    with open(out, newline="", encoding="utf-8") as f:
        allrows = [r for r in csv.DictReader(f)]
    if not allrows:
        return
    scores = [float(r["score"]) for r in allrows]
    best = max(allrows, key=lambda r: float(r["score"]))
    paper = PAPER_TABLE1_BERT.get(args.task)
    n = len(scores)
    mean = sum(scores) / n
    std = (sum((s - mean) ** 2 for s in scores) / max(1, n - 1)) ** 0.5

    print("\n" + "=" * 86)
    print(f"SWEEP SUMMARY  task={args.task}  metric={best['metric']}  trials={n}")
    print("-" * 86)
    print(f"best   = {float(best['score']):.2f}")
    print(f"mean   = {mean:.2f}   std across configs = {std:.2f}")
    print(f"paper  = {paper if paper is not None else 'n/a'}")
    if paper is not None:
        print(f"delta  = {float(best['score']) - paper:+.2f}")
        n_beat = sum(1 for s in scores if s >= paper)
        print(f"{n_beat}/{n} trials reach or beat the paper's number")
    print("-" * 86)
    print("best config:")
    for k in ("lr", "weight_decay", "num_layers", "gat_hidden_dim", "jk_mode", "proj_dim", "tau"):
        print(f"    {k:18s} {best[k]}")
    if args.noise_std:
        bias = args.noise_std * expected_max_of_n(n)
        print("-" * 86)
        print(f"SELECTION-BIAS CHECK (noise_std={args.noise_std:.2f}, N={n})")
        print(f"  expected best-of-N inflation from noise alone: +{bias:.2f}")
        print(f"  noise-corrected best estimate: {float(best['score']) - bias:.2f}")
        if paper is not None and float(best["score"]) - bias < paper:
            print("  => the apparent match is within selection bias; NOT a real reproduction.")
    print("=" * 86)


if __name__ == "__main__":
    main()
