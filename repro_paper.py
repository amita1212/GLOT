#!/usr/bin/env python
"""Phase R driver: reproduce the GLOT paper's numbers with pristine upstream code.

Runs ``glot_original/main.py`` (untouched except the mandatory
``glue`` -> ``nyu-mll/glue`` dataset-alias fix required by datasets>=4) and
parses the per-epoch metric lines it prints, recording the best epoch --
matching how the paper reports "best" scores.

Paper config (Appendix B.2 + "Training Details"):
    frozen backbone, Adam, lr 2e-4, weight decay 0.0, batch 32, eval batch 64,
    seed 42, 2 epochs, GATConv K=2, hidden 128, jk=cat, precomputed hidden states.

Targets
-------
Table 8 (BERT, tau sensitivity) -- the primary reproduction target, because it
shows Table 1 uses the *best tau per task*:

    tau     STS-B(Spear)  CoLA(MCC)  RTE(ACC)
    0.0     81.88         39.62      50.90
    0.2     82.12         39.82      50.90
    0.4     82.25         47.49      52.34
    0.6     83.86         43.16      59.21
    0.8     83.85         43.16      52.70

Table 1 (BERT, GLOT row):
    CoLA 47.49 | SST-2 90.25 | STS-B 83.86 | MRPC 82.58 (F1) | QQP 62.19 (F1)
    MNLI-m 54.39 | MNLI-mm 54.47 | QNLI 61.08 | RTE 59.21 | WNLI 54.93
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "glot_original", "main.py")

# --- per-epoch stdout formats emitted by upstream main.py --------------------
RE_STS = re.compile(r"epoch (\d+) MSE [\d.]+ Spearman ([-\d.]+) Pearson ([-\d.]+)")
RE_PAIR = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) f1 ([-\d.]+)")
RE_SINGLE = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) mcc ([-\d.]+)")

# Which metric the paper reports for each task (Table 1 caption).
PAPER_METRIC = {
    "cola": "mcc", "sst2": "acc", "stsb": "spearman", "mrpc": "f1",
    "qqp": "f1", "mnli": "acc", "qnli": "acc", "rte": "acc", "wnli": "acc",
    "imdb": "acc",
}

# Paper Table 1, BERT / GLOT row (scores x100).
PAPER_TABLE1_BERT = {
    "cola": 47.49, "sst2": 90.25, "stsb": 83.86, "mrpc": 82.58, "qqp": 62.19,
    "mnli": 54.39, "qnli": 61.08, "rte": 59.21, "wnli": 54.93,
}

# Paper Table 8, BERT: tau -> {task: score}
PAPER_TABLE8_BERT = {
    0.0: {"stsb": 81.88, "cola": 39.62, "rte": 50.90},
    0.2: {"stsb": 82.12, "cola": 39.82, "rte": 50.90},
    0.4: {"stsb": 82.25, "cola": 47.49, "rte": 52.34},
    0.6: {"stsb": 83.86, "cola": 43.16, "rte": 59.21},
    0.8: {"stsb": 83.85, "cola": 43.16, "rte": 52.70},
}

CSV_FIELDS = [
    "timestamp", "task", "tau", "metric", "score", "paper", "delta",
    "acc", "f1", "mcc", "spearman", "pearson",
    "epochs", "gat_hidden_dim", "num_layers", "proj_dim", "scorer_hidden",
    "lr", "weight_decay", "batch_size", "seed", "max_length", "elapsed_sec",
]


def run_one(task: str, tau: float, seed: int, args) -> dict:
    """Run upstream main.py once and return the best-epoch metrics."""
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
        "--graph_adj=threshold",
        f"--tau={tau}",
        f"--proj_dim={args.proj_dim}",
        "--precompute_hidden_states=1",
        f"--override_precompute={args.override_precompute}",
        "--finetune_backbone=0",
    ]

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"          # no-op wandb; does not affect numerics
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(f"\n{'=' * 78}\n[RUN] task={task} tau={tau} seed={seed}\n{'=' * 78}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    per_epoch: list[dict] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if args.stream:
            print(line, flush=True)
        m = RE_STS.search(line)
        if m:
            per_epoch.append({"spearman": float(m.group(2)), "pearson": float(m.group(3))})
            if not args.stream:
                print("   " + line, flush=True)
            continue
        m = RE_PAIR.search(line)
        if m:
            per_epoch.append({"acc": float(m.group(2)), "f1": float(m.group(3))})
            if not args.stream:
                print("   " + line, flush=True)
            continue
        m = RE_SINGLE.search(line)
        if m:
            per_epoch.append({"acc": float(m.group(2)), "mcc": float(m.group(3))})
            if not args.stream:
                print("   " + line, flush=True)
    proc.wait()
    elapsed = round(time.time() - t0, 2)

    if proc.returncode != 0 or not per_epoch:
        print(f"[FAIL] task={task} tau={tau} seed={seed} rc={proc.returncode} "
              f"epochs_parsed={len(per_epoch)}", flush=True)
        return {}

    key = PAPER_METRIC[task]
    best_idx = max(range(len(per_epoch)), key=lambda i: per_epoch[i].get(key, float("-inf")))
    best = per_epoch[best_idx]
    score = best.get(key, float("nan")) * 100.0
    paper = PAPER_TABLE8_BERT.get(tau, {}).get(task) or PAPER_TABLE1_BERT.get(task)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task, "tau": tau, "metric": key,
        "score": round(score, 2),
        "paper": paper if paper is not None else "",
        "delta": round(score - paper, 2) if paper is not None else "",
        "acc": round(best.get("acc", float("nan")) * 100, 2) if "acc" in best else "",
        "f1": round(best.get("f1", float("nan")) * 100, 2) if "f1" in best else "",
        "mcc": round(best.get("mcc", float("nan")) * 100, 2) if "mcc" in best else "",
        "spearman": round(best.get("spearman", float("nan")) * 100, 2) if "spearman" in best else "",
        "pearson": round(best.get("pearson", float("nan")) * 100, 2) if "pearson" in best else "",
        "epochs": args.epochs, "gat_hidden_dim": args.gat_hidden_dim,
        "num_layers": args.num_layers, "proj_dim": args.proj_dim,
        "scorer_hidden": args.scorer_hidden, "lr": args.lr,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "seed": seed, "max_length": max_length, "elapsed_sec": elapsed,
    }
    tag = f"paper {paper}  delta {row['delta']:+.2f}" if paper is not None else "no paper ref"
    print(f"[DONE] {task} tau={tau} seed={seed}  {key}={score:.2f}  ({tag})  [{elapsed}s]",
          flush=True)
    return row


def append_csv(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["task"], r["tau"], r["seed"]) for r in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["cola", "stsb", "rte"])
    p.add_argument("--taus", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8])
    p.add_argument("--model", default="bert-base-uncased")
    # Paper defaults (Appendix B.2 + Training Details)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gat_hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--scorer_hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--override_precompute", type=int, default=0,
                   help="1 = rebuild the hidden-state cache every run (COLD cache). "
                        "Upstream advances the global RNG while building the cache but not "
                        "when loading it, so cold and warm runs give different results for "
                        "the SAME seed. Use this to isolate that effect.")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Run each (task, tau) once per seed. Overrides --seed. "
                        "Use repeated identical values to measure pure run-to-run variance.")
    p.add_argument("--out", default=os.path.join(HERE, "results", "repro_table8.csv"))
    p.add_argument("--stream", action="store_true", help="echo full child stdout")
    p.add_argument("--resume", action="store_true", help="skip (task, tau, seed) already in the CSV")
    args = p.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]

    done = load_done(args.out) if args.resume else set()
    rows = []
    for task in args.tasks:
        for tau in args.taus:
            for seed in seeds:
                if (task, str(tau), str(seed)) in done:
                    print(f"[SKIP] {task} tau={tau} seed={seed} (already done)", flush=True)
                    continue
                row = run_one(task, tau, seed, args)
                if row:
                    append_csv(args.out, row)
                    rows.append(row)

    if not rows:
        return
    print("\n" + "=" * 86)
    print(f"{'task':6s} {'tau':>5s} {'seed':>5s} {'metric':>9s} {'ours':>8s} {'paper':>8s} {'delta':>8s}")
    print("-" * 86)
    for r in rows:
        paper = f"{r['paper']:.2f}" if r["paper"] != "" else "   n/a"
        delta = f"{r['delta']:+.2f}" if r["delta"] != "" else "   n/a"
        print(f"{r['task']:6s} {r['tau']:>5} {r['seed']:>5} {r['metric']:>9s} "
              f"{r['score']:>8.2f} {paper:>8s} {delta:>8s}")
    deltas = [abs(r["delta"]) for r in rows if r["delta"] != ""]
    if deltas:
        print("-" * 86)
        print(f"mean |delta| = {sum(deltas) / len(deltas):.2f}   max |delta| = {max(deltas):.2f}")

    # Spread across repeats of the same (task, tau) -- the noise floor.
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["task"], r["tau"]), []).append(r["score"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        print("\n" + "=" * 86)
        print("RUN-TO-RUN SPREAD (same config, repeated)")
        print(f"{'task':6s} {'tau':>5s} {'n':>3s} {'mean':>8s} {'std':>7s} {'min':>8s} {'max':>8s} {'range':>8s}")
        print("-" * 86)
        for (task, tau), vals in sorted(multi.items()):
            n = len(vals)
            mean = sum(vals) / n
            std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
            print(f"{task:6s} {tau:>5} {n:>3d} {mean:>8.2f} {std:>7.2f} "
                  f"{min(vals):>8.2f} {max(vals):>8.2f} {max(vals) - min(vals):>8.2f}")
    print("=" * 86)


if __name__ == "__main__":
    main()
