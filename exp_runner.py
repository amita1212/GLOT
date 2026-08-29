#!/usr/bin/env python
"""
Generic, resumable HyperGLOT experiment runner.

Motivation
----------
Every previous sweep script re-implemented subprocess launching, stdout metric
parsing and CSV resumption, and each copy drifted. This module centralises that
so new experiments are just a list of config dicts.

A "config" is a plain dict of CLI flags for main.py (without the leading `--`).
Anything not given falls back to PAPER_DEFAULTS, which are the ICLR-2026 GLOT
settings (2 epochs, lr 2e-4, wd 0.0, batch 32, hidden 128, jk=cat, frozen
backbone, precomputed hidden states).

Guard rails baked in (see /memories/repo/glot-known-bugs.md):
  * `override_precompute=0` always -> every run uses a WARM cache, so the
    cold-cache RNG confound (worth ~5 MCC on CoLA) cannot reappear.
  * `jk_mode` restricted to {cat, max}: `lstm` is in argparse's choices but
    crashes at runtime, and `mean`/`none` are advertised by the paper but
    rejected by argparse.
  * Results are appended row-by-row and the runner skips configs already
    present, so spot-instance preemption never loses work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "hyperglot", "main.py")
if not os.path.exists(MAIN):                     # running from inside the repo copy
    MAIN = os.path.join(HERE, "main.py")

RE_STS = re.compile(r"epoch (\d+) MSE [\d.]+ Spearman ([-\d.]+) Pearson ([-\d.]+)")
RE_PAIR = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) f1 ([-\d.]+)")
RE_SINGLE = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) mcc ([-\d.]+)")
RE_DENSITY = re.compile(r"\[graph\] density=([\d.]+) empty_frac=([\d.]+)")

PAPER_METRIC = {
    "cola": "mcc", "sst2": "acc", "stsb": "spearman", "mrpc": "f1",
    "qqp": "f1", "mnli": "acc", "qnli": "acc", "rte": "acc", "wnli": "acc",
    "imdb": "acc",
}

# ICLR-2026 GLOT training recipe (paper Appendix B.2), NOT the README's.
PAPER_DEFAULTS = {
    "model_name_or_path": "bert-base-uncased",
    "decoder_cls_last_token": 0,
    "max_length": 128,
    "adaptive_length": 0,
    "epochs": 2,
    "batch_size": 32,
    "eval_batch_size": 64,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "seed": 42,
    "verbose": 1,
    "pooling_method": "glot",
    "gnn_type": "gat",
    "scorer_hidden": 128,
    "gat_hidden_dim": 128,
    "num_layers": 2,
    "jk_mode": "cat",
    "graph_adj": "threshold",
    "tau": 0.6,
    "proj_dim": 256,
    "precompute_hidden_states": 1,
    "override_precompute": 0,     # NEVER change: keeps every run on a warm cache
    "finetune_backbone": 0,
}

ALLOWED_JK = {"cat", "max"}


def config_id(cfg: dict) -> str:
    """Stable hash of the *effective* config, used for resumption."""
    payload = json.dumps({k: str(v) for k, v in sorted(cfg.items())}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def build_cmd(cfg: dict) -> list:
    full = dict(PAPER_DEFAULTS)
    full.update(cfg)
    if str(full.get("jk_mode")) not in ALLOWED_JK:
        raise ValueError(
            f"jk_mode={full.get('jk_mode')!r} is not usable. Released GLOT accepts "
            f"{{cat,lstm,max}} but `lstm` crashes at runtime; the paper's `mean`/`none` "
            f"are rejected by argparse. Effective space is {sorted(ALLOWED_JK)}."
        )
    if int(full.get("override_precompute", 0)) != 0:
        raise ValueError("override_precompute must stay 0 (cold-cache RNG confound).")
    if int(full.get("verbose", 1)) == 0:
        # run_once recovers the score by regexing the per-epoch
        # "epoch N loss .. acc .. mcc .." line out of stdout. verbose=0
        # suppresses exactly that line, so the job trains fine, prints its
        # RESULT_JSON, exits 0 -- and is still recorded as a failure, silently
        # and for every single run. Fail loudly at build time instead.
        raise ValueError("verbose must be 1: run_once parses the per-epoch "
                         "stdout line, which verbose=0 suppresses.")
    cmd = [sys.executable, MAIN]
    cmd += [f"--{k}={v}" for k, v in full.items()]
    return cmd, full


def run_once(cfg: dict, extra_row: dict | None = None, cwd: str = HERE,
             timeout: int | None = None) -> dict | None:
    """Run one training job; return a result row or None if it failed."""
    cmd, full = build_cmd(cfg)
    task = full["task"]
    key = PAPER_METRIC[task]

    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    per_epoch, tail, densities = [], [], []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line)
        if len(tail) > 30:
            tail.pop(0)
        md = RE_DENSITY.search(line)
        if md:
            densities.append((float(md.group(1)), float(md.group(2))))
            continue
        for rx, keys in ((RE_STS, ("spearman", "pearson")),
                         (RE_PAIR, ("acc", "f1")),
                         (RE_SINGLE, ("acc", "mcc"))):
            m = rx.search(line)
            if m:
                per_epoch.append({keys[0]: float(m.group(2)),
                                  keys[1]: float(m.group(3))})
                break
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  [TIMEOUT] {cfg}", flush=True)
        return None
    elapsed = round(time.time() - t0, 2)

    if proc.returncode != 0 or not per_epoch:
        print(f"  [FAIL rc={proc.returncode}] {cfg}", flush=True)
        for ln in tail:
            print("    | " + ln.rstrip(), flush=True)
        return None

    best = max(per_epoch, key=lambda e: e.get(key, float("-inf")))
    score = best.get(key, float("nan")) * 100.0

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_id": config_id(full),
        "task": task,
        "metric": key,
        "score": round(score, 3),
        "elapsed_sec": elapsed,
        "mean_density": round(sum(d for d, _ in densities) / len(densities), 5)
                        if densities else "",
        "empty_frac": round(sum(e for _, e in densities) / len(densities), 5)
                      if densities else "",
    }
    row.update({k: v for k, v in full.items()})
    if extra_row:
        row.update(extra_row)
    return row


class ResultsCSV:
    """Append-only results file with automatic resumption + schema growth."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.rows = self._load()
        self.seen = {r.get("run_key") for r in self.rows if r.get("run_key")}

    def _load(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def has(self, run_key: str) -> bool:
        return run_key in self.seen

    def append(self, row: dict):
        row = dict(row)
        # Field set must be the union over EVERY row seen so far, not just the
        # first one. Arms have heterogeneous config keys (Stage A contributes
        # rho_quantile/feature_mode, Stage B readout_clip, ...), and keying the
        # schema off row 0 silently dropped every column an arm introduced after
        # the file was created -- the campaign CSV ended up without
        # `edge_weight_mode`, `feature_mode` or `graph_curvature` columns at all.
        # The runs themselves were unaffected (the `detail` string is what gets
        # replayed for confirmation), but the results were far harder to audit,
        # and "hard to audit" is how every earlier bug in this project survived.
        prev_fields = [k for r in self.rows for k in r.keys()]
        fields = list(dict.fromkeys(prev_fields + list(row.keys())))
        header_stale = bool(self.rows) and list(dict.fromkeys(prev_fields)) != fields
        self.rows.append(row)
        if header_stale or not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                for r in self.rows:
                    w.writerow(r)
        else:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writerow(row)
        if row.get("run_key"):
            self.seen.add(row["run_key"])


def sweep(configs, out_csv: str, cwd: str = HERE):
    """Run a list of (run_key, cfg, extra_row) triples, skipping finished ones."""
    res = ResultsCSV(out_csv)
    todo = [(k, c, e) for k, c, e in configs if not res.has(k)]
    print(f"{len(configs)} configs, {len(configs) - len(todo)} already done, "
          f"{len(todo)} to run.", flush=True)
    for i, (run_key, cfg, extra) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {run_key}", flush=True)
        extra = dict(extra or {})
        extra["run_key"] = run_key
        row = run_once(cfg, extra, cwd=cwd)
        if row:
            res.append(row)
            print(f"    -> {row['metric']}={row['score']:.2f} "
                  f"density={row['mean_density']} [{row['elapsed_sec']}s]", flush=True)
    return res
