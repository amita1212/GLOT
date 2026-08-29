#!/usr/bin/env python3
"""Re-run an already-confirmed arm on MORE seeds, or on a DIFFERENT task.

WHY THIS EXISTS
  Two of the paper's three open questions are open only because n=15.
    * Stage C's +1.42 MCC on CoLA decomposes into geometry / configuration /
      interaction, and NONE of the three parts is significant even though the
      total is. Resolving the interaction at 80% power needs ~65 seeds.
    * Stage A trends positive on CoLA (+0.70, 11/4) at p=0.118 and settles
      nothing.
  Both are cheap on BERT/CoLA (~152 s a run). Re-tuning would answer a
  different question, so this script does NOT tune: it lifts the exact
  confirmed configuration out of a finished campaign CSV and replays it.

WHY IT READS THE CONFIG FROM THE CSV INSTEAD OF A HARD-CODED DICT
  Hand-transcribing a config is how you silently compare two different models.
  The `detail` column is the authoritative record of what actually ran.

CROSS-MACHINE WARNING
  Every result in this paper is a PAIRED difference. Cache-warming order alone
  moved CoLA by ~5 MCC, so a delta whose two halves ran on different VMs
  absorbs the hardware difference. Therefore:
    - run ALL arms of a comparison on ONE machine, and
    - when extending seeds, re-run seeds 1..15 on that machine too
      (--from_seed 1) instead of pooling old and new seeds.
  The default below re-runs from seed 1 for exactly this reason.

usage:
  seed_extend.py --src results/campaign_wide_cola.csv \\
                 --arms baseline A --seeds 1 50 --out results/seedext_A_cola.csv
  seed_extend.py --src results/campaign_wide_cola.csv \\
                 --arms baseline B C BC --task sst2 --seeds 1 15 \\
                 --out results/sst2_reduced.csv
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from campaign import run_one                      # noqa: E402
from exp_runner import ResultsCSV, PAPER_METRIC   # noqa: E402

# keys in `detail` that describe the RUN, not the MODEL. Replaying them would
# pin the new run to the old seed/task and defeat the point.
NOT_CONFIG = {"seed", "task", "model", "model_name_or_path", "target",
              "arm", "stage", "trial", "verbose", "score", "elapsed_sec",
              "mean_density"}


def confirmed_config(src, arm):
    """The config an arm actually ran at confirmation, from the campaign log."""
    best = None
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("arm") != arm:
                continue
            if r.get("stage") != "confirm":
                continue
            best = r
            break                      # every confirm row of an arm is the
                                       # same config, by construction
    if best is None:
        raise SystemExit(f"no confirm-stage row for arm {arm!r} in {src}")
    cfg = {}
    for kv in (best.get("detail") or "").split(";"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k not in NOT_CONFIG:
            cfg[k] = v
    if not cfg:
        raise SystemExit(f"empty config parsed from detail for arm {arm!r}")
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="finished campaign CSV")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--task", default=None, help="default: the source task")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--target", default="glue")
    p.add_argument("--seeds", nargs=2, type=int, default=[1, 65],
                   metavar=("FIRST", "LAST"))
    p.add_argument("--hidden_layer", type=int, default=-1)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    with open(a.src, newline="") as f:
        src_rows = list(csv.DictReader(f))
    src_task = src_rows[0]["task"] if src_rows else None
    task = a.task or src_task
    if task is None:
        raise SystemExit("could not infer task; pass --task")

    seeds = list(range(a.seeds[0], a.seeds[1] + 1))
    setting = f"seedext_{task}"
    cfgs = {arm: confirmed_config(a.src, arm) for arm in a.arms}

    print(f"src      : {a.src}  (task {src_task})")
    print(f"replaying: {', '.join(a.arms)}")
    print(f"on task  : {task}   model {a.model}")
    print(f"seeds    : {seeds[0]}..{seeds[-1]}  ({len(seeds)} per arm, "
          f"{len(seeds) * len(a.arms)} runs)")
    for arm, cfg in cfgs.items():
        print(f"  {arm:10s} {';'.join(f'{k}={v}' for k, v in sorted(cfg.items()))}")
    sys.stdout.flush()

    res = ResultsCSV(a.out)
    todo = [(arm, s) for arm in a.arms for s in seeds]
    for i, (arm, seed) in enumerate(todo, 1):
        cfg = cfgs[arm]
        key = f"{a.target}|{a.model}|{setting}|{arm}|t0|s{seed}|confirm"
        if res.has(key):
            continue
        print(f"[{i}/{len(todo)}] {arm} seed={seed}", flush=True)
        r = run_one(a.target, cfg, task, seed, a.model, {}, a.hidden_layer)
        if not r:
            print("    !! FAILED", flush=True)
            continue
        res.append(dict(
            run_key=key, target=a.target, model=a.model, setting=setting,
            task=task, arm=arm, stage="confirm", trial=0, seed=seed,
            metric=PAPER_METRIC[task],
            detail=";".join(f"{k}={v}" for k, v in sorted(cfg.items())),
            **cfg, **r))
        print(f"    -> {r['score']:.2f}  [{r['elapsed_sec']}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
