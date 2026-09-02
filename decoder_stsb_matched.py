#!/usr/bin/env python
"""Matched TinyLlama STS-B campaign: baseline, B and C in ONE campaign file.

WHY THIS EXISTS
---------------
The decoder's B and C arms were written to their own results file four days
after the six-arm campaign, and that file contains no baseline. Every delta for
B or C therefore subtracts a cell from a different campaign -- the splice the
paper's own rule forbids, and worse than the factorial splice that taught us
the rule, because no arm is shared between the two files so the drift cannot be
bounded even in principle.

This script re-runs all three arms at the same 15 seeds, through one script,
one code path and one cache state, into one CSV. That makes B-vs-baseline and
C-vs-baseline admissible paired tests on a decoder -- the only architecture
family in the paper outside the two encoders.

CONFIGURATIONS ARE REUSED, NOT RE-TUNED
---------------------------------------
The configs below are transcribed from the confirmed rows of the two existing
campaigns. Reusing a recorded *configuration* is sound: it is a hyper-parameter
choice, not a measurement. What may not be reused is a recorded *score*, which
is exactly what this rerun stops doing.

Note that the decoder search never varied lr, depth or width -- those columns
do not exist in either decoder CSV -- so all three arms take identical
campaign defaults for them. That is matched, but it is a narrower search than
the encoders got, and the paper says so.

SEED-MAJOR ORDER: all three arms at seed 1, then all three at seed 2, ... so a
job killed partway leaves complete comparable blocks rather than complete arms.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from campaign import run_one                      # noqa: E402
from exp_runner import ResultsCSV, PAPER_METRIC   # noqa: E402

TARGET = "glue"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TASK = "stsb"
SETTING = "decoder_stsb_matched"
DEFAULT_OUT = os.path.join(HERE, "results", "campaign_decoder_stsb_matched.csv")

ARMS = {
    # baseline, transcribed from campaign_decoder_stsb.csv
    "baseline": ({
        "graph_metric": "cosine",
        "tau_quantile": "0.025",
    }, 700),
    # B, transcribed from campaign_decoder_stsb_BC.csv
    "B": ({
        "graph_metric": "cosine",
        "tau_quantile": "0.025",
        "curvature": "0.25",
        "hyperbolic_readout": "1",
        "readout_scale": "1",
        "learnable_curvature": "1",
    }, 701),
    # C, transcribed from campaign_decoder_stsb_BC.csv
    "C": ({
        "graph_metric": "cosine",
        "tau_quantile": "0.05",
        "curvature": "1.0",
        "hyperbolic_gnn": "1",
        "hyp_gnn_type": "gcn",
        "gnn_input_clip": "1.0",
    }, 702),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 16)))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    res = ResultsCSV(args.out)
    todo = [(a, ARMS[a][0], ARMS[a][1], s) for s in args.seeds for a in args.arms]
    print(f"arms={args.arms} seeds={len(args.seeds)} runs={len(todo)} "
          f"order=seed-major out={args.out}", flush=True)

    done = 0
    for arm, cfg, trial, seed in todo:
        key = f"{TARGET}|{MODEL}|{SETTING}|{arm}|t{trial}|s{seed}|confirm"
        if res.has(key):
            done += 1
            continue
        print(f"[{done + 1}/{len(todo)}] {arm} seed={seed}", flush=True)
        r = run_one(TARGET, cfg, TASK, seed, MODEL, {}, -1)
        done += 1
        if not r:
            print("    !! FAILED", flush=True)
            continue
        row = dict(run_key=key, target=TARGET, model=MODEL, setting=SETTING,
                   task=TASK, arm=arm, stage="confirm", trial=trial, seed=seed,
                   metric=PAPER_METRIC[TASK],
                   detail=";".join(f"{k}={v}" for k, v in sorted(cfg.items())),
                   **cfg, **r)
        res.append(row)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
