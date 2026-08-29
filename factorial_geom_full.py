#!/usr/bin/env python3
"""The density x curvature factorial on CoLA, ALL FOUR CELLS, at high n.

WHY A NEW FILE INSTEAD OF EDITING factorial_geom.py
  factorial_geom.py ran only the two OFF-DIAGONAL cells, because the other two
  (baseline, C) already existed in results/campaign_wide_cola.csv. That was the
  cheap way to build the 2x2 and it was correct at the time.

  It cannot be extended in place. The existing baseline and C cells were run in
  a different campaign, and pooling seeds across campaigns would put a
  cache-state / machine difference inside a paired delta -- the same class of
  error that already cost this project 5 MCC on CoLA. So every cell is re-run
  here, on ONE machine, over ONE seed range, and the resulting 2x2 is
  internally consistent and stands on its own.

WHY IT IS WORTH ~11 GPU-HOURS
  At n=15 the decomposition of Stage C's +1.42 MCC is exact and useless:
      geometry     +0.400  (8/7,  p=1.00)
      configuration -0.012  (10/5, p=0.30)
      interaction  +1.028  (10/5, p=0.30)
      TOTAL        +1.416  (13/2, p=0.0074)  <- significant
  Each component differences two independently trained cells, so it carries
  2-3x the variance of the total. At the observed effect and spread, resolving
  the interaction at 80% power needs ~65 seeds. This is the single experiment
  that could turn the paper's "real but unattributed" into an attribution.

  It may also come back null. That is a fine outcome and must be reported as
  one: do NOT re-run with a different seed count after seeing the answer.

usage:  factorial_geom_full.py [n_seeds]        (default 65)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from campaign import run_one                      # noqa: E402
from exp_runner import ResultsCSV, PAPER_METRIC   # noqa: E402

TARGET, MODEL, TASK = "glue", "bert-base-uncased", "cola"
SETTING = "cola_factorial65"
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 65
SEEDS = list(range(1, N_SEEDS + 1))
OUT = os.path.join(HERE, "results", f"factorial_geom_full_cola_n{N_SEEDS}.csv")

# exact confirmed configs, transcribed in factorial_geom.py from
# results/campaign_wide_cola.csv and unchanged here.
SHARED = {"graph_metric": "cosine", "jk_mode": "max", "lr": "0.001",
          "scorer_hidden": "128", "weight_decay": "5e-05"}
BASE_CFG = {**SHARED, "gat_hidden_dim": "256", "num_layers": "4",
            "proj_dim": "128", "tau_quantile": "0.1"}
C_CFG = {**SHARED, "gat_hidden_dim": "128", "num_layers": "2",
         "proj_dim": "256", "tau_quantile": "0.38"}
# knobs that exist only when hyperbolic_gnn=1, at C's own tuned values --
# the choice most favourable to C, kept deliberately.
HYP = {"curvature": "4.0", "gnn_input_clip": "0.7", "gnn_input_scale": "1",
       "hyp_gnn_type": "gat", "hyperbolic_gnn": "1"}

CELLS = {
    "base_euclid": (dict(BASE_CFG), 910),          # = the published baseline
    "base_hyp":    ({**BASE_CFG, **HYP}, 911),     # geometry alone
    "Ccfg_euclid": (dict(C_CFG), 912),             # configuration alone
    "Ccfg_hyp":    ({**C_CFG, **HYP}, 913),        # = Stage C
}


def main():
    res = ResultsCSV(OUT)
    # seed-major order: if the job is cut short, every cell has the SAME
    # completed seed set and the 2x2 is still paired and usable.
    todo = [(c, seed) for seed in SEEDS for c in CELLS]
    print(f"{len(todo)} runs -> {OUT}", flush=True)
    for i, (cell, seed) in enumerate(todo, 1):
        cfg, trial = CELLS[cell]
        key = f"{TARGET}|{MODEL}|{SETTING}|{cell}|t{trial}|s{seed}|confirm"
        if res.has(key):
            continue
        print(f"[{i}/{len(todo)}] {cell} seed={seed}", flush=True)
        r = run_one(TARGET, cfg, TASK, seed, MODEL, {}, -1)
        if not r:
            print("    !! FAILED", flush=True)
            continue
        res.append(dict(
            run_key=key, target=TARGET, model=MODEL, setting=SETTING,
            task=TASK, arm=cell, stage="confirm", trial=trial, seed=seed,
            metric=PAPER_METRIC[TASK],
            detail=";".join(f"{k}={v}" for k, v in sorted(cfg.items())),
            **cfg, **r))
        print(f"    -> {r['score']:.2f}  density={r['mean_density']} "
              f"[{r['elapsed_sec']}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
