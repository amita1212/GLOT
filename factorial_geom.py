"""2x2: {graph+architecture} x {Euclidean, hyperbolic message passing}, CoLA.

The wide campaign let every arm tune its own full config, so the headline
C - baseline = +1.416 MCC confounds Moebius message passing with FOUR other
factors that the tuner also moved:

    factor          baseline      C
    tau_quantile    0.10          0.38     (edge density, ~3.6x denser)
    num_layers      4             2
    gat_hidden_dim  256           128
    proj_dim        128           256

This script fills the two missing cells of the 2x2 so the effect decomposes:

                        Euclidean            hyperbolic
    base-config     baseline (have, 45.368)  C_at_base   (RUN)
    C-config        base_at_C     (RUN)      C (have, 46.784)

Row 'C_at_base' is the one-factor test: the baseline's exact graph and
architecture with ONLY hyperbolic_gnn flipped on. Its hyperbolic knobs
(curvature, clip, scale, conv type) are taken from C's tuned values, which is
the choice most favourable to C -- so a null here is strong evidence.

Writes to its own CSV; the released campaign files are not touched.

SEED COUNT. At n=15 the interaction is not resolvable: the component contrasts
carry 2-3x the variance of the total because each differences two independently
trained cells, and detecting the observed interaction at 80% power needs about
65 seeds. Pass --seeds to extend.

WHY ALL FOUR CELLS ARE RUN HERE. The n=15 version took the two diagonal cells
(baseline, C) from results/campaign_wide_cola.csv rather than running them. That
is fine at n=15, where those are literally the same runs the paper reports, but
it does not extend: the campaign only has seeds 1-15, and splicing cells from a
different campaign context into one factorial is the cross-context comparison
this paper criticises elsewhere. Running all four cells through one script, one
code path and one cache state makes the 2x2 internally consistent. Cost is 2x,
and `res.has(key)` still skips anything already finished.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from campaign import run_one                      # noqa: E402
from exp_runner import ResultsCSV, PAPER_METRIC   # noqa: E402

TARGET, MODEL, TASK, SETTING = "glue", "bert-base-uncased", "cola", "cola_wide"
DEFAULT_OUT = os.path.join(HERE, "results", "factorial_geom_cola_parity.csv")

# exact confirmed configs, transcribed from results/campaign_wide_cola.csv
#
# CONTROL PARITY (see test_edge_parity.py). Two asymmetries made the earlier
# n=65 factorial a contrast in more than curvature, and both are closed here:
#
#   * self-loops -- the hyperbolic layers added without removing first, so each
#     token carried twice the self-weight its Euclidean control did. Fixed in
#     hyperbolic_layers.py; the test re-verifies 105 vs 105 on the real graph.
#   * edge attributes -- the Euclidean GAT is GATConv(edge_dim=1) fed a constant
#     1.0, which is NOT inert because GATConv adds it before a leaky_relu.
#     `gat_edge_attr: "0"` drops that path, taking the Euclidean layer from
#     98,944 to 98,688 parameters against the hyperbolic layer's 98,689.
#
# Both cells of every contrast therefore differ in curvature alone. This changes
# the Euclidean arm relative to the wide campaign, so all four cells must be run
# fresh into this file; nothing here may be compared against an older campaign.
SHARED = {"graph_metric": "cosine", "jk_mode": "max", "lr": "0.001",
          "scorer_hidden": "128", "weight_decay": "5e-05",
          "gat_edge_attr": "0"}
BASE_CFG = {**SHARED, "gat_hidden_dim": "256", "num_layers": "4",
            "proj_dim": "128", "tau_quantile": "0.1"}
C_CFG = {**SHARED, "gat_hidden_dim": "128", "num_layers": "2",
         "proj_dim": "256", "tau_quantile": "0.38"}
# the knobs that only exist when hyperbolic_gnn=1, at C's tuned values
HYP = {"curvature": "4.0", "gnn_input_clip": "0.7", "gnn_input_scale": "1",
       "hyp_gnn_type": "gat", "hyperbolic_gnn": "1"}

ARMS = {
    # off-diagonal: the two cells that isolate one factor each
    "C_at_base":    ({**BASE_CFG, **HYP}, 900),  # base graph+arch, hyperbolic MP
    "base_at_C":    (dict(C_CFG), 901),          # C graph+arch, Euclidean MP
    # diagonal: the two cells the n=15 version borrowed from the wide campaign
    "base_at_base": (dict(BASE_CFG), 902),       # baseline cell
    "C_at_C":       ({**C_CFG, **HYP}, 903),     # Stage C cell
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 16)),
                   help="Confirmation seeds. Use 1..65 for the powered run.")
    p.add_argument("--arms", nargs="+", default=list(ARMS),
                   choices=list(ARMS),
                   help="Cells to run. Default: all four.")
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds
    res = ResultsCSV(args.out)
    # SEED-MAJOR order: all four cells at seed 1, then all four at seed 2, ...
    # Arm-major would finish one whole cell before starting the next, so a job
    # killed partway leaves complete ARMS and no complete pairs, and any drift
    # over the run's lifetime (thermal, driver, cache eviction) would land on
    # the cells unequally -- confounded with the very contrast being measured.
    # Seed-major keeps every 2x2 block contemporaneous and leaves the partial
    # result analysable at whatever seed count was reached.
    todo = [(a, ARMS[a][0], ARMS[a][1], s) for s in seeds for a in args.arms]
    print(f"cells={args.arms} seeds={len(seeds)} runs={len(todo)} "
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
        print(f"    -> {r['score']:.2f}  density={r['mean_density']} "
              f"[{r['elapsed_sec']}s]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
