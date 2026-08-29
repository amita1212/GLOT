#!/usr/bin/env python
"""
E2 -- Geometry search for HyperGLOT Stage A.

What changed since the negative result
--------------------------------------
Three defects made the previous Stage A search incapable of showing any effect:

  1. curvature was fixed at geoopt's default c = 1.0. Measured on real cached
     CoLA features, that saturates expmap0 and the Poincare graph becomes the
     cosine graph (edge Jaccard 0.9991). The arm WAS the baseline.
  2. raw BERT token norms have a coefficient of variation of 0.057, so the
     ball's "depth" axis -- the only thing hyperbolic geometry adds over cosine
     -- was essentially constant. Per-sentence centring raises it to 0.154.
  3. the Stage A builder dropped self-loops while upstream's `_threshold_edges`
     keeps one per node, so every hyperbolic arm was compared against a baseline
     with a different edge convention.

Parametrisation
---------------
Graph construction depends on the features only through r_eff = sqrt(c)*mean||x||
(the Poincare distance's 2/sqrt(c) prefactor cannot change rankings or
quantiles). `feature_mode='center_unit'` fixes mean||x|| = 1, so

        r_eff = sqrt(c)      exactly, independent of backbone / layer / task.

That makes curvature a scale-free hyper-parameter, and the grid below is stated
directly in r_eff. r_eff <~ 0.15 is the Euclidean limit, r_eff >~ 4 is the
cosine limit, and everything interesting is in between.

Arms
----
  cosine        GLOT baseline (upstream builder, untouched)
  A_center_unit Stage A, re-rooted + scale-free, sweeping r_eff
  A_unit        Stage A, scale-free but NOT re-rooted -> isolates centring
  A_raw         Stage A exactly as before (uncentred), sweeping curvature
  W_hyp         GLOT's cosine edge SET, with hyperbolic edge weights.
                Strictly additive: topology the paper validated is untouched.
  W_soft        control for W_hyp -- same soft weighting but derived from
                cosine, so any W_hyp gain cannot be credited to "soft weights
                are better than binary ones".
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_runner import sweep  # noqa: E402

# r_eff = sqrt(c) once mean||x|| == 1, so c = r_eff**2.
R_EFF_GRID = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
RAW_C_GRID = [1e-3, 1e-2, 3e-2, 1.0]      # uncentred: mean||x|| ~ 14.7
TAU_GRID = [0.4, 0.6, 0.8]
Q_MATCHED = 0.10                           # density-matched to cosine tau = 0.6


def stage1(task, seed):
    cfgs = []

    for tau in TAU_GRID:
        cfgs.append((f"{task}|s{seed}|cosine|tau{tau}",
                     dict(task=task, seed=seed, graph_metric="cosine", tau=tau),
                     dict(arm="cosine", detail=f"tau={tau}")))

    for mode in ("center_unit", "unit"):
        for r in R_EFF_GRID:
            c = r * r
            cfgs.append((f"{task}|s{seed}|A_{mode}|r{r}",
                         dict(task=task, seed=seed, graph_metric="poincare",
                              rho_quantile=Q_MATCHED, graph_curvature=c,
                              feature_mode=mode),
                         dict(arm=f"A_{mode}", detail=f"r_eff={r}")))

    for c in RAW_C_GRID:
        cfgs.append((f"{task}|s{seed}|A_raw|c{c:g}",
                     dict(task=task, seed=seed, graph_metric="poincare",
                          rho_quantile=Q_MATCHED, graph_curvature=c,
                          feature_mode="none"),
                     dict(arm="A_raw", detail=f"c={c:g}")))

    # Additive arms: keep GLOT's cosine topology, enrich the edge attribute.
    for r in (0.5, 1.0, 2.0):
        for t in (0.25, 1.0):
            cfgs.append((f"{task}|s{seed}|W_hyp|r{r}|t{t}",
                         dict(task=task, seed=seed, graph_metric="cosine", tau=0.6,
                              edge_weight_mode="hyp", graph_curvature=r * r,
                              feature_mode="center_unit", edge_temp=t),
                         dict(arm="W_hyp", detail=f"r_eff={r},T={t}")))
    for t in (0.25, 1.0):
        cfgs.append((f"{task}|s{seed}|W_soft|t{t}",
                     dict(task=task, seed=seed, graph_metric="cosine", tau=0.6,
                          edge_weight_mode="soft", edge_temp=t),
                     dict(arm="W_soft", detail=f"T={t}")))
    return cfgs


def stage_density(task, seed, mode, r):
    """Sweep the sparsity knob for the winning geometry, density-matched to tau."""
    cfgs = []
    for q in (0.025, 0.038, 0.10, 0.20, 0.38):
        cfgs.append((f"{task}|s{seed}|A_{mode}|r{r}|q{q}",
                     dict(task=task, seed=seed, graph_metric="poincare",
                          rho_quantile=q, graph_curvature=r * r, feature_mode=mode),
                     dict(arm=f"A_{mode}_density", detail=f"r_eff={r},q={q}")))
    return cfgs


def stage_weights(task, seed):
    """E3 -- the additive direction: keep GLOT's cosine topology, change only
    what the GAT is told about each edge.

    Stage 1 showed that *replacing* the topology with a hyperbolic one costs
    2-3 MCC at every curvature and every re-rooting, while *annotating* the
    existing topology with hyperbolic distance was the only arm at or above the
    baseline. This stage tunes that direction properly, with cosine-derived
    weightings as controls so a gain cannot be credited to "graded beats binary".
    """
    cfgs = []
    for tau in TAU_GRID:
        cfgs.append((f"{task}|s{seed}|cosine|tau{tau}",
                     dict(task=task, seed=seed, graph_metric="cosine", tau=tau),
                     dict(arm="cosine", detail=f"tau={tau}")))

    for mode in ("center_unit", "cls_root_unit"):
        for r in (1.0, 2.0, 3.0):
            cfgs.append((f"{task}|s{seed}|W_hypz|{mode}|r{r}",
                         dict(task=task, seed=seed, graph_metric="cosine", tau=0.6,
                              edge_weight_mode="hyp_z", graph_curvature=r * r,
                              feature_mode=mode),
                         dict(arm="W_hypz", detail=f"{mode},r={r}")))

    # Control: same standardised weighting but derived from COSINE, so any
    # W_hypz gain must come from the geometry and not from the conditioning.
    cfgs.append((f"{task}|s{seed}|W_softz",
                 dict(task=task, seed=seed, graph_metric="cosine", tau=0.6,
                      edge_weight_mode="soft_z"),
                 dict(arm="W_softz", detail="tau=0.6")))

    # Best sigmoid variant from stage 1, re-tuned over tau.
    for tau in (0.4, 0.8):
        cfgs.append((f"{task}|s{seed}|W_hyp|tau{tau}",
                     dict(task=task, seed=seed, graph_metric="cosine", tau=tau,
                          edge_weight_mode="hyp", graph_curvature=4.0,
                          feature_mode="center_unit", edge_temp=1.0),
                     dict(arm="W_hyp", detail=f"tau={tau},r=2.0,T=1.0")))
    return cfgs


def stage_confirm(prev_csv, task, seeds, top_k, always_keep=("cosine",)):
    """Re-run the best config of each arm across several seeds.

    Selection happens on the stage-1 seed only, and confirmation runs on the
    remaining seeds, so the reported mean is not the same number that was
    maximised over. Same-seed GPU nondeterminism alone is 0.2-0.5 MCC and the
    seed-to-seed std on CoLA is 0.81, so nothing below ~1.5 sigma is reportable
    and single-seed maxima are systematically optimistic.
    """
    import csv as _csv
    from collections import defaultdict

    rows = [r for r in _csv.DictReader(open(prev_csv, newline="", encoding="utf-8"))
            if r["task"] == task]
    if not rows:
        raise SystemExit(f"no stage-1 rows for task {task} in {prev_csv}")

    best_per_arm = {}
    for r in rows:
        arm = r.get("arm", "?")
        cur = best_per_arm.get(arm)
        if cur is None or float(r["score"]) > float(cur["score"]):
            best_per_arm[arm] = r

    ranked = sorted(best_per_arm.values(), key=lambda r: -float(r["score"]))
    chosen, seen = [], set()
    for r in ranked:
        if len(chosen) < top_k or r.get("arm") in always_keep:
            if r.get("arm") not in seen:
                chosen.append(r)
                seen.add(r.get("arm"))
    for a in always_keep:
        if a not in seen and a in best_per_arm:
            chosen.append(best_per_arm[a])

    print("promoting to multi-seed confirmation:")
    cfgs = []
    for r in chosen:
        print(f"  {r.get('arm'):<18} {r.get('detail'):<20} stage1={r['score']}")
        for seed in seeds:
            base = dict(
                task=task, seed=seed,
                graph_metric=r["graph_metric"],
                graph_adj=r.get("graph_adj", "threshold"),
                tau=float(r["tau"]),
                rho_quantile=float(r.get("rho_quantile", -1.0)),
                graph_curvature=float(r.get("graph_curvature", -1.0)),
                feature_mode=r.get("feature_mode", "none"),
                edge_weight_mode=r.get("edge_weight_mode", "binary"),
                edge_temp=float(r.get("edge_temp", 0.25)),
            )
            cfgs.append((f"{task}|confirm|{r.get('arm')}|{r.get('detail')}|s{seed}",
                         base, dict(arm=r.get("arm"), detail=r.get("detail"))))
    return cfgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="1",
                   choices=["1", "density", "weights", "confirm"])
    p.add_argument("--tasks", nargs="+", default=["cola"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--mode", default="center_unit")
    p.add_argument("--r_eff", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--from_csv", default="")
    p.add_argument("--out", default=os.path.join(HERE, "results", "e2_geometry.csv"))
    p.add_argument("--cwd", default=HERE)
    args = p.parse_args()

    cfgs = []
    for task in args.tasks:
        if args.stage == "confirm":
            cfgs += stage_confirm(args.from_csv or args.out, task,
                                  args.seeds, args.top_k)
            continue
        for seed in args.seeds:
            if args.stage == "1":
                cfgs += stage1(task, seed)
            elif args.stage == "weights":
                cfgs += stage_weights(task, seed)
            else:
                cfgs += stage_density(task, seed, args.mode, args.r_eff)
    sweep(cfgs, args.out, cwd=args.cwd)


if __name__ == "__main__":
    main()
