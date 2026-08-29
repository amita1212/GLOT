#!/usr/bin/env python
"""
E1 -- Curvature sweep for HyperGLOT Stage A (hyperbolic token graph).

Why this experiment exists
--------------------------
`diagnose_curvature.py` on real cached BERT features shows that the curvature
used by EVERY Stage A run so far (geoopt's default c = 1.0) puts all tokens on
the ball boundary, where the Poincare distance is a monotone function of the
angle alone. Measured edge-set Jaccard against GLOT's cosine graph at matched
density: **0.9991**. In other words the "hyperbolic" arm was, numerically, the
cosine baseline -- the knob was never connected to anything.

The genuinely hyperbolic window is c in roughly [3e-3, 3e-2] (effective radius
sqrt(c)*mean||x|| in [0.8, 2.5]), where the Spearman correlation between the
Poincare distance and the token norms rises from 0.30 to 0.71, i.e. the token
*norm* -- the hierarchy signal cosine discards -- finally participates.

Design
------
Sparsity is held fixed by `rho_quantile` (a quantile of the observed pairwise
distances), so moving `curvature` changes ONLY the geometry, never the density.
That isolates the effect we care about. The cosine baseline is run at the
density-matched `tau` values so the comparison is like-for-like.

Two degenerate anchors are included on purpose as internal controls:
    c = 1e-5  -> expmap0 is linear   => distance == Euclidean
    c = 1.0   -> boundary saturation => distance == cosine (the old default)
If the sweep is wired correctly, c = 1.0 must land on the cosine baseline's
score and the middle of the grid must differ from both anchors.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_runner import sweep  # noqa: E402

# Density-matched pairs, measured in diagnose_graph_density.py:
#   cosine tau 0.4 -> density 0.38 | 0.6 -> 0.099 | 0.8 -> 0.025
DENSITY_PAIRS = [(0.4, 0.38), (0.6, 0.10), (0.8, 0.025)]

# Anchors (1e-5 == Euclidean, 1.0 == cosine) bracket the real hyperbolic window.
CURVATURES = [1e-5, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["cola"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--curvatures", nargs="+", type=float, default=CURVATURES)
    p.add_argument("--out", default=os.path.join(HERE, "results", "e1_curvature.csv"))
    p.add_argument("--cwd", default=HERE)
    args = p.parse_args()

    configs = []
    for task in args.tasks:
        for seed in args.seeds:
            # --- cosine control at each density ---
            for tau, _q in DENSITY_PAIRS:
                configs.append((
                    f"{task}|s{seed}|cosine|tau{tau}",
                    dict(task=task, seed=seed, graph_metric="cosine",
                         graph_adj="threshold", tau=tau),
                    dict(arm="cosine_baseline", curvature_swept="", density_tag=tau),
                ))
            # --- Stage A at every curvature, density held fixed ---
            for c in args.curvatures:
                for tau, q in DENSITY_PAIRS:
                    configs.append((
                        f"{task}|s{seed}|poincare|c{c:g}|q{q}",
                        dict(task=task, seed=seed, graph_metric="poincare",
                             graph_adj="threshold", rho_quantile=q, curvature=c),
                        dict(arm="A_thresh", curvature_swept=c, density_tag=tau),
                    ))

    sweep(configs, args.out, cwd=args.cwd)


if __name__ == "__main__":
    main()
