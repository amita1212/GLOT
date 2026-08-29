#!/usr/bin/env python
"""Diagnose the Stage-A graph: is the Poincare threshold rule actually building
any edges on REAL BERT token features?

Motivation
----------
In the fair ablation, every `*_thresh` arm with graph_metric=poincare returned a
bit-identical score for rho in {0.5, 1.0, 1.5, 2.0, 3.0}. A hyperparameter that
changes nothing is the signature of a degenerate graph.

Suspected mechanism: BERT last-layer token norms are ~10-25. At curvature c=1,
    expmap0(x) = tanh(||x||) * x/||x||
and tanh(15) = 1 - 2e-13, so EVERY token is mapped onto the ball boundary. The
Poincare distance between two near-boundary points diverges, so `D < rho` is
false for every pair at any sane rho -> the graph is empty, and Stage A silently
degenerates to "no graph at all".

This script measures, on real cached hidden states:
  * the distribution of raw token norms,
  * the radius they map to after expmap0,
  * the edge density produced by cosine@tau, poincare@rho, and both kNN rules.
"""

from __future__ import annotations

import os
import sys
import glob
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperglot"))

from hyperbolic_graph import (  # noqa: E402
    HyperbolicGraphConfig,
    pairwise_cosine_single,
    pairwise_poincare_single,
    _threshold_edges_from_sim,
    _threshold_edges_from_dist,
    _knn_edges,
)
import geoopt  # noqa: E402

CACHE = "data/bert-base-uncased_cola_val_batches"


def main() -> None:
    files = sorted(glob.glob(os.path.join(CACHE, "batch_*.pt")))
    if not files:
        print(f"No cached batches at {CACHE}")
        return
    blob = torch.load(files[0], map_location="cpu")
    hs, ms = blob["a_hs"], blob["a_ms"]
    print(f"loaded {files[0]}  hidden={tuple(hs.shape)} mask={tuple(ms.shape)}")

    # ---- token norm statistics -------------------------------------------
    norms = []
    for b in range(hs.size(0)):
        v = hs[b][ms[b].bool()]
        if v.numel():
            norms.append(v.norm(dim=-1))
    norms = torch.cat(norms)
    print("\n=== raw token L2 norms ===")
    for q in (0.0, 0.05, 0.5, 0.95, 1.0):
        print(f"  q{q:<5} {norms.quantile(q).item():8.3f}")
    print(f"  mean  {norms.mean().item():8.3f}")

    for c in (0.25, 1.0, 2.0):
        r = torch.tanh(torch.sqrt(torch.tensor(c)) * norms) / torch.sqrt(torch.tensor(c))
        gap = 1.0 / torch.sqrt(torch.tensor(c)) - r
        print(f"\n  curvature c={c}: expmap0 radius mean={r.mean().item():.10f} "
              f"boundary gap min={gap.min().item():.3e} max={gap.max().item():.3e}")

    # ---- edge density -----------------------------------------------------
    n_graphs = min(16, hs.size(0))
    print(f"\n=== edge density over {n_graphs} real sentences ===")
    print(f"{'rule':34s} {'mean edges':>11s} {'density':>9s} {'empty graphs':>13s}")
    print("-" * 72)

    def summarize(label, builder):
        tot_e, tot_d, empty = 0.0, 0.0, 0
        for b in range(n_graphs):
            x = hs[b][ms[b].bool()]
            n = x.size(0)
            if n < 2:
                continue
            ei, _ = builder(x)
            e = ei.size(1)
            tot_e += e
            tot_d += e / (n * (n - 1))
            empty += int(e == 0)
        print(f"{label:34s} {tot_e / n_graphs:11.1f} {tot_d / n_graphs:9.4f} "
              f"{empty:>8d}/{n_graphs}")

    for tau in (0.0, 0.2, 0.4, 0.6, 0.8):
        summarize(f"cosine  tau={tau}",
                  lambda x, t=tau: _threshold_edges_from_sim(pairwise_cosine_single(x), t))

    for c in (1.0,):
        ball = geoopt.PoincareBall(c=c)
        for rho in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0):
            summarize(
                f"poincare c={c} rho={rho}",
                lambda x, b=ball, r=rho: _threshold_edges_from_dist(
                    pairwise_poincare_single(x, b, False, 1e-5), r),
            )

    ball = geoopt.PoincareBall(c=1.0)
    for k in (4, 8, 16):
        summarize(f"poincare knn k={k}",
                  lambda x, b=ball, kk=k: _knn_edges(
                      pairwise_poincare_single(x, b, False, 1e-5), kk, False))

    # ---- what do the actual distances look like? --------------------------
    x = hs[0][ms[0].bool()]
    D = pairwise_poincare_single(x, ball, False, 1e-5)
    off = D[~torch.eye(D.size(0), dtype=torch.bool)]
    print("\n=== Poincare pairwise distances, sentence 0 (c=1, no feature_norm) ===")
    print(f"  min={off.min().item():.4f}  median={off.median().item():.4f}  "
          f"max={off.max().item():.4f}")
    print(f"  fraction below rho=3.0 : {(off < 3.0).float().mean().item():.4f}")
    print(f"  fraction below rho=20.0: {(off < 20.0).float().mean().item():.4f}")

    Dn = pairwise_poincare_single(x, ball, True, 1e-5)   # feature_norm=True
    offn = Dn[~torch.eye(Dn.size(0), dtype=torch.bool)]
    print("\n=== same, WITH feature_norm=True ===")
    print(f"  min={offn.min().item():.4f}  median={offn.median().item():.4f}  "
          f"max={offn.max().item():.4f}")
    print(f"  fraction below rho=1.0 : {(offn < 1.0).float().mean().item():.4f}")


if __name__ == "__main__":
    main()
