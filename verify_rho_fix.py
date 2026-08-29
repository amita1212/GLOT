#!/usr/bin/env python
"""Verify the rho_quantile fix: the Poincare threshold rule must now produce
edge densities that MATCH the cosine rule, instead of empty graphs."""

from __future__ import annotations

import os
import sys
import glob
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperglot"))

from hyperbolic_graph import (  # noqa: E402
    pairwise_cosine_single,
    pairwise_poincare_single,
    _threshold_edges_from_sim,
    _threshold_edges_from_dist,
    _quantile_rho,
)
import geoopt  # noqa: E402

CACHE = "data/bert-base-uncased_cola_val_batches"
TAU_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
Q_GRID = [0.97, 0.69, 0.38, 0.10, 0.025]


def main() -> None:
    files = sorted(glob.glob(os.path.join(CACHE, "batch_*.pt")))
    blob = torch.load(files[0], map_location="cpu")
    hs, ms = blob["a_hs"], blob["a_ms"]
    ball = geoopt.PoincareBall(c=1.0)
    n_graphs = min(16, hs.size(0))

    def density(builder):
        tot, empty = 0.0, 0
        for b in range(n_graphs):
            x = hs[b][ms[b].bool()]
            n = x.size(0)
            if n < 2:
                continue
            e = builder(x).size(1)
            tot += e / (n * (n - 1))
            empty += int(e == 0)
        return tot / n_graphs, empty

    print(f"{'cosine tau':>12s} {'density':>9s} {'empty':>7s}   |"
          f" {'rho_quantile':>13s} {'density':>9s} {'empty':>7s}")
    print("-" * 78)
    for tau, q in zip(TAU_GRID, Q_GRID):
        dc, ec = density(
            lambda x, t=tau: _threshold_edges_from_sim(pairwise_cosine_single(x), t)[0])

        def poin(x, qq=q):
            D = pairwise_poincare_single(x, ball, False, 1e-5)
            return _threshold_edges_from_dist(D, _quantile_rho(D, qq))[0]

        dp, ep = density(poin)
        print(f"{tau:>12} {dc:>9.4f} {ec:>4d}/{n_graphs}   |"
              f" {q:>13} {dp:>9.4f} {ep:>4d}/{n_graphs}")

    print("\nPASS if the two density columns are comparable and no graph is empty.")


if __name__ == "__main__":
    main()
