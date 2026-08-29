#!/usr/bin/env python
"""Bug #6 probe: does Stage C (hyperbolic GNN) actually respond to the graph?

Symptom that triggered this: on STS-B, the `C_thresh` arm returned 83.26 for
ALL FIVE tau values -- but that arm uses the COSINE graph, whose edge density
provably varies with tau (0.97 -> 0.025). On CoLA the same arm did vary. So
either the layer ignores edge_index, or something upstream collapses the
features so the graph cannot matter.

Two levels of test:
  L1  the conv layers in isolation: same features, different edge_index.
  L2  the whole GLOT pooler on REAL cached features: same weights, graphs built
      at tau=0.0 (dense) vs tau=0.8 (sparse) -- this is the path training uses.

L2 is the one that matters; L1 localises the fault if L2 fails.
"""

from __future__ import annotations

import os
import sys
import glob
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "hyperglot"))

import geoopt  # noqa: E402
from hyperbolic_layers import HyperbolicGCNConv, HyperbolicGATConv  # noqa: E402

CACHE = "data/bert-base-uncased_cola_val_batches"


def l1_layer_test() -> None:
    print("=" * 78)
    print("L1: conv layers in isolation (same x, different edge_index)")
    print("=" * 78)
    torch.manual_seed(0)
    ball = geoopt.PoincareBall(c=1.0)
    n, d = 24, 32
    x = ball.projx(ball.expmap0(torch.randn(n, d) * 0.1))

    # sparse chain vs dense complete graph
    chain = torch.stack([torch.arange(n - 1), torch.arange(1, n)])
    chain = torch.cat([chain, chain.flip(0)], dim=1)
    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    m = ii != jj
    dense = torch.stack([ii[m], jj[m]])

    for name, layer in (("HyperbolicGCNConv", HyperbolicGCNConv(d, d, ball)),
                        ("HyperbolicGATConv", HyperbolicGATConv(d, d, ball))):
        torch.manual_seed(1)
        with torch.no_grad():
            a = layer(x, chain)
            b = layer(x, dense)
        delta = (a - b).abs().max().item()
        rel = delta / (a.abs().max().item() + 1e-12)
        verdict = "EDGE-SENSITIVE" if rel > 1e-4 else "*** IGNORES GRAPH ***"
        print(f"  {name:20s} max|chain-dense| = {delta:.3e}  rel = {rel:.3e}   {verdict}")


def l2_pooler_test() -> None:
    print()
    print("=" * 78)
    print("L2: full GLOT pooler on real cached features, tau=0.0 vs tau=0.8")
    print("=" * 78)
    files = sorted(glob.glob(os.path.join(CACHE, "batch_*.pt")))
    if not files:
        print(f"  no cache at {CACHE}; skipping")
        return
    blob = torch.load(files[0], map_location="cpu")
    hs, ms = blob["a_hs"][:8], blob["a_ms"][:8]

    import main as glot_main  # our modified main.py

    def build(**kw):
        torch.manual_seed(1234)
        return glot_main.GLOT(
            in_dim=hs.size(-1), hidden_dim=64, num_layers=2, jk_mode="cat",
            conv="gat", adjacency="threshold",
            device=torch.device("cpu"), **kw
        )

    configs = [
        ("euclidean (Stage 0)", dict(graph_metric="cosine", hyperbolic_gnn=False)),
        ("Stage C, no clip", dict(graph_metric="cosine", hyperbolic_gnn=True,
                                  hyp_gnn_type="gat")),
        ("Stage C, clip=0.7 (as used in ablation)",
         dict(graph_metric="cosine", hyperbolic_gnn=True, hyp_gnn_type="gat",
              gnn_input_clip=0.7, gnn_input_scale=True)),
    ]

    for label, kw in configs:
        try:
            pooler = build(**kw)
            pooler.eval()
            with torch.no_grad():
                pooler.tau = 0.0
                z_dense = pooler(hs, ms)
                pooler.tau = 0.8
                z_sparse = pooler(hs, ms)
            delta = (z_dense - z_sparse).abs().max().item()
            scale = z_dense.abs().max().item() + 1e-12
            rel = delta / scale
            verdict = "EDGE-SENSITIVE" if rel > 1e-4 else "*** IGNORES GRAPH ***"
            print(f"  {label:42s} rel delta = {rel:.3e}   {verdict}")
        except Exception as e:  # noqa: BLE001
            print(f"  {label:42s} ERROR {type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    l1_layer_test()
    l2_pooler_test()
