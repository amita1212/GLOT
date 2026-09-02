#!/usr/bin/env python
"""Stage C control parity: is the Euclidean/hyperbolic contrast curvature-only?

TWO ASYMMETRIES, MEASURED ON THE SAME FOOTING
---------------------------------------------
1. SELF-LOOPS. The graph builder already emits one self-loop per node. PyG's
   GATConv removes then adds, netting one. The hyperbolic layers used to add
   without removing, netting two. Fixed in hyperbolic_layers.py; re-verified
   here so the fix cannot silently regress.

2. EDGE ATTRIBUTES. The Euclidean GAT is built as GATConv(edge_dim=1) and fed
   edge_attr; HyperbolicGATConv accepts edge_weight and ignores it. At
   edge_weight_mode="binary" every attribute is exactly 1.0, which looks inert
   -- but GATConv computes leaky_relu(a_src + a_dst + a_edge), and leaky_relu is
   not translation-equivariant, so a constant does NOT cancel in the softmax.

Both must be closed before a factorial can attribute anything to curvature.
Reference perturbations are measured on the same pooler so the magnitudes mean
something: a dense-to-sparse graph swap is the largest structural change the
method admits, and the same call twice is the numerical noise floor.

Forward passes only. No training. CPU is fine. Exit status is always 0.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (HERE, os.path.join(HERE, "hyperglot"), os.path.dirname(HERE)):
    if os.path.exists(os.path.join(_cand, "hyperbolic_layers.py")):
        sys.path.insert(0, _cand)
        PKG = _cand
        break
else:  # pragma: no cover
    raise SystemExit("could not find hyperbolic_layers.py near %s" % HERE)

import geoopt  # noqa: E402
from torch_geometric.nn import GATConv  # noqa: E402
from torch_geometric.utils import add_self_loops, remove_self_loops  # noqa: E402

from hyperbolic_layers import HyperbolicGATConv  # noqa: E402

torch.manual_seed(0)

# Stage C's tuned cell: c=4.0, HGAT, clip 0.7, q=0.38, K=2, h=128.
C_CURVATURE, HID = 4.0, 128
N, IN = 105, 768          # the real BERT/CoLA graph size used in app:defects


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def build_graph(density: float) -> torch.Tensor:
    sim = torch.randn(N, N)
    sim = (sim + sim.t()) / 2
    thr = torch.quantile(sim.flatten(), 1.0 - density)
    A = (sim > thr).float()
    A.fill_diagonal_(1.0)                     # builder always emits self-loops
    return A.nonzero().t().contiguous()


x = torch.randn(N, IN)
ei_sparse = build_graph(0.38)                 # Stage C's tuned q
ei_dense = build_graph(0.99)
E = ei_sparse.size(1)
print(f"graph: {N} nodes, {E} edges at q=0.38 "
      f"(self-loops emitted by builder: {int(ei_sparse[0].eq(ei_sparse[1]).sum())})\n")

# ---------------------------------------------------------------- self-loops
print("=== 1. SELF-LOOP PARITY ===")
eu = GATConv(IN, HID, edge_dim=1).eval()
n_eu = add_self_loops(remove_self_loops(ei_sparse)[0], num_nodes=N)[0]
print(f"  Euclidean GATConv nets      {int(n_eu[0].eq(n_eu[1]).sum())} self-loops "
      f"({n_eu.size(1)} edges)")

seen = {}
_orig_add = __import__("hyperbolic_layers").add_self_loops


def _spy(edge_index, *a, **kw):
    out = _orig_add(edge_index, *a, **kw)
    ei = out[0] if isinstance(out, tuple) else out
    seen["loops"] = int(ei[0].eq(ei[1]).sum())
    seen["edges"] = int(ei.size(1))
    return out


import hyperbolic_layers as HL  # noqa: E402
HL.add_self_loops = _spy
ball = geoopt.PoincareBall(c=C_CURVATURE)
hyp = HyperbolicGATConv(IN, HID, ball).eval()
with torch.no_grad():
    hyp(ball.projx(ball.expmap0(x * 0.7 / x.norm(dim=-1, keepdim=True).clamp_min(1e-5))),
        ei_sparse, torch.ones(E))
HL.add_self_loops = _orig_add
print(f"  HyperbolicGATConv nets      {seen['loops']} self-loops "
      f"({seen['edges']} edges)")
ok_loops = seen["loops"] == int(n_eu[0].eq(n_eu[1]).sum())
print(f"  -> {'MATCHED' if ok_loops else 'ASYMMETRIC -- fix not applied'}\n")

# ----------------------------------------------------------- edge attributes
print("=== 2. EDGE-ATTRIBUTE PARITY (Euclidean GAT) ===")
with torch.no_grad():
    ones = torch.ones(E, 1)
    out_on = eu(x, ei_sparse, edge_attr=ones)          # gat_edge_attr=1, the paper
    out_off = eu(x, ei_sparse, edge_attr=None)         # gat_edge_attr=0, matched
    out_twice = eu(x, ei_sparse, edge_attr=ones)       # noise floor
    out_dense = eu(x, ei_dense, edge_attr=torch.ones(ei_dense.size(1), 1))

d_edge = rel(out_on, out_off)
d_noise = rel(out_on, out_twice)
d_dense = rel(out_on, out_dense)
print(f"  edge path ON vs OFF (the asymmetry)   {d_edge:.3e}")
print(f"  REFERENCE dense-vs-sparse graph swap  {d_dense:.3e}")
print(f"  REFERENCE same call twice (noise)     {d_noise:.3e}")
share = 100.0 * d_edge / d_dense if d_dense else float("nan")
print(f"  -> the asymmetry is {share:.1f}% of the largest structural change\n")

n_on = sum(p.numel() for p in GATConv(IN, HID, edge_dim=1).parameters())
n_off = sum(p.numel() for p in GATConv(IN, HID).parameters())
n_hyp = sum(p.numel() for p in HyperbolicGATConv(IN, HID, ball).parameters())
print("=== 3. PARAMETER PARITY ===")
print(f"  GATConv(edge_dim=1)  {n_on:>8,}   <- gat_edge_attr=1")
print(f"  GATConv()            {n_off:>8,}   <- gat_edge_attr=0")
print(f"  HyperbolicGATConv    {n_hyp:>8,}")
print(f"  edge path costs {n_on - n_off} parameters the hyperbolic arm never had.")
print("\nVERDICT: run the corrective factorial with gat_edge_attr=0 in all four "
      "cells." if ok_loops else "\nVERDICT: self-loop fix missing -- do not launch.")
