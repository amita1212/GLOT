#!/usr/bin/env python
"""Tier 0 / Tier 0.5 probe: do the Stage C conv layers get DOUBLE self-loops?

THE HYPOTHESIS
--------------
The graph builder already emits one self-loop per node (cosine(i,i)=1 > tau
always; d(i,i)=0 < rho always; `_knn_edges` fills the diagonal). Then:

  * the EUCLIDEAN baseline uses PyG's `GATConv`, which -- when
    `add_self_loops=True`, its default -- calls `remove_self_loops` FIRST and
    then `add_self_loops`. Net: exactly ONE self-loop per node.

  * `HyperbolicGCNConv` / `HyperbolicGATConv` call `add_self_loops` WITHOUT
    removing first. PyG's `add_self_loops` appends rather than de-duplicating.
    Net: TWO self-loops per node.

If true, every Stage C arm gave each token roughly double the self-weight its
Euclidean control had -- a difference in smoothing that has nothing to do with
curvature, and that sits inside the paper's headline +1.42 MCC.

WHAT THIS SCRIPT DOES -- no training, forward passes only, CPU is fine.

  Tier 0    Counts self-loops actually seen by each layer at runtime. The
            Euclidean count is read from `GATConv(..., return_attention_weights
            =True)`, i.e. measured, not simulated. The hyperbolic count is read
            by wrapping the `add_self_loops` symbol the layers actually call.

  Tier 0.5  Magnitude. Runs the full GLOT pooler on real cached CoLA features
            with Stage C's own tuned settings, once as-is and once with a
            de-duplicating `add_self_loops`, and reports the relative change in
            the pooled sentence vector.

            A bare number means nothing, so two REFERENCE perturbations are
            measured on the same pooler: (a) swapping the graph from dense to
            sparse, which is the largest structural change the method admits,
            and (b) nothing at all (same call twice), which gives the numerical
            noise floor. The self-loop delta is only worth a re-run if it sits
            closer to (a) than to (b).

Exit status is always 0; this is a measurement, not a test.
"""

from __future__ import annotations

import glob
import os
import sys

import torch

# --- locate the package -----------------------------------------------------
# The VM nests the clone at ~/glot/hyperglot/ while a fresh checkout has
# main.py at the top level. Accept either rather than hard-coding one.
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
from torch_geometric.utils import remove_self_loops  # noqa: E402

import hyperbolic_layers as HL  # noqa: E402
from hyperbolic_graph import (  # noqa: E402
    HyperbolicGraphConfig,
    build_pyg_graphs_hyper,
)

CACHE_CANDIDATES = [
    "data/bert-base-uncased_cola_val_batches",
    os.path.join(os.path.dirname(PKG), "data", "bert-base-uncased_cola_val_batches"),
]

# Stage C's own tuned configuration, from Appendix "Selected hyper-parameters"
# and the 2x2 factorial: c=4.0, HGAT, input clip 0.7, q=0.38, K=2, h=128.
C_CURVATURE = 4.0
C_QUANTILE = 0.38
BASE_QUANTILE = 0.10


def load_features(n_graphs: int = 8):
    for c in CACHE_CANDIDATES:
        files = sorted(glob.glob(os.path.join(c, "batch_*.pt")))
        if files:
            blob = torch.load(files[0], map_location="cpu")
            return blob["a_hs"][:n_graphs], blob["a_ms"][:n_graphs]
    raise SystemExit("no CoLA val cache found; tried %s" % CACHE_CANDIDATES)


def selfloops_per_node(edge_index: torch.Tensor, n_nodes: int) -> float:
    if edge_index.numel() == 0:
        return 0.0
    return float((edge_index[0] == edge_index[1]).sum()) / max(n_nodes, 1)


# ---------------------------------------------------------------------------
# Tier 0
# ---------------------------------------------------------------------------
def tier0(hs, ms) -> None:
    print("=" * 78)
    print("TIER 0  self-loops per node, as actually seen inside each conv layer")
    print("=" * 78)

    cfg = HyperbolicGraphConfig(
        graph_metric="cosine", adjacency="threshold",
        tau_quantile=C_QUANTILE, curvature=C_CURVATURE,
    )
    batch = build_pyg_graphs_hyper(hs, ms, cfg, device=torch.device("cpu"))
    x, ei = batch.x, batch.edge_index
    n = x.size(0)
    ea = batch.edge_attr

    print("  graph as built  : %d nodes, %d edges, %.3f self-loops/node"
          % (n, ei.size(1), selfloops_per_node(ei, n)))
    print()

    # -- Euclidean baseline: measured, via the edge_index GATConv returns ----
    torch.manual_seed(0)
    gat = GATConv(x.size(-1), 16, edge_dim=1)
    with torch.no_grad():
        _, (ei_gat, _) = gat(x, ei, edge_attr=ea, return_attention_weights=True)
    n_eu = selfloops_per_node(ei_gat, n)
    print("  EUCLIDEAN GATConv (baseline)      -> %.3f self-loops/node  (%d edges)"
          % (n_eu, ei_gat.size(1)))

    # -- Hyperbolic: wrap the symbol the layers call ------------------------
    seen = {}
    real_add = HL.add_self_loops

    def recording_add(edge_index, edge_attr=None, **kw):
        out = real_add(edge_index, edge_attr, **kw)
        seen["edge_index"] = out[0]
        return out

    ball = geoopt.PoincareBall(c=C_CURVATURE)
    x_ball = ball.projx(ball.expmap0(x * 0.01))

    HL.add_self_loops = recording_add
    try:
        for name, layer in (
            ("HYPERBOLIC HyperbolicGATConv (Stage C)", HL.HyperbolicGATConv(x.size(-1), 16, ball)),
            ("HYPERBOLIC HyperbolicGCNConv (Stage C)", HL.HyperbolicGCNConv(x.size(-1), 16, ball)),
        ):
            seen.clear()
            torch.manual_seed(0)
            with torch.no_grad():
                layer(x_ball, ei, ea)
            got = seen.get("edge_index")
            if got is None:
                print("  %-33s -> layer never called add_self_loops" % name)
                continue
            n_hyp = selfloops_per_node(got, n)
            print("  %-33s -> %.3f self-loops/node  (%d edges)"
                  % (name, n_hyp, got.size(1)))
    finally:
        HL.add_self_loops = real_add

    print()
    print("  VERDICT: the two paths %s."
          % ("DISAGREE -- bug confirmed" if abs(n_eu - n_hyp) > 1e-6
             else "agree -- no duplication"))


# ---------------------------------------------------------------------------
# Tier 0.5
# ---------------------------------------------------------------------------
def _dedup_add(real_add):
    """`add_self_loops` that removes existing self-loops first, as PyG's GAT does."""
    def patched(edge_index, edge_attr=None, **kw):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return real_add(edge_index, edge_attr, **kw)
    return patched


def tier05(hs, ms) -> None:
    print()
    print("=" * 78)
    print("TIER 0.5  how much does the duplicate self-loop move the output?")
    print("=" * 78)

    import main as glot_main

    def build(q):
        torch.manual_seed(1234)
        return glot_main.GLOT(
            in_dim=hs.size(-1), hidden_dim=128, num_layers=2, jk_mode="max",
            conv="gat", adjacency="threshold", device=torch.device("cpu"),
            graph_metric="cosine", tau_quantile=q,
            hyperbolic_gnn=True, hyp_gnn_type="gat",
            curvature=C_CURVATURE, gnn_input_clip=0.7,
        )

    def pooled(pooler):
        with torch.no_grad():
            return pooler(hs, ms)

    def rel(a, b):
        return (a - b).abs().max().item() / (a.abs().max().item() + 1e-12)

    real_add = HL.add_self_loops
    for label, q in (("C config   (q=%.2f)" % C_QUANTILE, C_QUANTILE),
                     ("base config (q=%.2f)" % BASE_QUANTILE, BASE_QUANTILE)):
        pooler = build(q)
        pooler.eval()

        z_buggy = pooled(pooler)
        z_buggy2 = pooled(pooler)              # reference (b): noise floor
        HL.add_self_loops = _dedup_add(real_add)
        try:
            z_fixed = pooled(pooler)
        finally:
            HL.add_self_loops = real_add

        # reference (a): the largest structural change the method admits
        pooler.tau_quantile = 0.95
        z_dense = pooled(pooler)
        pooler.tau_quantile = 0.02
        z_sparse = pooled(pooler)
        pooler.tau_quantile = q

        r_bug = rel(z_buggy, z_fixed)
        r_noise = rel(z_buggy, z_buggy2)
        r_graph = rel(z_dense, z_sparse)

        print("  %s" % label)
        print("    self-loop dedup            rel delta = %.3e   <-- the bug" % r_bug)
        print("    reference (a) dense->sparse graph      = %.3e" % r_graph)
        print("    reference (b) same call twice (noise)  = %.3e" % r_noise)
        if r_graph > 0:
            print("    bug / graph-change ratio               = %.3f" % (r_bug / r_graph))
        print()

    print("  READ IT LIKE THIS: if the bug's delta is near the noise floor it is")
    print("  cosmetic and the Stage C numbers stand. If it is a sizeable fraction")
    print("  of the dense->sparse reference, it is a real perturbation and the two")
    print("  hyperbolic cells of the factorial must be re-run before the 65-seed run.")


if __name__ == "__main__":
    hs, ms = load_features()
    print("features: %s, mask %s\n" % (tuple(hs.shape), tuple(ms.shape)))
    tier0(hs, ms)
    tier05(hs, ms)
