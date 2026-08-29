#!/usr/bin/env python
"""
Verification suite for the Stage A graph-conditioning additions.

Standing rule for this project: a knob that produces bit-identical results
across its whole grid is a DISCONNECTED knob, and we have already lost one
generation of results to exactly that (absolute `rho` -> empty graph, and
`curvature=1.0` -> the cosine graph). So every new option is checked here for
(a) not breaking the original GLOT path and (b) actually doing something.

Checks
------
 1. REGRESSION: `cosine + threshold + binary` still routes to upstream's
    `build_pyg_graphs` and yields a bit-identical edge set.
 2. SELF-CONSISTENCY: `feature_mode='l2'` must reproduce the cosine graph
    exactly (Jaccard 1.0) -- this is the analytic claim in the module docstring.
 3. CONNECTEDNESS: `feature_mode` and `curvature` must each change the edge set.
 4. SOFT WEIGHTS: `edge_weight_mode='soft'` keeps the edge SET identical to
    'binary' while producing graded weights strictly inside (0, 1).
 5. NON-EMPTY: no configuration may yield an empty graph.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "hyperglot"))

from hyperbolic_graph import (  # noqa: E402
    HyperbolicGraphConfig, build_pyg_graphs_hyper, preprocess_tokens,
)
import main as glot_main  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def edge_set(batch):
    ei = batch.edge_index
    return {(int(a), int(b)) for a, b in zip(ei[0].tolist(), ei[1].tolist())}


def make_batch(hidden, mask, **kw):
    cfg = HyperbolicGraphConfig(**kw)
    return build_pyg_graphs_hyper(hidden, mask, cfg, device=hidden.device)


def real_features(cache_dir, n_sents=8):
    """Load real cached BERT token features -- synthetic Gaussians do not have
    BERT's anisotropy or its norm distribution, and the failure modes we are
    guarding against are precisely artefacts of those statistics."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from diagnose_curvature import load_sentences
    sents = load_sentences(cache_dir, n_sents, min_tokens=12)
    L = max(x.size(0) for x in sents)
    d = sents[0].size(1)
    hidden = torch.zeros(len(sents), L, d)
    mask = torch.zeros(len(sents), L, dtype=torch.long)
    for i, x in enumerate(sents):
        hidden[i, :x.size(0)] = x
        mask[i, :x.size(0)] = 1
    return hidden, mask


def main():
    torch.manual_seed(0)
    cache = os.environ.get(
        "GLOT_CACHE", "/home/t-amitalfasi/glot/data/bert-base-uncased_cola_train_batches")
    hidden, mask = real_features(cache)
    print(f"Using REAL cached features: {tuple(hidden.shape)}, "
          f"mean||x||={hidden[mask.bool()].norm(dim=-1).mean():.3f}\n")

    print("1. REGRESSION -- original GLOT cosine path is untouched")
    up = glot_main.build_pyg_graphs(hidden, mask, adjacency="threshold", tau=0.6)
    ours = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold", tau=0.6)
    check("cosine+threshold edge set identical to upstream",
          edge_set(up) == edge_set(ours),
          f"|E| upstream={up.edge_index.size(1)} ours={ours.edge_index.size(1)}")
    check("cosine+threshold edge weights all 1.0",
          bool(torch.allclose(ours.edge_attr, torch.ones_like(ours.edge_attr))))
    n_nodes = int(mask.sum())
    n_self = int((up.edge_index[0] == up.edge_index[1]).sum())
    check("upstream keeps one self-loop per node (convention we must match)",
          n_self == n_nodes, f"{n_self} self-loops / {n_nodes} nodes")

    print("\n2. SELF-CONSISTENCY -- feature_mode='l2' must equal the cosine graph")
    # kNN is the setting where the equality is exact for any curvature -- but
    # only while float32 can still resolve the distance ranking. Past c ~ 10 all
    # unit-norm points are clamped to the same radius and artanh saturates, so
    # the ranking degenerates into ties. That is a numerical limit worth knowing,
    # not a logic error, so it is reported rather than asserted.
    cos_knn = make_batch(hidden, mask, graph_metric="cosine", adjacency="knn", k=6)
    for c in (1e-3, 1.0):
        hyp_knn = make_batch(hidden, mask, graph_metric="poincare", adjacency="knn",
                             k=6, curvature=c, feature_mode="l2")
        check(f"poincare-kNN(l2, c={c:g}) == cosine-kNN",
              edge_set(cos_knn) == edge_set(hyp_knn))
    e10 = edge_set(make_batch(hidden, mask, graph_metric="poincare", adjacency="knn",
                              k=6, curvature=10.0, feature_mode="l2"))
    j10 = len(e10 & edge_set(cos_knn)) / max(len(e10 | edge_set(cos_knn)), 1)
    print(f"  [note] c=10 float32 tie-breaking: Jaccard vs cosine-kNN = {j10:.3f} "
          f"(expected <1: artanh saturates)")

    print("\n3. CONNECTEDNESS -- new knobs must change the graph")
    ref = edge_set(make_batch(hidden, mask, graph_metric="poincare",
                              adjacency="threshold", rho_quantile=0.10,
                              curvature=1.0, feature_mode="none"))
    jac = lambda a, b: len(a & b) / max(len(a | b), 1)

    for c in (1e-3, 1e-2, 3e-2):
        e = edge_set(make_batch(hidden, mask, graph_metric="poincare",
                                adjacency="threshold", rho_quantile=0.10,
                                curvature=c, feature_mode="none"))
        check(f"curvature c={c:g} changes the edge set", e != ref,
              f"Jaccard vs c=1.0: {jac(e, ref):.3f}")

    for mode in ("center", "center_unit", "unit"):
        e = edge_set(make_batch(hidden, mask, graph_metric="poincare",
                                adjacency="threshold", rho_quantile=0.10,
                                curvature=0.01, feature_mode=mode))
        base = edge_set(make_batch(hidden, mask, graph_metric="poincare",
                                   adjacency="threshold", rho_quantile=0.10,
                                   curvature=0.01, feature_mode="none"))
        check(f"feature_mode='{mode}' changes the edge set", e != base,
              f"Jaccard vs 'none': {jac(e, base):.3f}")

    print("\n4. SOFT WEIGHTS -- same edges, graded attributes")
    for metric, kw in (("cosine", dict(graph_metric="cosine", tau=0.6)),
                       ("poincare", dict(graph_metric="poincare", rho_quantile=0.10,
                                         curvature=0.01, feature_mode="center_unit"))):
        b_bin = make_batch(hidden, mask, adjacency="threshold",
                           edge_weight_mode="binary", **kw)
        b_soft = make_batch(hidden, mask, adjacency="threshold",
                            edge_weight_mode="soft", edge_temp=0.25, **kw)
        w = b_soft.edge_attr.view(-1)
        off = w[b_soft.edge_index[0] != b_soft.edge_index[1]]
        check(f"{metric}: soft keeps the edge SET",
              edge_set(b_bin) == edge_set(b_soft))
        check(f"{metric}: soft weights inside (0,1]",
              bool((w > 0).all() and (w <= 1.0).all()),
              f"min={w.min():.4f} max={w.max():.4f}")
        check(f"{metric}: soft weights are not constant",
              float(off.std()) > 1e-4, f"std(non-self)={float(off.std()):.4f}")

    print("\n4b. HYP WEIGHTS -- GLOT's cosine edge set, hyperbolic edge_attr")
    b_bin = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold",
                       tau=0.6, edge_weight_mode="binary")
    b_hyp = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold",
                       tau=0.6, edge_weight_mode="hyp", curvature=0.01,
                       feature_mode="center_unit", edge_temp=0.5)
    wh = b_hyp.edge_attr.view(-1)
    offh = wh[b_hyp.edge_index[0] != b_hyp.edge_index[1]]
    check("hyp: topology identical to the cosine baseline",
          edge_set(b_bin) == edge_set(b_hyp))
    check("hyp: weights are graded, not constant", float(offh.std()) > 1e-3,
          f"std={float(offh.std()):.4f} min={float(offh.min()):.4f}")
    # Must not simply reproduce the cosine ordering, else it adds nothing.
    b_soft = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold",
                        tau=0.6, edge_weight_mode="soft", edge_temp=0.5)
    ws = b_soft.edge_attr.view(-1)[b_soft.edge_index[0] != b_soft.edge_index[1]]
    corr = float(torch.corrcoef(torch.stack([offh, ws]))[0, 1])
    check("hyp: carries information beyond the cosine weight", abs(corr) < 0.99,
          f"corr(hyp, cosine-soft) = {corr:.3f}")

    print("\n4c. Z-SCORED WEIGHTS -- better conditioned than the sigmoid")
    b_z = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold",
                     tau=0.6, edge_weight_mode="hyp_z", curvature=0.01,
                     feature_mode="center_unit")
    wz = b_z.edge_attr.view(-1)
    offz = wz[b_z.edge_index[0] != b_z.edge_index[1]]
    check("hyp_z: topology identical to the cosine baseline",
          edge_set(b_bin) == edge_set(b_z))
    check("hyp_z: standardised (mean~0, std~1 over real edges)",
          abs(float(offz.mean())) < 0.15 and abs(float(offz.std()) - 1.0) < 0.25,
          f"mean={float(offz.mean()):+.3f} std={float(offz.std()):.3f}")
    check("hyp_z: self-loops pinned to 0",
          float(wz[b_z.edge_index[0] == b_z.edge_index[1]].abs().max()) < 1e-6)
    check("hyp_z: spread is wider than the sigmoid version",
          float(offz.std()) > float(offh.std()),
          f"z std={float(offz.std()):.3f} vs sigmoid std={float(offh.std()):.3f}")
    b_sz = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold",
                      tau=0.6, edge_weight_mode="soft_z")
    check("soft_z: topology identical to the cosine baseline",
          edge_set(b_bin) == edge_set(b_sz))
    b_pz = make_batch(hidden, mask, graph_metric="poincare", adjacency="threshold",
                      rho_quantile=0.10, curvature=0.01, feature_mode="center_unit",
                      edge_weight_mode="hyp_z")
    check("poincare + hyp_z produces graded weights",
          float(b_pz.edge_attr.view(-1).std()) > 0.1,
          f"std={float(b_pz.edge_attr.view(-1).std()):.3f}")

    print("\n4d. DEPTH -- the only ASYMMETRIC edge feature")
    b_d = make_batch(hidden, mask, graph_metric="cosine", adjacency="threshold",
                     tau=0.6, edge_weight_mode="depth", feature_mode="center_unit")
    wd = b_d.edge_attr.view(-1)
    ei = b_d.edge_index
    check("depth: topology identical to the cosine baseline",
          edge_set(b_bin) == edge_set(b_d))
    check("depth: self-loops are exactly 0",
          float(wd[ei[0] == ei[1]].abs().max()) < 1e-6)
    # Antisymmetry: for every edge (a,b) present with weight w, the reverse edge
    # (b,a) must carry -w. This is what makes it a hierarchy DIRECTION.
    wmap = {(int(a), int(b)): float(w)
            for a, b, w in zip(ei[0].tolist(), ei[1].tolist(), wd.tolist())}
    pairs = [(k, v) for k, v in wmap.items() if k[0] != k[1] and (k[1], k[0]) in wmap]
    anti = all(abs(v + wmap[(k[1], k[0])]) < 1e-4 for k, v in pairs)
    check("depth: antisymmetric (w_ij = -w_ji)", anti and len(pairs) > 0,
          f"checked {len(pairs)} reciprocal pairs")
    check("depth: not degenerate", float(wd.std()) > 0.1,
          f"std={float(wd.std()):.3f}")

    print("\n5. NON-EMPTY -- no configuration may produce an empty graph")
    for mode in ("none", "unit", "center", "center_unit", "l2"):
        for c in (1e-3, 1e-2, 1.0):
            b = make_batch(hidden, mask, graph_metric="poincare",
                           adjacency="threshold", rho_quantile=0.10,
                           curvature=c, feature_mode=mode)
            check(f"non-empty: mode={mode} c={c:g}", b.edge_index.size(1) > 0,
                  f"|E|={b.edge_index.size(1)}")

    print("\n6. PREPROCESS SANITY")
    x = hidden[0][mask[0].bool()]
    check("center: mean is ~0",
          float(preprocess_tokens(x, "center").mean(0).norm()) < 1e-3)
    check("center_unit: mean norm is ~1",
          abs(float(preprocess_tokens(x, "center_unit").norm(dim=-1).mean()) - 1.0) < 1e-4)
    check("unit: mean norm is ~1",
          abs(float(preprocess_tokens(x, "unit").norm(dim=-1).mean()) - 1.0) < 1e-4)
    check("none: identity", bool(torch.equal(preprocess_tokens(x, "none"), x)))

    print("\n" + "=" * 60)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED:")
        for f in FAILED:
            print("  - " + f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
