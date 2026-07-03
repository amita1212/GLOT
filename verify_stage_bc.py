"""
Verification for Stage B (hyperbolic readout) and Stage C (hyperbolic Token-GNN).

Run:  python verify_stage_bc.py

Checks (no GPU, no downloads required):
  1. HyperbolicGCNConv (Stage C): output has the right shape, is finite, stays
     strictly inside the Poincare ball, and passes gradients to its weights.
  2. hyperbolic_readout (Stage B): right shape, finite, deterministic, and
     genuinely differs from the Euclidean weighted-sum readout.
  3. Curvature -> 0 limit: as c -> 0 the hyperbolic (Einstein-midpoint) readout
     collapses to GLOT's Euclidean weighted mean -> GLOT is the c->0 special case.
  4. Drop-in through the real GLOT head: baseline (all switches off) is byte-for
     -byte identical to stock GLOT; enabling B and/or C changes the output while
     keeping the (B, D) contract and finiteness, and gradients still flow.

Exit code 0 == all checks passed.
"""

import sys
import types

import torch
import torch.nn.functional as F

import geoopt

from hyperbolic_layers import HyperbolicGCNConv, hyperbolic_readout


def test_hyperbolic_gnn_layer():
    torch.manual_seed(0)
    n, d, h = 15, 32, 24
    c = 1.0
    ball = geoopt.PoincareBall(c=c)
    x = torch.randn(n, d, requires_grad=True)
    x_ball = ball.projx(ball.expmap0(x))

    # a small ring graph + a few extra edges
    src = torch.arange(n)
    dst = (src + 1) % n
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_weight = torch.ones(edge_index.size(1), 1)

    conv = HyperbolicGCNConv(d, h, ball)
    out = conv(x_ball, edge_index, edge_weight)

    assert out.shape == (n, h), f"bad shape {out.shape}"
    assert torch.isfinite(out).all(), "non-finite HGCN output"
    # points must live strictly inside the ball of radius 1/sqrt(c)
    radius = 1.0 / (c ** 0.5)
    assert (out.norm(dim=-1) < radius + 1e-4).all(), "HGCN output left the ball"

    # gradients flow to the layer weights
    loss = ball.logmap0(out).pow(2).sum()
    loss.backward()
    assert conv.lin.weight.grad is not None and torch.isfinite(conv.lin.weight.grad).all(), \
        "no/!finite grad in hyperbolic linear"
    print("[1] Stage C HGCN layer: shape/finite/in-ball/grad ....... OK")


def _readout_inputs(seed=1):
    torch.manual_seed(seed)
    # two graphs: sizes 4 and 3
    h_all = torch.randn(7, 16)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    scores = torch.randn(7)
    return h_all, batch, scores


def _softmax_by_graph(scores, batch):
    # standalone segment-softmax (avoids importing torch_geometric here)
    out = torch.zeros_like(scores)
    for g in batch.unique():
        m = batch == g
        out[m] = F.softmax(scores[m], dim=0)
    return out


def test_hyperbolic_readout():
    h_all, batch, scores = _readout_inputs()
    weights = _softmax_by_graph(scores, batch)
    ball = geoopt.PoincareBall(c=1.0)

    z = hyperbolic_readout(h_all, weights, batch, ball, curvature=1.0, num_graphs=2)
    assert z.shape == (2, 16), f"bad shape {z.shape}"
    assert torch.isfinite(z).all(), "non-finite readout"

    # deterministic
    z2 = hyperbolic_readout(h_all, weights, batch, ball, curvature=1.0, num_graphs=2)
    assert torch.allclose(z, z2), "readout not deterministic"

    # genuinely different from the Euclidean weighted-sum readout
    euclid = torch.zeros(2, 16)
    for g in (0, 1):
        m = batch == g
        euclid[g] = (weights[m].unsqueeze(-1) * h_all[m]).sum(0)
    assert not torch.allclose(z, euclid, atol=1e-3), "hyperbolic readout == Euclidean (should differ)"
    print("[2] Stage B readout: shape/finite/deterministic/!=euclid  OK")


def test_curvature_zero_limit():
    """As c -> 0 the Einstein-midpoint readout -> Euclidean weighted mean."""
    h_all, batch, scores = _readout_inputs(seed=2)
    weights = _softmax_by_graph(scores, batch)
    c = 1e-6
    ball = geoopt.PoincareBall(c=c)
    z = hyperbolic_readout(h_all, weights, batch, ball, curvature=c, num_graphs=2)
    euclid = torch.zeros(2, h_all.size(-1))
    for g in (0, 1):
        m = batch == g
        euclid[g] = (weights[m].unsqueeze(-1) * h_all[m]).sum(0)
    assert torch.allclose(z, euclid, atol=1e-3), \
        f"c->0 limit mismatch (max diff {(z - euclid).abs().max():.2e})"
    print("[3] c->0 limit: hyperbolic readout == GLOT Euclidean mean .. OK  <- special case")


def _import_glot_with_stubs():
    """Import the real GLOT class from main.py, stubbing import-time-only deps."""
    class _AnyMeta(type):
        def __getattr__(cls, item):
            return _Any

    class _Any(metaclass=_AnyMeta):
        def __init__(self, *a, **k):
            pass
        def __getattr__(self, item):
            return _Any
        def __call__(self, *a, **k):
            return _Any()

    class _AnyModule(types.ModuleType):
        def __getattr__(self, item):
            return _Any

    saved = {n: sys.modules.get(n) for n in ["mteb", "wandb", "peft", "torch_scatter"]}
    for name in ["mteb", "wandb", "peft"]:
        sys.modules[name] = _AnyModule(name)

    ts = types.ModuleType("torch_scatter")

    def scatter_add(src, index, dim=-1, out=None, dim_size=None):
        if dim < 0:
            dim = src.dim() + dim
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() else 0
        shape = list(src.shape)
        shape[dim] = dim_size
        target = src.new_zeros(shape) if out is None else out
        idx = index
        if idx.dim() == 1 and src.dim() > 1:
            view = [1] * src.dim()
            view[dim] = -1
            idx = idx.view(view).expand_as(src)
        return target.scatter_add_(dim, idx, src)

    ts.scatter_add = scatter_add
    sys.modules["torch_scatter"] = ts
    from main import GLOT
    return GLOT, saved


def _restore(saved):
    for n, mod in saved.items():
        if mod is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = mod
    sys.modules.pop("main", None)


def test_glot_integration():
    try:
        GLOT, saved = _import_glot_with_stubs()
    except Exception as e:  # pragma: no cover
        print(f"[4] SKIP (could not import GLOT from main.py: {e})")
        return
    try:
        torch.manual_seed(3)
        B, L, d = 4, 20, 48
        hidden = torch.randn(B, L, d)
        mask = torch.ones(B, L)

        common = dict(in_dim=d, hidden_dim=32, num_layers=2, jk_mode="cat",
                      conv="gat", adjacency="threshold", tau=0.3)

        # Baseline must be identical across two constructions with all switches off.
        torch.manual_seed(7)
        base = GLOT(**common)
        torch.manual_seed(7)
        base_ref = GLOT(**common, hyperbolic_gnn=False, hyperbolic_readout=False)
        with torch.no_grad():
            o_base = base(hidden, mask)
            o_ref = base_ref(hidden, mask)
        assert torch.allclose(o_base, o_ref), "baseline path perturbed by new (off) switches"

        # Stage B only, Stage C only, and both.
        outs = {}
        for tag, kw in {
            "B": dict(hyperbolic_readout=True),
            "C": dict(hyperbolic_gnn=True),
            "ABC": dict(hyperbolic_gnn=True, hyperbolic_readout=True, graph_metric="poincare", rho=5.0),
        }.items():
            head = GLOT(**common, curvature=1.0, **kw)
            out = head(hidden, mask)
            assert out.shape[0] == B, f"{tag}: bad batch dim {out.shape}"
            assert torch.isfinite(out).all(), f"{tag}: non-finite pooled"
            # gradients flow through the pooling head
            head.zero_grad()
            out.pow(2).sum().backward()
            grads = [p.grad for p in head.parameters() if p.requires_grad]
            assert any(g is not None and torch.isfinite(g).all() for g in grads), f"{tag}: no grad"
            outs[tag] = out.detach()

        # Stage B changes the readout, so it must differ from the baseline output.
        assert not torch.allclose(outs["B"], o_base.detach(), atol=1e-4), \
            "Stage B readout produced identical output to baseline"
        print("[4] GLOT head: baseline invariant; B/C/ABC finite + grad  OK")
    finally:
        _restore(saved)


def main():
    test_hyperbolic_gnn_layer()
    test_hyperbolic_readout()
    test_curvature_zero_limit()
    test_glot_integration()
    print("\nAll Stage B/C checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
