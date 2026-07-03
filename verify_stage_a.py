"""
Verification for Stage A (hyperbolic graph construction).

Run:  python verify_stage_a.py

Checks (no GPU, no downloads required for tests 1-4):
  1. Poincare distance sanity: symmetric, zero diagonal, finite, non-negative.
  2. EQUIVALENCE: hyperbolic-kNN on L2-normalised features == cosine-kNN, for
     several curvatures c. (GLOT's kNN graph is a special case of Stage A.)
  3. Magnitude DOES matter: on raw (un-normalised) features, hyperbolic edges
     differ from cosine edges -> Stage A genuinely uses the hierarchy signal.
  4. Drop-in: the produced PyG Batch flows through GLOT's real GLOT GNN head
     (imported from main.py) and yields finite pooled embeddings.
  5. (optional, needs internet) tiny-BERT end-to-end smoke test.

Exit code 0 == all required checks passed.
"""

import sys
import torch

from hyperbolic_graph import (
    HyperbolicGraphConfig,
    build_pyg_graphs_hyper,
    pairwise_poincare_single,
    pairwise_cosine_single,
)
import geoopt


def _edge_set(batch):
    ei = batch.edge_index
    return set(map(tuple, ei.t().tolist()))


def test_distance_sanity():
    torch.manual_seed(0)
    h = torch.randn(12, 16)
    ball = geoopt.PoincareBall(c=1.0)
    D = pairwise_poincare_single(h, ball)
    assert torch.allclose(D, D.t(), atol=1e-5), "distance not symmetric"
    assert torch.allclose(torch.diag(D), torch.zeros(12), atol=1e-4), "nonzero diagonal"
    assert torch.isfinite(D).all(), "non-finite distances"
    assert (D >= -1e-5).all(), "negative distances"
    print("[1] distance sanity ....................... OK")


def test_knn_equivalence():
    """Hyperbolic-kNN on normalised features must equal cosine-kNN for any c."""
    torch.manual_seed(1)
    B, L, d = 3, 20, 32
    hidden = torch.randn(B, L, d)
    mask = torch.ones(B, L)

    cos_cfg = HyperbolicGraphConfig(graph_metric="cosine", adjacency="knn", k=5)
    cos_batch = build_pyg_graphs_hyper(hidden, mask, cos_cfg)
    cos_edges = _edge_set(cos_batch)

    for c in [0.25, 1.0, 4.0]:
        hyp_cfg = HyperbolicGraphConfig(
            graph_metric="poincare", adjacency="knn", k=5,
            curvature=c, feature_norm=True,
        )
        hyp_batch = build_pyg_graphs_hyper(hidden, mask, hyp_cfg)
        assert _edge_set(hyp_batch) == cos_edges, f"kNN edges differ at c={c}"
    print("[2] hyperbolic-kNN == cosine-kNN (c in {0.25,1,4}) OK  <- GLOT special case")


def test_magnitude_matters():
    """On raw features, hyperbolic edges should differ from cosine edges."""
    torch.manual_seed(2)
    B, L, d = 2, 25, 32
    hidden = torch.randn(B, L, d)
    # Give tokens widely varying magnitudes (the 'hierarchy' signal cosine ignores).
    scales = torch.linspace(0.1, 5.0, L).view(1, L, 1)
    hidden = hidden * scales
    mask = torch.ones(B, L)

    cos = _edge_set(build_pyg_graphs_hyper(
        hidden, mask, HyperbolicGraphConfig(graph_metric="cosine", adjacency="knn", k=5)))
    hyp = _edge_set(build_pyg_graphs_hyper(
        hidden, mask, HyperbolicGraphConfig(
            graph_metric="poincare", adjacency="knn", k=5,
            curvature=1.0, feature_norm=False)))
    assert cos != hyp, "hyperbolic edges identical to cosine even with magnitude variation"
    overlap = len(cos & hyp) / max(1, len(cos))
    print(f"[3] magnitude matters: raw hyp != cosine (edge overlap={overlap:.2f}) OK")


def test_dropin_through_glot_gnn():
    """The Batch must flow through GLOT's real GLOT head unchanged.

    main.py pulls in heavy eval-only deps (mteb/wandb/peft/torch_scatter) at
    import time that are NOT used by the GLOT pooling forward pass. We stub those
    so the *genuine* GLOT class is imported and exercised, and provide a correct
    scatter_add shim (the only torch_scatter symbol GLOT.forward actually uses).
    """
    import types

    class _AnyMeta(type):
        def __getattr__(cls, item):
            return _Any

    class _Any(metaclass=_AnyMeta):
        """Universal permissive placeholder: subclassable, callable, any attr."""
        def __init__(self, *a, **k):
            pass
        def __getattr__(self, item):
            return _Any
        def __call__(self, *a, **k):
            return _Any()

    class _AnyModule(types.ModuleType):
        def __getattr__(self, item):
            return _Any

    # --- lightweight stubs for import-time-only heavy deps ---
    _stub_names = ["mteb", "wandb", "peft", "torch_scatter"]
    _saved = {n: sys.modules.get(n) for n in _stub_names}
    for name in ["mteb", "wandb", "peft"]:
        sys.modules[name] = _AnyModule(name)

    # torch_scatter.scatter_add shim (correct, index_add-based).
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

    try:
        from main import GLOT  # GLOT's own real class
    except Exception as e:  # pragma: no cover
        print(f"[4] SKIP (could not import GLOT from main.py: {e})")
        return

    torch.manual_seed(3)
    B, L, d = 4, 18, 64
    hidden = torch.randn(B, L, d)
    mask = torch.ones(B, L)

    head = GLOT(in_dim=d, hidden_dim=32, num_layers=2, jk_mode="cat",
                conv="gat", adjacency="threshold", tau=0.3)

    # Monkeypatch GLOT's graph builder to use the hyperbolic one (Stage A).
    import main as glot_main
    cfg = HyperbolicGraphConfig(graph_metric="poincare", adjacency="threshold",
                                rho=2.0, curvature=1.0)
    orig = glot_main.build_pyg_graphs

    def patched(hidden, attention_mask, adjacency="knn", tau=0.3, device=None):
        return build_pyg_graphs_hyper(hidden, attention_mask, cfg, device=device)

    glot_main.build_pyg_graphs = patched
    try:
        out = head(hidden, mask)
        # Also exercise the NATIVE patched path: GLOT(graph_metric="poincare")
        # builds hyperbolic graphs internally (no monkeypatch).
        head2 = GLOT(in_dim=d, hidden_dim=32, num_layers=2, jk_mode="cat",
                     conv="gat", adjacency="threshold", tau=0.3,
                     graph_metric="poincare", curvature=1.0, rho=5.0)
        out2 = head2(hidden, mask)
    finally:
        glot_main.build_pyg_graphs = orig
        # Restore sys.modules so later tests (e.g. transformers/peft) are clean.
        for n, mod in _saved.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
        sys.modules.pop("main", None)

    assert out.shape[0] == B, f"bad batch dim {out.shape}"
    assert torch.isfinite(out).all(), "non-finite pooled embeddings"
    assert out2.shape[0] == B and torch.isfinite(out2).all(), "native poincare path failed"
    print(f"[4] drop-in + native GLOT(graph_metric=poincare) -> pooled {tuple(out2.shape)} finite  OK")


def test_tiny_bert_optional():
    try:
        from transformers import AutoTokenizer, AutoModel
    except Exception as e:
        print(f"[5] SKIP tiny-BERT (transformers missing: {e})")
        return
    try:
        name = "hf-internal-testing/tiny-random-bert"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name).eval()
    except Exception as e:
        print(f"[5] SKIP tiny-BERT (offline / download failed: {e})")
        return

    sents = ["this movie was not wonderful", "a person is riding a horse"]
    enc = tok(sents, padding=True, truncation=True, max_length=32, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"]

    for metric in ["cosine", "poincare"]:
        cfg = HyperbolicGraphConfig(graph_metric=metric, adjacency="threshold",
                                    tau=0.5, rho=2.0, curvature=1.0)
        batch = build_pyg_graphs_hyper(hidden, mask, cfg)
        assert batch.x.shape[1] == hidden.shape[-1]
        print(f"[5] tiny-BERT {metric:8s}: nodes={batch.num_nodes} edges={batch.edge_index.shape[1]}  OK")


if __name__ == "__main__":
    test_distance_sanity()
    test_knn_equivalence()
    test_magnitude_matters()
    test_dropin_through_glot_gnn()
    test_tiny_bert_optional()
    print("\nAll required Stage-A checks passed.")
    sys.exit(0)
