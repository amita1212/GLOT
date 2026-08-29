"""
Stage A of HyperGLOT: Hyperbolic token-graph construction.

This module is a drop-in replacement / extension for GLOT's
``build_pyg_graphs`` (see ``main.py``). It keeps the *entire* downstream GLOT
pipeline (Token-GNN + attention readout + classifier) unchanged and only swaps
*how token edges are formed*:

    GLOT (Euclidean):   edge(i, j) if  cosine_sim(x_i, x_j)            >  tau
    Stage A (hyperbolic): edge(i, j) if  d_Poincare(exp0(x_i), exp0(x_j)) <  rho

Two graph builders are provided, mirroring GLOT's options:
  * ``threshold`` : connect pairs closer than a distance threshold ``rho``.
  * ``knn``       : connect each node to its ``k`` nearest neighbours.

Key, *verifiable* property (used in ``verify_stage_a.py``):
    With L2-normalised token features, hyperbolic-kNN produces the **identical**
    edge set as cosine-kNN for **any** curvature ``c``. Intuitively, expmap0
    sends all unit-norm vectors to the same hyperbolic radius, and the Poincare
    distance is a strictly increasing function of Euclidean distance at a fixed
    radius, so neighbour *rankings* are preserved. Hence GLOT's kNN graph is a
    special case of Stage A -- the geometry only starts to differ once we use the
    *magnitude* of the token vectors (which is exactly the hierarchy signal that
    cosine throws away).

Dependencies: torch, torch_geometric, geoopt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

import geoopt


@dataclass
class HyperbolicGraphConfig:
    """Configuration for Stage A graph construction."""
    graph_metric: str = "cosine"   # {"cosine", "poincare"}
    adjacency: str = "threshold"    # {"threshold", "knn"}
    tau: float = 0.6                # cosine threshold (used when graph_metric == "cosine")
    tau_quantile: float = -1.0      # if in (0, 1): keep the most-similar fraction q of
                                    # pairs instead of using an absolute `tau`. This is
                                    # the cosine twin of `rho_quantile` and exists so the
                                    # BASELINE can be density-matched to the hyperbolic
                                    # arms. Without it the comparison is rigged: an
                                    # absolute tau grid cannot reach the low densities
                                    # `rho_quantile` reaches by construction, so a
                                    # hyperbolic arm can "win" purely on sparsity.
    rho: float = 1.0                # ABSOLUTE hyperbolic-distance threshold -- see warning below
    rho_quantile: float = -1.0      # if in (0, 1): use the q-quantile of the observed
                                    # pairwise distances as the threshold instead of `rho`
    k: int = 8                      # neighbours for knn adjacency
    curvature: float = 1.0          # Poincare ball curvature c (> 0)
    feature_norm: bool = False      # legacy alias for feature_mode == "l2"
    eps: float = 1e-5               # boundary safety clamp for the ball
    # --- geometry conditioning (see FEATURE CONDITIONING note below) ---------
    feature_mode: str = "none"      # {"none","l2","unit","center","center_unit"}
    graph_scale: float = 1.0        # extra multiplier applied after `feature_mode`
    # --- edge weighting ------------------------------------------------------
    # binary            1.0 everywhere (upstream GLOT)
    # soft / soft_z     graded weight from the SAME metric that formed the edges
    # hyp  / hyp_z      graded weight from the HYPERBOLIC distance, whatever
    #                   metric formed the edges -> lets GLOT keep its validated
    #                   cosine topology while still seeing hyperbolic structure
    # depth             SIGNED gap in hyperbolic depth along the directed edge;
    #                   the only one of these that is asymmetric
    # *_z variants standardise instead of squashing through a sigmoid
    edge_weight_mode: str = "binary"  # {binary, soft, soft_z, hyp, hyp_z, depth}
    # --- word-order structure (see PERMUTATION INVARIANCE note below) --------
    positional_window: int = 0      # if > 0, additionally connect every pair of tokens
                                    # whose ORIGINAL sequence positions differ by <= w.
                                    # 0 = off, reproducing upstream GLOT exactly.
    positional_only: bool = False   # if True, use ONLY the positional edges and drop the
                                    # similarity edges. This is the ablation that says
                                    # how much of any gain is word order rather than
                                    # semantics -- without it a positional arm that wins
                                    # cannot be attributed.
    edge_temp: float = 0.25           # sigmoid temperature, in units of the
                                      # observed spread (ignored by *_z modes)
    # --- self-loop convention ------------------------------------------------
    # Upstream GLOT's `_threshold_edges` never zeroes the diagonal, and
    # cosine(i, i) = 1 > tau always, so the ORIGINAL graph carries one self-loop
    # per node. The Stage A builder used to drop them, which meant every
    # hyperbolic arm was compared against a baseline with a different edge
    # convention -- a silent confound independent of geometry. Default True
    # reproduces upstream exactly.
    self_loops: bool = True


# --------------------------------------------------------------------------- #
# WHY `rho_quantile` EXISTS -- read before using absolute `rho`.
#
# BERT last-layer token norms are ~10-17 (measured: mean 14.73). At curvature
# c=1, expmap0(x) = tanh(||x||) * x/||x||, and tanh(14.7) rounds to exactly 1.0
# in float32 -- every token lands *on* the ball boundary (measured boundary gap:
# 0.0e+00). Pairwise Poincare distances then sit in a narrow, far-from-zero band:
#
#     measured on real CoLA features, c=1: min 8.85, median 11.15, max 11.76
#
# So the "natural looking" grid rho in {0.5, 1.0, 1.5, 2.0, 3.0} yields ZERO
# edges on 16/16 sentences, and rho=20 yields a COMPLETE graph. There is no
# usable signal outside roughly rho in [9, 13], and that band shifts with the
# backbone, the layer, and the curvature.
#
# Absolute `rho` is therefore a trap: it silently degenerates to "no graph" and
# every downstream metric becomes invariant to it. `rho_quantile` instead picks
# the threshold from the observed distance distribution, so it (a) always
# produces a usable graph, (b) is comparable across models and curvatures, and
# (c) can be density-matched against cosine's tau, which isolates the effect of
# *geometry* from the effect of *sparsity*.
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# FEATURE CONDITIONING -- why `feature_mode` and `curvature` decide whether
# Stage A is hyperbolic at all.
#
# expmap0 depends on the features only through sqrt(c)*||x||, and the Poincare
# distance carries a global 2/sqrt(c) factor that cannot change edge rankings or
# distance quantiles. So *for graph construction* curvature and a feature
# rescale are literally the same knob, and the only quantity that matters is the
# EFFECTIVE RADIUS  r_eff = sqrt(c) * mean||x||.  Measured on real CoLA features:
#
#   r_eff        regime                         Jaccard vs GLOT's cosine graph
#   <~ 0.15      expmap0 is linear  -> Euclidean            0.77
#   ~ 0.8 - 2.5  genuinely hyperbolic                  0.66 -> 0.37
#   >~ 4         projx clamps all tokens to the same
#                radius -> distance is a monotone
#                function of the ANGLE ONLY -> cosine        0.999
#
# Raw BERT tokens have mean||x|| = 14.68, so the default c = 1.0 gives
# r_eff = 14.68: every Stage A experiment ever run was sitting in the third
# regime, i.e. it *was* the cosine baseline (Jaccard 0.9991). The knob was never
# connected to anything.
#
# Second problem: hierarchy needs the norms to VARY. Raw BERT token norms have a
# coefficient of variation of only 0.057, so the "depth" coordinate is nearly
# constant and the ball cannot express hierarchy at any curvature. BERT's cloud
# is a cone displaced from the origin (||mean(x)|| = 6.88, mean cos(x, mu) =
# 0.47), and the ball's origin is supposed to be the ROOT of the hierarchy.
# Re-centring on the per-sentence token mean moves the root into the cloud and
# raises the coefficient of variation to 0.154 -- 2.7x more depth signal --
# which shows up as Spearman(distance, endpoint norms) rising from 0.30 to 0.93.
#
#   feature_mode  transform                        cv(||x||)   sp(D, norms)
#   none          x                                  0.057        +0.30
#   l2            x / ||x||                          0.000        -0.04   (== cosine)
#   unit          x / mean||x||                      0.057        +0.30
#   center        x - mu                             0.154        +0.80
#   center_unit   (x - mu) / mean||x - mu||          0.154        +0.78
#
# `center_unit` additionally fixes mean||x|| = 1, so r_eff = sqrt(c) exactly and
# the curvature becomes a scale-free hyper-parameter that transfers across
# backbones, layers and tasks instead of silently depending on the norm of
# whatever encoder produced the features.
# --------------------------------------------------------------------------- #


def preprocess_tokens(h: torch.Tensor, mode: str = "none", scale: float = 1.0,
                      eps: float = 1e-6) -> torch.Tensor:
    """Condition token features before they are lifted into the Poincare ball.

    See the FEATURE CONDITIONING note above. ``mode='none'`` with ``scale=1.0``
    is the identity, so the original Stage A behaviour is preserved exactly.

    The ``cls_root`` variants deserve a word. In a Poincare ball the origin is
    the ROOT of the hierarchy and ``||x||`` is depth. For a sentence graph the
    natural root is not the coordinate origin and not even the token centroid --
    it is the sentence representation itself. Rooting at ``[CLS]`` therefore
    makes ``||x_i||`` read as "how far token i is from the sentence gist", i.e.
    a specificity score, which is exactly the quantity cosine similarity is
    invariant to.
    """
    if mode == "l2":
        z = F.normalize(h, dim=-1)
    elif mode == "unit":
        z = h / h.norm(dim=-1).mean().clamp_min(eps)
    elif mode == "center":
        z = h - h.mean(0, keepdim=True)
    elif mode == "center_unit":
        z = h - h.mean(0, keepdim=True)
        z = z / z.norm(dim=-1).mean().clamp_min(eps)
    elif mode == "cls_root":
        z = h - h[:1]
    elif mode == "cls_root_unit":
        z = h - h[:1]
        z = z / z.norm(dim=-1).mean().clamp_min(eps)
    elif mode == "none":
        z = h
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")
    return z * scale if scale != 1.0 else z


# --------------------------------------------------------------------------- #
# Pairwise similarity / distance on a single (padded-stripped) sequence.
# --------------------------------------------------------------------------- #
def pairwise_cosine_single(h: torch.Tensor) -> torch.Tensor:
    """h: (n, d) valid tokens -> sim: (n, n) cosine similarity in [-1, 1]."""
    return F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)


def pairwise_poincare_single(
    h: torch.Tensor,
    ball: geoopt.PoincareBall,
    feature_norm: bool = False,
    eps: float = 1e-5,
    feature_mode: str = "none",
    graph_scale: float = 1.0,
) -> torch.Tensor:
    """
    h: (n, d) valid tokens -> D: (n, n) geodesic (Poincare) distances >= 0.

    Tokens are conditioned (see :func:`preprocess_tokens`) and then lifted into
    the ball with the exponential map at the origin. ``feature_norm=True`` is the
    legacy alias for ``feature_mode='l2'`` (the setting under which
    hyperbolic-kNN == cosine-kNN, see module docstring).
    """
    mode = "l2" if feature_norm else feature_mode
    h = preprocess_tokens(h, mode, graph_scale)
    # Lift into the Poincare ball, then clamp just inside the boundary for safety.
    h_ball = ball.expmap0(h)
    h_ball = ball.projx(h_ball)  # geoopt projection keeps points strictly inside
    a = h_ball.unsqueeze(1)      # (n, 1, d)
    b = h_ball.unsqueeze(0)      # (1, n, d)
    D = ball.dist(a, b, dim=-1)  # (n, n)
    # The arccosh-based distance is symmetric in exact arithmetic; in float32 it
    # picks up ~1e-2 asymmetry. Symmetrise and zero the diagonal so edge builders
    # (especially kNN) are deterministic and ties are broken consistently.
    D = 0.5 * (D + D.transpose(-1, -2))
    D.fill_diagonal_(0.0)
    return D


# --------------------------------------------------------------------------- #
# Edge builders -> (edge_index, edge_weight) like GLOT's _threshold_edges.
# --------------------------------------------------------------------------- #
def _threshold_edges_from_sim(sim: torch.Tensor, tau: float, self_loops: bool = True):
    """Binary edges where sim > tau (GLOT's original rule).

    ``self_loops=True`` reproduces upstream exactly: it never zeroes the
    diagonal, and cosine(i, i) = 1 > tau, so every node keeps a self-loop.
    """
    A = (sim > tau).float()
    if not self_loops:
        A.fill_diagonal_(0.0)
    return dense_to_sparse(A)


def _threshold_edges_from_dist(D: torch.Tensor, rho: float, self_loops: bool = True):
    """Binary edges where Poincare distance < rho.

    d(i, i) = 0 < rho, so ``self_loops=True`` keeps one self-loop per node,
    matching upstream's cosine convention.
    """
    A = (D < rho).float()
    if not self_loops:
        A.fill_diagonal_(0.0)
    return dense_to_sparse(A)


def _quantile_rho(D: torch.Tensor, q: float) -> float:
    """Threshold that keeps (about) the closest fraction `q` of node pairs.

    Computed from the off-diagonal entries only, so the self-distances (all 0)
    do not drag the quantile down.
    """
    n = D.size(0)
    if n < 2:
        return float("inf")
    off = D[~torch.eye(n, dtype=torch.bool, device=D.device)]
    return torch.quantile(off.float(), q).item()


def _quantile_tau(sim: torch.Tensor, q: float) -> float:
    """Cosine threshold keeping (about) the most-similar fraction `q` of pairs.

    The similarity twin of :func:`_quantile_rho`: same off-diagonal treatment,
    but the tail of interest is the upper one.
    """
    n = sim.size(0)
    if n < 2:
        return float("-inf")
    off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    return torch.quantile(off.float(), 1.0 - q).item()


def _zscore_weights(score: torch.Tensor, edge_index: torch.Tensor,
                    larger_is_closer: bool) -> torch.Tensor:
    """Per-graph standardised edge weights (zero mean, unit variance).

    The sigmoid weights produced by :func:`_soft_weights` are squashed into a
    narrow band just below 1.0 (measured: [0.77, 1.0], std 0.045), because every
    surviving edge is by construction on the "close" side of the threshold. That
    is a badly conditioned input for the GAT's edge encoder. Standardising the
    raw score across the graph's own edges keeps the full ordering, removes the
    arbitrary offset and scale, and is invariant to curvature (which only
    rescales Poincare distances by 2/sqrt(c)).

    Self-loops are pinned to 0 (the mean) rather than their degenerate extreme
    score, so they neither dominate nor distort the standardisation.
    """
    if edge_index.numel() == 0:
        return score.new_zeros(0)
    s = score[edge_index[0], edge_index[1]].float()
    if not larger_is_closer:
        s = -s                              # make "larger == closer" everywhere
    self_mask = edge_index[0] == edge_index[1]
    real = s[~self_mask]
    if real.numel() < 2:
        return torch.zeros_like(s)
    z = (s - real.mean()) / real.std().clamp_min(1e-6)
    z = torch.where(self_mask, torch.zeros_like(z), z)
    return z.clamp(-4.0, 4.0)               # guard against outlier tokens


def _depth_weights(x_cond: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Signed depth gap along each (directed) edge, standardised per graph.

    Every scalar GLOT can attach to an edge so far is SYMMETRIC: cosine
    similarity and Poincare distance both satisfy w_ij = w_ji, so the graph can
    say *how related* two tokens are but never *which of them is more general*.
    In a Poincare ball the norm is depth, so the signed gap

        g_(j->i) = ||x_j|| - ||x_i||

    tells the receiving token i whether the neighbour it is aggregating from sits
    closer to the root (more generic, g < 0) or further out (more specific,
    g > 0). PyG's GAT is directed -- ``edge_index[0]`` is the source and the
    message flows source -> target -- so an asymmetric attribute is genuinely
    usable, and this is the one piece of hierarchy information a cosine graph is
    structurally incapable of carrying.

    ``x_cond`` must be the CONDITIONED features (see :func:`preprocess_tokens`),
    because depth is only meaningful once the ball has been re-rooted; on raw
    BERT tokens the norms have a coefficient of variation of just 0.057.
    """
    if edge_index.numel() == 0:
        return x_cond.new_zeros(0)
    nrm = x_cond.norm(dim=-1)
    g = nrm[edge_index[0]] - nrm[edge_index[1]]
    self_mask = edge_index[0] == edge_index[1]
    real = g[~self_mask]
    if real.numel() < 2:
        return torch.zeros_like(g)
    g = g / real.std().clamp_min(1e-6)      # mean is already ~0 by antisymmetry
    return g.clamp(-4.0, 4.0)


def _edge_weights(mode: str, score: torch.Tensor, thr: float,
                  edge_index: torch.Tensor, larger_is_closer: bool,
                  temp: float) -> torch.Tensor:
    if mode in ("soft", "hyp"):
        return _soft_weights(score, thr, edge_index, larger_is_closer, temp)
    if mode in ("soft_z", "hyp_z"):
        return _zscore_weights(score, edge_index, larger_is_closer)
    raise ValueError(f"Unknown edge_weight_mode: {mode}")


def _soft_weights(score: torch.Tensor, thr: float, edge_index: torch.Tensor,
                  larger_is_closer: bool, temp: float) -> torch.Tensor:
    """Graded edge weights in (0, 1] for the edges that survived thresholding.

    EDGE WEIGHTING
    --------------
    GLOT hands its GAT an ``edge_dim=1`` attribute but always fills it with 1.0,
    so the *strength* of a relation is discarded the moment the edge passes the
    threshold: a pair at cosine 0.61 and a pair at 0.99 are indistinguishable.
    This is the Fermi-Dirac / sigmoid decoder used by Poincare embeddings
    (Nickel & Kiela 2017) and HGCN (Chami et al. 2019):

        w_ij = sigmoid( (thr - d_ij) / T )      for a distance
        w_ij = sigmoid( (s_ij - thr) / T )      for a similarity

    ``temp`` is expressed as a fraction of the observed spread of ``score`` so
    the same value behaves the same way for cosine similarities in [-1, 1] and
    for Poincare distances in [8.8, 11.8]. The edge SET is unchanged, so this
    isolates the effect of weighting from the effect of sparsity.
    """
    if edge_index.numel() == 0:
        return score.new_zeros(0)
    off = score[~torch.eye(score.size(0), dtype=torch.bool, device=score.device)]
    spread = off.float().std().clamp_min(1e-6)
    T = max(temp * float(spread), 1e-6)
    s = score[edge_index[0], edge_index[1]]
    delta = (s - thr) if larger_is_closer else (thr - s)
    w = torch.sigmoid(delta / T)
    # Self-loops carry the degenerate extreme score (cos=1 / dist=0); pin them to
    # 1.0 so the node's own contribution is never damped by the temperature.
    w = torch.where(edge_index[0] == edge_index[1], torch.ones_like(w), w)
    return w


def _knn_edges(score: torch.Tensor, k: int, larger_is_closer: bool,
               self_loops: bool = True):
    """
    Build a symmetric kNN graph from a pairwise score matrix.

    score: (n, n). If ``larger_is_closer`` (cosine sim) we take top-k largest;
    otherwise (Poincare distance) we take the k smallest. Self is excluded from
    the neighbour search and then re-added if ``self_loops`` (upstream keeps
    one self-loop per node). Returns (edge_index, edge_weight), weights 1.0.
    """
    n = score.size(0)
    s = score.clone()
    diag = torch.arange(n, device=s.device)
    if larger_is_closer:
        s[diag, diag] = float("-inf")
        kk = min(k, n - 1) if n > 1 else 0
        idx = s.topk(kk, dim=-1).indices if kk > 0 else s.new_zeros((n, 0)).long()
    else:
        s[diag, diag] = float("inf")
        kk = min(k, n - 1) if n > 1 else 0
        idx = (-s).topk(kk, dim=-1).indices if kk > 0 else s.new_zeros((n, 0)).long()

    A = torch.zeros((n, n), device=score.device)
    if idx.numel() > 0:
        rows = torch.arange(n, device=score.device).unsqueeze(1).expand_as(idx)
        A[rows.reshape(-1), idx.reshape(-1)] = 1.0
        A = torch.maximum(A, A.t())  # symmetrise
    A.fill_diagonal_(1.0 if self_loops else 0.0)
    return dense_to_sparse(A)


# --------------------------------------------------------------------------- #
# Edge-density telemetry.
#
# Standing rule from the post-mortem: an entire generation of results was
# invalidated because `poincare + threshold` silently produced ZERO edges and
# nobody looked. Every builder call now updates a counter, and the first summary
# is printed early in training so an empty graph is impossible to miss.
# --------------------------------------------------------------------------- #
def _positional_edges(token_idx: torch.Tensor, window: int,
                      self_loops: bool = True):
    """Edges between tokens whose ORIGINAL sequence positions differ by <= window.

    WHY THIS EXISTS: PERMUTATION INVARIANCE
    ---------------------------------------
    GLOT builds edges purely from feature similarity, so the resulting token
    graph is INVARIANT TO TOKEN ORDER: permute the tokens and you get the same
    graph with permuted labels, hence the same pooled embedding. The features
    themselves carry BERT's positional encoding, but the topology the GNN
    propagates over does not, so message passing cannot represent word order.

    That is a poor fit for CoLA, which measures linguistic ACCEPTABILITY -- very
    largely a word-order property. A structure that cannot distinguish
    "the cat sat" from "sat the cat" cannot represent what the task scores, and
    CoLA is indeed one of the tasks where every geometric arm came out flat.

    Adding the local sequence window restores order to the topology. window=1 is
    the plain chain i <-> i+1; larger windows give local n-gram context.

    `token_idx` holds the positions in the ORIGINAL padded sequence, so this is
    correct after padding has been masked out and rows compacted.
    """
    n = token_idx.numel()
    device = token_idx.device
    if n == 0 or window <= 0:
        empty = torch.zeros((2, 0), dtype=torch.long, device=device)
        return empty, torch.zeros((0,), dtype=torch.float, device=device)

    pos = token_idx.view(-1, 1).to(torch.long)
    gap = (pos - pos.t()).abs()
    adj = gap <= int(window)
    if not self_loops:
        adj = adj & (gap > 0)
    src, dst = adj.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=device)
    return edge_index, edge_weight


def _union_edges(ei_a, ew_a, ei_b, ew_b, n: int):
    """Union of two edge sets, de-duplicated, keeping the max weight per edge.

    Encodes (src, dst) as a single integer so duplicates can be removed with
    `torch.unique`; taking the max keeps a graded similarity weight in
    preference to the constant 1.0 a positional edge carries.
    """
    if ei_a.numel() == 0:
        return ei_b, ew_b
    if ei_b.numel() == 0:
        return ei_a, ew_a
    ei = torch.cat([ei_a, ei_b], dim=1)
    ew = torch.cat([ew_a, ew_b], dim=0)
    key = ei[0] * n + ei[1]
    uniq, inv = torch.unique(key, return_inverse=True)
    out_w = torch.zeros(uniq.numel(), dtype=ew.dtype, device=ew.device)
    out_w.scatter_reduce_(0, inv, ew, reduce="amax", include_self=False)
    out_ei = torch.stack([uniq // n, uniq % n], dim=0)
    return out_ei, out_w


class _GraphStats:
    def __init__(self, report_after: int = 256):
        self.report_after = report_after
        self.reset()

    def reset(self):
        self.n_graphs = 0
        self.sum_density = 0.0
        self.n_empty = 0
        self.reported = False

    def observe(self, n_nodes: int, edges):
        """Accumulate OFF-DIAGONAL edge density.

        `edges` may be an edge_index tensor (preferred: self-loops are counted
        exactly) or an int edge count (legacy: self-loops are assumed present,
        one per node, which is what every adjacency in this file emits when
        cfg.self_loops is set).

        The previous version divided a self-loop-inclusive count by n(n-1) and
        so returned n/(n-1) > 1 for a complete graph. Densities logged before
        this fix are inflated by 1/(n-1) and must not be compared with ones
        logged after it.
        """
        if n_nodes < 2:
            return
        if hasattr(edges, "size"):          # edge_index tensor (2, E)
            n_edges = int(edges.size(1))
            n_self = int((edges[0] == edges[1]).sum()) if n_edges else 0
        else:                                # legacy int
            n_edges = int(edges)
            n_self = min(n_nodes, n_edges)
        n_off = max(0, n_edges - n_self)
        self.n_graphs += 1
        self.sum_density += n_off / (n_nodes * (n_nodes - 1))
        self.n_empty += int(n_off == 0)
        if not self.reported and self.n_graphs >= self.report_after:
            self.reported = True
            print(f"[graph] density={self.sum_density / self.n_graphs:.5f} "
                  f"empty_frac={self.n_empty / self.n_graphs:.5f} "
                  f"(over {self.n_graphs} graphs)", flush=True)


_GRAPH_STATS = _GraphStats()


# --------------------------------------------------------------------------- #
# Public API: drop-in builder mirroring GLOT.build_pyg_graphs.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_pyg_graphs_hyper(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    cfg: HyperbolicGraphConfig,
    device: Optional[torch.device] = None,
) -> Batch:
    """
    Convert a batch of token sequences into a PyG ``Batch`` of graphs, choosing
    edges by cosine similarity (GLOT) or hyperbolic distance (Stage A).

    Args:
        hidden:         (B, L, d) last-layer token features (frozen LLM output).
        attention_mask: (B, L) 1=valid, 0=pad.
        cfg:            HyperbolicGraphConfig.
        device:         where to place the resulting graphs.

    Returns:
        torch_geometric.data.Batch with per-graph fields x, edge_index,
        edge_attr (shape (E, 1)), token_idx -- identical schema to GLOT so the
        existing GLOT GNN/readout consume it unchanged.
    """
    assert hidden.dim() == 3 and attention_mask.dim() == 2, "Bad input shapes"
    B, L, d = hidden.shape
    device = device or hidden.device

    ball = None
    if cfg.graph_metric == "poincare" or cfg.edge_weight_mode in ("hyp", "hyp_z"):
        ball = geoopt.PoincareBall(c=cfg.curvature)

    graphs: List[Data] = []
    for b in range(B):
        mask_b = attention_mask[b].to(dtype=torch.bool)
        x_b = hidden[b, mask_b]                                   # (n, d)
        token_idx = torch.arange(L, device=device)[mask_b]       # (n,)

        if cfg.graph_metric == "cosine":
            sim = pairwise_cosine_single(x_b)
            if cfg.adjacency == "threshold":
                tau_eff = (_quantile_tau(sim, cfg.tau_quantile)
                           if 0.0 < cfg.tau_quantile < 1.0 else cfg.tau)
                edge_index, edge_weight = _threshold_edges_from_sim(
                    sim, tau_eff, cfg.self_loops)
                if cfg.edge_weight_mode in ("soft", "soft_z"):
                    edge_weight = _edge_weights(
                        cfg.edge_weight_mode, sim, tau_eff, edge_index,
                        larger_is_closer=True, temp=cfg.edge_temp)
                elif cfg.edge_weight_mode in ("hyp", "hyp_z"):
                    # Keep GLOT's proven cosine edge SET, but describe each edge
                    # with hyperbolic proximity instead of a constant 1.0. This
                    # is strictly additive: the topology the paper validated is
                    # untouched, and the GAT simply receives information cosine
                    # thresholding throws away (see EDGE WEIGHTING note).
                    D = pairwise_poincare_single(
                        x_b, ball, cfg.feature_norm, cfg.eps,
                        feature_mode=cfg.feature_mode, graph_scale=cfg.graph_scale)
                    edge_weight = _edge_weights(
                        cfg.edge_weight_mode, D, _quantile_rho(D, 0.5), edge_index,
                        larger_is_closer=False, temp=cfg.edge_temp)
                elif cfg.edge_weight_mode == "depth":
                    edge_weight = _depth_weights(
                        preprocess_tokens(x_b, cfg.feature_mode, cfg.graph_scale),
                        edge_index)
            elif cfg.adjacency == "knn":
                edge_index, edge_weight = _knn_edges(
                    sim, cfg.k, larger_is_closer=True, self_loops=cfg.self_loops)
            else:
                raise ValueError(f"Unknown adjacency: {cfg.adjacency}")
        elif cfg.graph_metric == "poincare":
            D = pairwise_poincare_single(
                x_b, ball, cfg.feature_norm, cfg.eps,
                feature_mode=cfg.feature_mode, graph_scale=cfg.graph_scale,
            )
            if cfg.adjacency == "threshold":
                if 0.0 < cfg.rho_quantile < 1.0:
                    rho_eff = _quantile_rho(D, cfg.rho_quantile)
                else:
                    rho_eff = cfg.rho
                edge_index, edge_weight = _threshold_edges_from_dist(
                    D, rho_eff, cfg.self_loops)
                if cfg.edge_weight_mode == "depth":
                    edge_weight = _depth_weights(
                        preprocess_tokens(x_b, cfg.feature_mode, cfg.graph_scale),
                        edge_index)
                elif cfg.edge_weight_mode != "binary":
                    edge_weight = _edge_weights(
                        cfg.edge_weight_mode.replace("hyp", "soft"), D, rho_eff,
                        edge_index, larger_is_closer=False, temp=cfg.edge_temp)
            elif cfg.adjacency == "knn":
                edge_index, edge_weight = _knn_edges(
                    D, cfg.k, larger_is_closer=False, self_loops=cfg.self_loops)
            else:
                raise ValueError(f"Unknown adjacency: {cfg.adjacency}")
        else:
            raise ValueError(f"Unknown graph_metric: {cfg.graph_metric}")

        # --- word-order structure (default off, upstream unchanged) ----------
        # The similarity graph above is permutation-invariant over tokens; these
        # edges are the only thing that lets message passing see word order.
        # See _positional_edges for why that matters (CoLA is an order task).
        if cfg.positional_window > 0:
            pos_ei, pos_ew = _positional_edges(
                token_idx, cfg.positional_window, cfg.self_loops)
            if cfg.positional_only:
                # Ablation: order ONLY, no semantics. Tells us how much of any
                # gain is word order rather than feature similarity.
                edge_index, edge_weight = pos_ei, pos_ew
            else:
                edge_index, edge_weight = _union_edges(
                    edge_index, edge_weight, pos_ei, pos_ew, x_b.size(0))

        _GRAPH_STATS.observe(x_b.size(0), edge_index)

        # GLOT's GAT uses edge_dim=1, so keep edge_attr shaped (E, 1).
        edge_weight = edge_weight.view(-1, 1).float() if edge_weight.numel() else edge_weight.view(-1, 1)
        data = Data(x=x_b, edge_index=edge_index, edge_attr=edge_weight).to(device)
        data.token_idx = token_idx
        graphs.append(data)

    return Batch.from_data_list(graphs)
