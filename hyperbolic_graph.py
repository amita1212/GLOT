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
    rho: float = 1.0                # hyperbolic-distance threshold (graph_metric == "poincare")
    k: int = 8                      # neighbours for knn adjacency
    curvature: float = 1.0          # Poincare ball curvature c (> 0)
    feature_norm: bool = False      # L2-normalise token features before mapping into the ball
    eps: float = 1e-5               # boundary safety clamp for the ball


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
) -> torch.Tensor:
    """
    h: (n, d) valid tokens -> D: (n, n) geodesic (Poincare) distances >= 0.

    Tokens are lifted into the ball with the exponential map at the origin
    (expmap0). Optionally L2-normalise first (this is the setting under which
    hyperbolic-kNN == cosine-kNN, see module docstring).
    """
    if feature_norm:
        h = F.normalize(h, dim=-1)
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
def _threshold_edges_from_sim(sim: torch.Tensor, tau: float):
    """Binary edges where sim > tau (GLOT's original rule), no self loops."""
    A = (sim > tau).float()
    A.fill_diagonal_(0.0)
    return dense_to_sparse(A)


def _threshold_edges_from_dist(D: torch.Tensor, rho: float):
    """Binary edges where Poincare distance < rho, no self loops."""
    A = (D < rho).float()
    A.fill_diagonal_(0.0)
    return dense_to_sparse(A)


def _knn_edges(score: torch.Tensor, k: int, larger_is_closer: bool):
    """
    Build a symmetric kNN graph from a pairwise score matrix.

    score: (n, n). If ``larger_is_closer`` (cosine sim) we take top-k largest;
    otherwise (Poincare distance) we take the k smallest. Self is excluded.
    Returns (edge_index, edge_weight) with binary weights (1.0).
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
    A.fill_diagonal_(0.0)
    return dense_to_sparse(A)


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
    if cfg.graph_metric == "poincare":
        ball = geoopt.PoincareBall(c=cfg.curvature)

    graphs: List[Data] = []
    for b in range(B):
        mask_b = attention_mask[b].to(dtype=torch.bool)
        x_b = hidden[b, mask_b]                                   # (n, d)
        token_idx = torch.arange(L, device=device)[mask_b]       # (n,)

        if cfg.graph_metric == "cosine":
            sim = pairwise_cosine_single(x_b)
            if cfg.adjacency == "threshold":
                edge_index, edge_weight = _threshold_edges_from_sim(sim, cfg.tau)
            elif cfg.adjacency == "knn":
                edge_index, edge_weight = _knn_edges(sim, cfg.k, larger_is_closer=True)
            else:
                raise ValueError(f"Unknown adjacency: {cfg.adjacency}")
        elif cfg.graph_metric == "poincare":
            D = pairwise_poincare_single(x_b, ball, cfg.feature_norm, cfg.eps)
            if cfg.adjacency == "threshold":
                edge_index, edge_weight = _threshold_edges_from_dist(D, cfg.rho)
            elif cfg.adjacency == "knn":
                edge_index, edge_weight = _knn_edges(D, cfg.k, larger_is_closer=False)
            else:
                raise ValueError(f"Unknown adjacency: {cfg.adjacency}")
        else:
            raise ValueError(f"Unknown graph_metric: {cfg.graph_metric}")

        # GLOT's GAT uses edge_dim=1, so keep edge_attr shaped (E, 1).
        edge_weight = edge_weight.view(-1, 1).float() if edge_weight.numel() else edge_weight.view(-1, 1)
        data = Data(x=x_b, edge_index=edge_index, edge_attr=edge_weight).to(device)
        data.token_idx = token_idx
        graphs.append(data)

    return Batch.from_data_list(graphs)
