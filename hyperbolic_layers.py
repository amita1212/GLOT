"""
Stage B (hyperbolic readout) and Stage C (hyperbolic Token-GNN) of HyperGLOT.

This module is a companion to ``hyperbolic_graph.py`` (Stage A). It provides the
two remaining hyperbolic building blocks described in the research report, kept
completely **modular** so that the original GLOT path is never touched:

  * **Stage B - hyperbolic readout** (:func:`hyperbolic_readout`):
      aggregate the refined token vectors with a curvature-aware
      **Einstein / gyro midpoint** in the Poincare ball (instead of GLOT's
      Euclidean weighted sum), then map the sentence vector back to the tangent
      space so the *existing* Euclidean classifier/projection heads consume it
      unchanged. The attention weights ``pi_i`` are exactly GLOT's readout
      scores, so only the *aggregation geometry* changes.

  * **Stage C - hyperbolic Token-GNN** (:class:`HyperbolicGCNConv`):
      an HGCN-style message-passing layer. Node features live on the Poincare
      ball; each layer does a **hyperbolic linear** map (tangent-space linear at
      the origin) followed by **neighbourhood aggregation in the tangent space**
      (log -> weighted sum -> exp), then a hyperbolic non-linearity. This mirrors
      the Stage C math in the report:

          h'_i = W (x)_c h_i (+)_c b            (Mobius / tangent linear)
          a_i  = exp0( sum_j alpha_ij log0(h'_j) )  (tangent aggregation)
          h_i  = exp0( sigma( log0(a_i) ) )        (hyperbolic activation)

Everything is written against :mod:`geoopt`'s :class:`PoincareBall`. When the
curvature ``c -> 0`` the ball becomes Euclidean, so GLOT is recovered as the
``c -> 0`` limit (mirroring how GLOT generalises mean pooling).

Dependencies: torch, torch_geometric, geoopt.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax as pyg_softmax

try:
    from torch_scatter import scatter_add
except Exception:  # pragma: no cover - fallback when torch_scatter isn't installed
    def scatter_add(src, index, dim=-1, out=None, dim_size=None):
        """index_add-based drop-in for torch_scatter.scatter_add."""
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

import geoopt


# --------------------------------------------------------------------------- #
# Poincare <-> Klein helpers (curvature-aware) for the Einstein midpoint.
# --------------------------------------------------------------------------- #
def _poincare_to_klein(x: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Map Poincare-ball points to the Klein model (ball of curvature -c)."""
    x2 = (x * x).sum(dim=-1, keepdim=True)
    return 2.0 * x / (1.0 + c * x2 + eps)


def _klein_to_poincare(k: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Map Klein-model points back to the Poincare ball."""
    k2 = (k * k).sum(dim=-1, keepdim=True)
    denom = 1.0 + torch.sqrt(torch.clamp(1.0 - c * k2, min=eps))
    return k / denom


def _klein_lorentz_factor(k: torch.Tensor, c: float, eps: float = 1e-5) -> torch.Tensor:
    """Lorentz factor gamma = 1 / sqrt(1 - c ||k||^2) for Klein points."""
    k2 = (k * k).sum(dim=-1, keepdim=True)
    return 1.0 / torch.sqrt(torch.clamp(1.0 - c * k2, min=eps))


# --------------------------------------------------------------------------- #
# Stage B: hyperbolic (Einstein / gyro midpoint) readout.
# --------------------------------------------------------------------------- #
def hyperbolic_readout(
    h_all: torch.Tensor,
    weights: torch.Tensor,
    batch: torch.Tensor,
    ball: geoopt.PoincareBall,
    curvature,
    num_graphs: Optional[int] = None,
    eps: float = 1e-5,
    scale: Optional[torch.Tensor] = None,
    clip: float = 0.0,
) -> torch.Tensor:
    """Weighted hyperbolic pooling via the Einstein (Klein) midpoint.

    Args:
        h_all:   (N, D) refined token features (tangent / Euclidean space).
        weights: (N,)  per-token attention weights (GLOT's softmax readout).
        batch:   (N,)  graph id of each token (PyG ``batch`` vector).
        ball:    geoopt ``PoincareBall`` used for the exp/log maps.
        curvature: ball curvature magnitude ``c`` (float or a 0-dim tensor when
            the curvature is learnable; must match ``ball``).
        num_graphs: number of graphs in the batch (defaults to ``batch.max()+1``).
        scale:   optional learnable scalar multiplying the tangent features before
            the exponential map. Controls how far into the ball tokens are lifted
            (Khrulkov et al. 2020); ``None`` reproduces the original behaviour.
        clip:    optional max Euclidean norm for the tangent features before the
            exp map. Clipping prevents ``expmap0`` from saturating at the ball
            boundary (the vanishing-gradient failure mode diagnosed by Guo et al.
            2022). ``0`` disables clipping (original behaviour).

    Returns:
        z: (B, D) sentence vectors mapped back to the tangent space so the
           downstream Euclidean head is unchanged. The aggregation itself is a
           genuine curvature-dependent gyro-midpoint (not a Euclidean sum).
    """
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0

    # Keep ``c`` as-is so a learnable (tensor) curvature stays differentiable.
    c = curvature

    # --- Stage B stabilisation (fixes the boundary-saturation failure) --------
    h = h_all
    if scale is not None:
        h = h * scale                              # learnable input scale
    if clip and clip > 0:
        norm = h.norm(dim=-1, keepdim=True).clamp_min(eps)
        factor = torch.clamp(clip / norm, max=1.0)  # ||h|| <= clip
        h = h * factor

    # Lift refined tokens into the Poincare ball, then to the Klein model.
    p = ball.projx(ball.expmap0(h))               # (N, D) on the ball
    k = _poincare_to_klein(p, c, eps=eps)          # (N, D) Klein coords
    gamma = _klein_lorentz_factor(k, c, eps=eps)   # (N, 1) Lorentz factors

    # Einstein midpoint: weight each point by (attention * Lorentz factor).
    w = (weights.unsqueeze(-1) * gamma)            # (N, 1)
    num = scatter_add(w * k, batch, dim=0, dim_size=num_graphs)   # (B, D)
    den = scatter_add(w, batch, dim=0, dim_size=num_graphs).clamp_min(eps)
    mid_k = num / den                              # (B, D) Klein midpoint

    mid_p = _klein_to_poincare(mid_k, c, eps=eps)  # (B, D) back to the ball
    mid_p = ball.projx(mid_p)
    z = ball.logmap0(mid_p)                         # (B, D) tangent for the head
    return z


# --------------------------------------------------------------------------- #
# Stage C: hyperbolic Token-GNN layer (HGCN-style).
# --------------------------------------------------------------------------- #
class HyperbolicLinear(nn.Module):
    """Hyperbolic linear map: log0 -> Euclidean linear -> exp0 (+ Mobius bias).

    Operates on points that live on the Poincare ball and returns points on the
    ball. Implemented via the tangent space at the origin, which is numerically
    stable and is the standard Hyperbolic-Neural-Network formulation.
    """

    def __init__(self, in_dim: int, out_dim: int, ball: geoopt.PoincareBall, use_bias: bool = True):
        super().__init__()
        self.ball = ball
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x_ball: torch.Tensor) -> torch.Tensor:
        xt = self.ball.logmap0(x_ball)          # ball -> tangent (Euclidean)
        xt = F.linear(xt, self.weight)          # Euclidean linear
        out = self.ball.expmap0(xt)             # tangent -> ball
        if self.use_bias:
            b = self.ball.expmap0(self.bias.unsqueeze(0))
            out = self.ball.mobius_add(out, b)
        return self.ball.projx(out)


class HyperbolicGCNConv(MessagePassing):
    """HGCN-style graph conv: hyperbolic linear + tangent-space aggregation.

    Node features enter and leave on the Poincare ball. Aggregation is performed
    in the tangent space at the origin with GCN-style symmetric normalisation
    (optionally scaled by the incoming ``edge_weight``), then mapped back to the
    ball. A hyperbolic ReLU is applied at the end.
    """

    def __init__(self, in_dim: int, out_dim: int, ball: geoopt.PoincareBall):
        super().__init__(aggr="add")
        self.ball = ball
        self.lin = HyperbolicLinear(in_dim, out_dim, ball)

    def forward(self, x_ball: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = x_ball.size(0)

        # Hyperbolic linear map, then move to the tangent space to aggregate.
        h = self.lin(x_ball)                    # (n, out_dim) on the ball
        ht = self.ball.logmap0(h)               # (n, out_dim) tangent

        # GCN-style symmetric normalisation with self loops.
        if edge_weight is not None:
            edge_weight = edge_weight.view(-1).float()
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=n
        )
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=ht.device)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=n).clamp_min(1e-12)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        agg_t = self.propagate(edge_index, x=ht, norm=norm)   # tangent aggregation

        # Back to the ball, then a hyperbolic non-linearity.
        out = self.ball.projx(self.ball.expmap0(agg_t))
        out = self.ball.projx(self.ball.expmap0(F.relu(self.ball.logmap0(out))))
        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class HyperbolicGATConv(MessagePassing):
    """Attention-weighted hyperbolic graph conv (hyperbolic GAT).

    Improves on :class:`HyperbolicGCNConv` by replacing the fixed symmetric
    degree normalisation with **learned attention** over neighbours, mirroring
    Hyperbolic Attention Networks (Gulcehre et al. 2019) and HGCN's attention
    aggregation (Chami et al. 2019). This removes the confound whereby the plain
    hyperbolic-GCN arm silently downgraded GLOT's GAT attention to GCN.

    Node features enter and leave on the Poincare ball. A hyperbolic linear map
    is applied, features are moved to the tangent space, GAT-style attention
    weights are computed there and used for aggregation, then the result is
    mapped back to the ball with a hyperbolic ReLU.
    """

    def __init__(self, in_dim: int, out_dim: int, ball: geoopt.PoincareBall,
                 negative_slope: float = 0.2):
        super().__init__(aggr="add", node_dim=0)
        self.ball = ball
        self.lin = HyperbolicLinear(in_dim, out_dim, ball)
        self.att_src = nn.Parameter(torch.empty(1, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.negative_slope = negative_slope

    def forward(self, x_ball: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        n = x_ball.size(0)

        # Hyperbolic linear map, then to the tangent space for attention.
        h = self.lin(x_ball)                    # (n, out_dim) on the ball
        ht = self.ball.logmap0(h)               # (n, out_dim) tangent

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n)

        # GAT-style additive attention logits (source/target projections).
        alpha_src = (ht * self.att_src).sum(dim=-1)   # (n,)
        alpha_dst = (ht * self.att_dst).sum(dim=-1)   # (n,)

        agg_t = self.propagate(
            edge_index, x=ht, alpha=(alpha_src, alpha_dst)
        )                                        # tangent aggregation

        # Back to the ball with a single exp map, then a hyperbolic ReLU.
        out = self.ball.projx(self.ball.expmap0(F.relu(agg_t)))
        return out

    def message(self, x_j: torch.Tensor, alpha_j: torch.Tensor,
                alpha_i: torch.Tensor, index: torch.Tensor,
                ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor:
        alpha = F.leaky_relu(alpha_j + alpha_i, self.negative_slope)
        alpha = pyg_softmax(alpha, index, ptr, size_i)   # attention over neighbours
        return x_j * alpha.unsqueeze(-1)
