#!/usr/bin/env python3
"""Quick sanity check for the new hyperbolic layers (ABfix readout + hyperbolic GAT)."""
import torch
import geoopt
from hyperbolic_layers import HyperbolicGATConv, HyperbolicGCNConv, hyperbolic_readout

torch.manual_seed(0)
ball = geoopt.PoincareBall(c=1.0, learnable=True)

# --- Hyperbolic GAT: in-ball, finite, differentiable ---
x_leaf = torch.randn(6, 16, requires_grad=True)   # leaf we track grads on
x = ball.projx(ball.expmap0(x_leaf))
ei = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2], [1, 0, 3, 2, 5, 4, 2, 0]])
gat = HyperbolicGATConv(16, 8, ball)
out = gat(x, ei)
assert out.shape == (6, 8), out.shape
assert torch.isfinite(out).all(), "GAT output not finite"
out.sum().backward()
assert x_leaf.grad is not None and torch.isfinite(x_leaf.grad).all(), "GAT grad bad"
assert gat.att_src.grad is not None and torch.isfinite(gat.att_src.grad).all(), "GAT attention grad bad"
print("HyperbolicGATConv OK:", tuple(out.shape), "input+attention grads finite")

# --- Fixed Stage B readout: scale + clip + learnable curvature ---
h = (torch.randn(6, 32) * 15.0).requires_grad_(True)  # large-norm features (the failure regime)
w = torch.softmax(torch.randn(6), dim=0)
batch = torch.tensor([0, 0, 0, 1, 1, 1])
scale = torch.nn.Parameter(torch.tensor(0.1))
z_fixed = hyperbolic_readout(h, w, batch, ball, ball.c, num_graphs=2, scale=scale, clip=2.0)
assert z_fixed.shape == (2, 32), z_fixed.shape
assert torch.isfinite(z_fixed).all(), "fixed readout not finite"
z_fixed.sum().backward()
assert torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0, "fixed readout grad vanished"
assert scale.grad is not None and torch.isfinite(scale.grad), "scale grad bad"
# Learnable curvature: geoopt returns ball.c as a derived tensor, so check the
# ball's actual trainable parameter(s) received a finite gradient.
ball_param_grads = [p.grad for p in ball.parameters() if p.grad is not None]
assert ball_param_grads and all(torch.isfinite(g).all() for g in ball_param_grads), "curvature grad bad"
print("Fixed Stage B readout OK: grad flows, scale.grad=%.3e, curvature-param grads finite" %
      (scale.grad.item(),))

# --- Compare grad magnitude vs the OLD unclipped path (should be >= as healthy) ---
h2 = (torch.randn(6, 32) * 15.0).requires_grad_(True)
ball2 = geoopt.PoincareBall(c=1.0)
z_old = hyperbolic_readout(h2, w, batch, ball2, 1.0, num_graphs=2)  # no scale/clip
z_old.sum().backward()
print("OLD readout grad-norm: %.3e | FIXED grad-norm: %.3e" %
      (h2.grad.norm().item(), h.grad.norm().item()))
print("ALL SANITY CHECKS PASSED")
