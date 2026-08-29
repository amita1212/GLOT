"""
Curvature diagnostic for HyperGLOT Stage A.

Central question
----------------
Every Stage A run so far used the geoopt default curvature ``c = 1.0``. Measured
BERT token norms are ~14.7, so ``expmap0`` maps every token to
``tanh(sqrt(c)*||x||)/sqrt(c)``, i.e. relative radius ``tanh(14.7) = 1.0`` in
float32 -- EXACTLY the ball boundary.

On the boundary the Poincare distance degenerates: two points at the same radius
have a geodesic distance that is a strictly increasing function of the angle
between them and nothing else. If that is what happened, then

    "Stage A (poincare)"  ==  "GLOT (cosine)"          (identical edge sets)

and the entire negative result for Stage A is a measurement of a knob that was
never connected to anything.

This script tests that on REAL cached features, and looks for the curvature
range where the token *norm* actually contributes to the geometry.

Note on parametrisation
-----------------------
expmap0 depends on the features only through ``sqrt(c) * ||x||``, and the
Poincare distance is ``(2/sqrt(c)) * artanh(...)`` -- a global positive factor
that cannot change edge rankings or distance quantiles. Therefore, *for graph
construction*, curvature ``c`` and a feature rescale ``s`` are the same knob
with ``s = sqrt(c)``. We report the "effective radius" ``sqrt(c)*mean||x||``
because that is the quantity that actually decides whether we saturate.

Usage:
    python diagnose_curvature.py --cache_dir ./data/bert-base-uncased_cola_train_batches
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F
import geoopt


# --------------------------------------------------------------------------- #
# Cache loading
# --------------------------------------------------------------------------- #
def load_sentences(cache_dir: str, max_sents: int = 64, min_tokens: int = 8):
    """Yield (n, d) float32 token matrices for real sentences from the cache."""
    meta_path = os.path.join(cache_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        batch_files = meta["batch_files"]
    else:
        batch_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    out = []
    for bf in batch_files:
        if not os.path.isabs(bf) and not os.path.exists(bf):
            bf = os.path.join(cache_dir, os.path.basename(bf))
        data = torch.load(bf, map_location="cpu")
        for hs, ms in zip(data["a_hs"], data["a_ms"]):
            mask = ms.to(torch.bool)
            x = hs[mask].float()
            if x.size(0) >= min_tokens:
                out.append(x)
            if len(out) >= max_sents:
                return out
    return out


# --------------------------------------------------------------------------- #
# Rank utilities (Spearman without scipy)
# --------------------------------------------------------------------------- #
def _rank(v: torch.Tensor) -> torch.Tensor:
    """Average-free ordinal ranks (ties broken by sort order; fine at this scale)."""
    order = v.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(v.numel(), dtype=torch.float64)
    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / denom)


def offdiag(M: torch.Tensor) -> torch.Tensor:
    n = M.size(0)
    return M[~torch.eye(n, dtype=torch.bool)]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def poincare_dist_matrix(x: torch.Tensor, c: float) -> torch.Tensor:
    ball = geoopt.PoincareBall(c=c)
    p = ball.projx(ball.expmap0(x))
    D = ball.dist(p.unsqueeze(1), p.unsqueeze(0), dim=-1)
    D = 0.5 * (D + D.t())
    D.fill_diagonal_(0.0)
    return D


def relative_radius(x: torch.Tensor, c: float) -> torch.Tensor:
    """||expmap0(x)|| / (1/sqrt(c))  in [0, 1). 1.0 == on the boundary."""
    ball = geoopt.PoincareBall(c=c)
    p = ball.expmap0(x)
    return p.norm(dim=-1) * (c ** 0.5)


def edges_at_density(score: torch.Tensor, density: float, larger_is_closer: bool):
    """Boolean off-diagonal edge mask keeping exactly `density` of the pairs."""
    v = offdiag(score)
    q = density if larger_is_closer else density
    if larger_is_closer:
        thr = torch.quantile(v.float(), 1.0 - q)
        return offdiag(score) > thr
    thr = torch.quantile(v.float(), q)
    return offdiag(score) < thr


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    inter = (a & b).sum().item()
    union = (a | b).sum().item()
    return inter / union if union else 1.0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str,
                    default="./data/bert-base-uncased_cola_train_batches")
    ap.add_argument("--max_sents", type=int, default=64)
    ap.add_argument("--density", type=float, default=0.10,
                    help="edge density at which to compare graphs (cosine tau=0.6 ~ 0.10)")
    ap.add_argument("--curvatures", type=str,
                    default="1e-6,1e-5,1e-4,3e-4,1e-3,3e-3,5e-3,1e-2,3e-2,1e-1,3e-1,1.0,3.0,10.0")
    args = ap.parse_args()

    sents = load_sentences(args.cache_dir, args.max_sents)
    print(f"Loaded {len(sents)} sentences from {args.cache_dir}")
    norms = torch.cat([x.norm(dim=-1) for x in sents])
    print(f"token L2 norm: mean {norms.mean():.3f}  min {norms.min():.3f}  "
          f"max {norms.max():.3f}  std {norms.std():.3f}")
    print()

    curvatures = [float(s) for s in args.curvatures.split(",")]

    header = (f"{'c':>9} {'sqrt(c)*|x|':>11} {'rel_r mean':>10} {'rel_r max':>10} "
              f"{'satur%':>7} {'sp(D,cos)':>10} {'sp(D,eucl)':>10} "
              f"{'sp(D,norms)':>11} {'Jacc_vs_cos':>11}")
    print(header)
    print("-" * len(header))

    rows = []
    for c in curvatures:
        rr_all, sp_cos, sp_euc, sp_nrm, jac, sat = [], [], [], [], [], []
        for x in sents:
            n = x.size(0)
            D = poincare_dist_matrix(x, c)
            S = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
            E = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
            nn_ = x.norm(dim=-1)
            NS = nn_.unsqueeze(1) + nn_.unsqueeze(0)   # norm "depth" surrogate

            rr = relative_radius(x, c)
            rr_all.append(rr.mean().item())
            sat.append((rr > 1.0 - 1e-6).float().mean().item())

            d, s, e, ns = offdiag(D), offdiag(S), offdiag(E), offdiag(NS)
            sp_cos.append(spearman(d, -s))     # +1 => distance is a pure angle function
            sp_euc.append(spearman(d, e))      # +1 => distance is essentially Euclidean
            sp_nrm.append(spearman(d, ns))

            ec = edges_at_density(S, args.density, larger_is_closer=True)
            ep = edges_at_density(D, args.density, larger_is_closer=False)
            jac.append(jaccard(ec, ep))

        m = lambda v: sum(v) / len(v)
        rows.append(dict(c=c, eff=(c ** 0.5) * norms.mean().item(),
                         rel_r=m(rr_all), sat=m(sat), sp_cos=m(sp_cos),
                         sp_euc=m(sp_euc), sp_nrm=m(sp_nrm), jacc=m(jac)))
        r = rows[-1]
        print(f"{c:9.2e} {r['eff']:11.3f} {r['rel_r']:10.6f} "
              f"{max(rr_all):10.6f} {100*r['sat']:6.1f}% {r['sp_cos']:10.4f} "
              f"{r['sp_euc']:10.4f} {r['sp_nrm']:11.4f} {r['jacc']:11.4f}")

    print()
    print("READING THE TABLE")
    print("  sp(D,cos)   = Spearman(poincare distance, -cosine sim). 1.0000 means the")
    print("                hyperbolic graph is a RELABELLING of the cosine graph:")
    print("                Stage A can never differ from GLOT at any density.")
    print("  Jacc_vs_cos = edge-set overlap with cosine at MATCHED density.")
    print("                1.0 = identical graph. This is the honest 'am I doing")
    print("                anything at all' number.")
    print("  satur%      = fraction of tokens sitting exactly on the ball boundary.")

    best = min(rows, key=lambda r: r["jacc"])
    print()
    print(f"Most DIFFERENT-from-cosine curvature: c = {best['c']:.2e} "
          f"(Jaccard {best['jacc']:.4f}, effective radius {best['eff']:.3f})")


if __name__ == "__main__":
    main()
