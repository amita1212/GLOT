"""
Anisotropy / preprocessing diagnostic for HyperGLOT Stage A.

Why
---
In a Poincare ball the ORIGIN is the root of the hierarchy and a point's norm is
its depth. Stage A lifts raw BERT tokens with ``expmap0``, which silently asserts
that the coordinate origin is a meaningful root. BERT is famously anisotropic:
its token cloud sits inside a narrow cone displaced from the origin by a large
common mean vector. If ``||mean(x)||`` is comparable to ``mean||x||`` then every
token has almost the same norm, the "depth" coordinate carries no variance, and
the hyperbolic graph cannot express hierarchy no matter what curvature we pick.

This script quantifies that, and compares candidate preprocessings:

    none         x                            (what Stage A does today)
    center       x - mu                       (mu = per-sentence token mean)
    center_unit  (x - mu) / mean||x - mu||    (scale-free: curvature transfers)
    global_center x - mu_corpus               (mu over the whole sample)
    l2           x / ||x||                    (provably == cosine, control)

Reported per mode:
  cv_norm     coefficient of variation of the token norms. This is the amount of
              hierarchy signal available. Near 0 => no depth information.
  sp(D,norms) Spearman(Poincare distance, sum of endpoint norms) at the best
              curvature: does the geometry actually use the depth coordinate?
  jacc_cos    edge-set Jaccard vs the cosine graph at matched density. 1.0 means
              "identical to GLOT", i.e. the arm cannot possibly behave differently.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from diagnose_curvature import (  # noqa: E402
    load_sentences, spearman, offdiag, poincare_dist_matrix,
    edges_at_density, jaccard,
)


def preprocess(x: torch.Tensor, mode: str, mu_global: torch.Tensor) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "l2":
        return F.normalize(x, dim=-1)
    if mode == "center":
        return x - x.mean(0, keepdim=True)
    if mode == "center_unit":
        z = x - x.mean(0, keepdim=True)
        return z / z.norm(dim=-1).mean().clamp_min(1e-6)
    if mode == "global_center":
        return x - mu_global
    if mode == "global_center_unit":
        z = x - mu_global
        return z / z.norm(dim=-1).mean().clamp_min(1e-6)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="./data/bert-base-uncased_cola_train_batches")
    ap.add_argument("--max_sents", type=int, default=64)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--curvatures", type=str, default="1e-3,3e-3,1e-2,3e-2,1e-1,1.0")
    args = ap.parse_args()

    sents = load_sentences(args.cache_dir, args.max_sents)
    allx = torch.cat(sents, 0)
    mu_global = allx.mean(0, keepdim=True)

    print(f"{len(sents)} sentences, {allx.size(0)} tokens, dim {allx.size(1)}")
    print(f"mean||x||        = {allx.norm(dim=-1).mean():.4f}")
    print(f"||mean(x)||      = {mu_global.norm():.4f}   <-- anisotropy: if this is close")
    print(f"                       to mean||x||, the cloud is a displaced cone and the")
    print(f"                       coordinate origin is NOT a meaningful hierarchy root.")
    cos_to_mu = F.cosine_similarity(allx, mu_global, dim=-1)
    print(f"mean cos(x, mu)  = {cos_to_mu.mean():.4f}  (1.0 = perfectly collinear cone)")
    print()

    modes = ["none", "l2", "center", "center_unit", "global_center", "global_center_unit"]
    curvatures = [float(s) for s in args.curvatures.split(",")]

    hdr = f"{'mode':>19} {'mean|x|':>8} {'cv_norm':>8} | " + " ".join(
        f"{'c=' + f'{c:g}':>11}" for c in curvatures)
    print(hdr)
    print(f"{'':>19} {'':>8} {'':>8} | " + " ".join(f"{'jacc/spN':>11}" for _ in curvatures))
    print("-" * len(hdr))

    for mode in modes:
        norms, cvs = [], []
        cells = []
        proc = [preprocess(x, mode, mu_global) for x in sents]
        for x in proc:
            n = x.norm(dim=-1)
            norms.append(n.mean().item())
            cvs.append((n.std() / n.mean().clamp_min(1e-9)).item())

        for c in curvatures:
            jac, spn = [], []
            for x, xraw in zip(proc, sents):
                D = poincare_dist_matrix(x, c)
                # cosine graph is ALWAYS built on the raw features (that is GLOT)
                S = F.cosine_similarity(xraw.unsqueeze(1), xraw.unsqueeze(0), dim=-1)
                nn_ = x.norm(dim=-1)
                NS = nn_.unsqueeze(1) + nn_.unsqueeze(0)
                spn.append(spearman(offdiag(D), offdiag(NS)))
                ec = edges_at_density(S, args.density, larger_is_closer=True)
                ep = edges_at_density(D, args.density, larger_is_closer=False)
                jac.append(jaccard(ec, ep))
            m = lambda v: sum(v) / len(v)
            cells.append(f"{m(jac):.2f}/{m(spn):+.2f}")

        print(f"{mode:>19} {sum(norms)/len(norms):8.3f} {sum(cvs)/len(cvs):8.4f} | "
              + " ".join(f"{c:>11}" for c in cells))

    print()
    print("cv_norm  = std/mean of token norms = how much DEPTH signal exists at all.")
    print("jacc     = edge overlap with GLOT's cosine graph (1.00 => same graph).")
    print("spN      = Spearman(distance, sum of endpoint norms) => is depth being used?")


if __name__ == "__main__":
    main()
