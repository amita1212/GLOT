"""
Backbone / layer screening for HyperGLOT: WHERE is there hyperbolic structure?

Motivation
----------
Every negative result so far was obtained on the LAST layer of bert-base, which
is the layer GLOT uses. But hyperbolic geometry only pays off when the data is
approximately TREE-LIKE, and we never measured whether it is. Two cheap
observations reframe the whole search:

  * Hewitt & Manning (2019) show syntax trees are best recoverable from BERT's
    MIDDLE layers (~7-9), not the final one. If tree structure lives at layer 8,
    Stage A has been building hyperbolic graphs on the single layer where the
    signal is weakest.
  * The token norm is the ball's "depth" coordinate. On bert-base last layer its
    coefficient of variation is 0.057 -- essentially constant, so the ball has
    nothing to represent. Other backbones may differ a lot.

Metrics reported (all on REAL sentences, forward passes only -- no training)
---------------------------------------------------------------------------
delta_rel   Gromov 4-point delta-hyperbolicity, normalised by the diameter:
                delta_rel = 2*delta / diam
            0.0 = a perfect tree (hyperbolic space is the ideal host)
            1.0 = maximally NON tree-like (hyperbolic cannot help)
            This is the standard measure used to justify hyperbolic embeddings
            (Nickel & Kiela 2017; Chami et al. 2019). It is the single most
            informative number here: if delta_rel is high everywhere, the
            negative result is EXPLAINED rather than merely observed.
cv_raw      std/mean of token norms = how much depth signal exists as-is.
cv_ctr      same after per-sentence centring (re-rooting the ball).
jacc        edge-set overlap of the density-matched Poincare graph with GLOT's
            cosine graph. 1.0 means Stage A cannot differ from the baseline.
sp_norm     Spearman(Poincare distance, sum of endpoint norms): is the depth
            coordinate actually being used by the geometry?

Usage:
    python screen_backbones.py --models bert-base-uncased roberta-base
    python screen_backbones.py --models bert-base-uncased --layers all
"""

from __future__ import annotations

import argparse
import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Gromov delta-hyperbolicity (4-point condition)
# --------------------------------------------------------------------------- #
def delta_hyperbolicity(D: torch.Tensor) -> tuple[float, float]:
    """Return (delta, diameter) for a finite metric space given its distance matrix.

    Uses the Gromov-product formulation with the max-min matrix product, which is
    O(n^3) instead of the O(n^4) of the naive four-point scan:

        G_w(x, y) = 1/2 ( d(x,w) + d(y,w) - d(x,y) )
        delta     = max_{x,y} ( (G o G)(x,y) - G(x,y) ),
        (A o B)(x,y) = max_z min( A(x,z), B(z,y) )

    The result is independent of the base point w up to a factor of 2, and the
    base point is taken to be node 0 as is conventional.
    """
    n = D.size(0)
    if n < 4:
        return float("nan"), float("nan")
    D = D.double()
    row = D[0]
    G = 0.5 * (row.unsqueeze(0) + row.unsqueeze(1) - D)   # Gromov products
    # max-min product: maxz min(G[x,z], G[z,y])
    A = torch.minimum(G.unsqueeze(-1), G.unsqueeze(0))     # (x, z, y)
    GG = A.max(dim=1).values                               # (x, y)
    delta = float((GG - G).max())
    diam = float(D.max())
    return delta, diam


def rel_delta(D: torch.Tensor) -> float:
    d, diam = delta_hyperbolicity(D)
    if diam is None or diam <= 0 or d != d:
        return float("nan")
    return 2.0 * d / diam


def angular_rel_delta(x: torch.Tensor) -> float:
    """delta-hyperbolicity of the ANGULAR metric -- immune to norm outliers.

    WHY THIS EXISTS
    ---------------
    A star graph is a tree, so delta = 0. If a handful of tokens carry enormous
    norms (the well-known 'rogue dimension' phenomenon in modern encoders --
    measured mean||x|| ~ 2700 for ModernBERT and ~50000 for Flan-T5, with
    cv_raw up to 3.3), those tokens alone set the diameter, every other pair
    looks equidistant, and the Euclidean delta_rel collapses toward 0 for a
    completely degenerate reason. That is NOT linguistic hierarchy.

    The angular distance d(x, y) = arccos(cos(x, y)) throws the magnitudes away,
    so if a model is still tree-like under this metric the structure is real.
    Comparing the two columns separates genuine hierarchy from outlier artefacts.
    """
    xn = F.normalize(x, dim=-1)
    cos = (xn @ xn.t()).clamp(-1.0, 1.0)
    D = torch.arccos(cos)
    D = 0.5 * (D + D.t())
    D.fill_diagonal_(0.0)
    return rel_delta(D)


def norm_outlier_ratio(x: torch.Tensor) -> float:
    """max||x|| / median||x||. >~3 means a few tokens dominate the geometry."""
    n = x.norm(dim=-1)
    return float(n.max() / n.median().clamp_min(1e-9))


# --------------------------------------------------------------------------- #
def _rank(v):
    order = v.argsort()
    r = torch.empty_like(order, dtype=torch.float64)
    r[order] = torch.arange(v.numel(), dtype=torch.float64)
    return r


def spearman(a, b):
    ra, rb = _rank(a) - _rank(a).mean(), _rank(b) - _rank(b).mean()
    return float((ra @ rb) / (ra.norm() * rb.norm()).clamp_min(1e-12))


def offdiag(M):
    n = M.size(0)
    return M[~torch.eye(n, dtype=torch.bool, device=M.device)]


def jaccard_vs_cosine(x_cond, x_raw, c, density=0.10):
    import geoopt
    ball = geoopt.PoincareBall(c=c)
    p = ball.projx(ball.expmap0(x_cond))
    D = ball.dist(p.unsqueeze(1), p.unsqueeze(0), dim=-1)
    D = 0.5 * (D + D.t())
    D.fill_diagonal_(0.0)
    S = F.cosine_similarity(x_raw.unsqueeze(1), x_raw.unsqueeze(0), dim=-1)
    d, s = offdiag(D), offdiag(S)
    thr_d = torch.quantile(d.float(), density)
    thr_s = torch.quantile(s.float(), 1.0 - density)
    ed, es = d < thr_d, s > thr_s
    inter = (ed & es).sum().item()
    union = (ed | es).sum().item()
    nn_ = x_cond.norm(dim=-1)
    NS = offdiag(nn_.unsqueeze(1) + nn_.unsqueeze(0))
    return (inter / union if union else 1.0), spearman(d, NS), D


SENTENCES = [
    "The keys to the cabinet that the janitor had carefully locked were missing.",
    "Although the report which the committee commissioned was late, nobody complained.",
    "The scientist whose experiment the journal rejected published it elsewhere.",
    "Every student who the teacher praised passed the difficult examination easily.",
    "The book that the professor recommended to the class was surprisingly short.",
    "While the engineers debugged the system, the manager quietly rewrote the plan.",
    "The dog that chased the cat that caught the mouse ran into the garden.",
    "Nobody believed the rumour that the company which acquired them would relocate.",
    "The letter the lawyer sent the client explained the settlement in detail.",
    "Because the bridge had collapsed, the trucks that carried the grain turned back.",
    "The painting the curator hung in the hall attracted considerable attention.",
    "Whoever finishes the assignment that the instructor assigned may leave early.",
]


def analyse(model_name, layers, device):
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    # Decoder-only models (GPT-2, Qwen, ...) ship without a pad token, which
    # makes batched padding raise. Reuse EOS, exactly as the training code does.
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})

    cfg = AutoConfig.from_pretrained(model_name)
    is_enc_dec = bool(getattr(cfg, "is_encoder_decoder", False))
    if is_enc_dec:
        # For a true encoder-decoder (T5, BART) the token representations GLOT
        # would pool are the ENCODER's, so screen the encoder stack only.
        from transformers import AutoModel as _AM
        mdl = _AM.from_pretrained(model_name).encoder
        mdl.config.output_hidden_states = True
    else:
        mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    mdl = mdl.to(device).eval()

    enc = tok(SENTENCES, padding=True, truncation=True, max_length=64,
              return_tensors="pt").to(device)
    enc.pop("token_type_ids", None) if not hasattr(mdl, "embeddings") else None
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True)
    hs = out.hidden_states                      # tuple: (emb, layer1..layerN)
    n_layers = len(hs) - 1
    if layers == "all":
        sel = list(range(1, n_layers + 1))
    elif layers == "last":
        sel = [n_layers]
    else:
        sel = [int(s) for s in layers.split(",") if 1 <= int(s) <= n_layers]

    mask = enc["attention_mask"].bool().cpu()
    rows = []
    for li in sel:
        H = hs[li].float().cpu()
        cvs_raw, cvs_ctr, deltas, jaccs, spns, norms = [], [], [], [], [], []
        dangs, outs = [], []
        for b in range(H.size(0)):
            x = H[b][mask[b]]
            if x.size(0) < 6:
                continue
            nrm = x.norm(dim=-1)
            norms.append(nrm.mean().item())
            cvs_raw.append((nrm.std() / nrm.mean().clamp_min(1e-9)).item())
            z = x - x.mean(0, keepdim=True)
            zn = z.norm(dim=-1)
            cvs_ctr.append((zn.std() / zn.mean().clamp_min(1e-9)).item())
            # delta-hyperbolicity of the EUCLIDEAN token metric (what the graph
            # is ultimately built from). Scale-free, so conditioning is irrelevant.
            deltas.append(rel_delta(torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)))
            dangs.append(angular_rel_delta(x))
            outs.append(norm_outlier_ratio(x))
            zc = z / zn.mean().clamp_min(1e-6)         # center_unit conditioning
            j, spn, _ = jaccard_vs_cosine(zc, x, c=1.0)
            jaccs.append(j)
            spns.append(spn)
        m = lambda v: sum(v) / len(v) if v else float("nan")
        rows.append((li, m(norms), m(cvs_raw), m(cvs_ctr), m(deltas), m(jaccs),
                     m(spns), m(dangs), m(outs)))
    del mdl
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert-base-uncased"])
    ap.add_argument("--layers", default="last", help="'last', 'all', or '4,8,12'")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hdr = (f"{'model':<30} {'L':>3} {'mean|x|':>9} {'cv_raw':>7} {'cv_ctr':>7} "
           f"{'d_eucl':>7} {'d_ang':>7} {'out':>6} {'jacc':>6} {'sp_norm':>8}")
    print(hdr)
    print("-" * len(hdr))
    best = []
    for mn in args.models:
        try:
            for (li, nrm, cvr, cvc, dl, jc, sp, dang, out) in analyse(mn, args.layers, device):
                print(f"{mn[:30]:<30} {li:>3} {nrm:>9.1f} {cvr:>7.3f} {cvc:>7.3f} "
                      f"{dl:>7.4f} {dang:>7.4f} {out:>6.1f} {jc:>6.3f} {sp:>8.3f}")
                best.append((dang, dl, cvc, out, mn, li))
        except Exception as e:
            print(f"{mn[:30]:<30}  FAILED: {type(e).__name__}: {e}")

    print()
    print("d_eucl = delta-hyperbolicity of the Euclidean token metric.")
    print("d_ang  = SAME on the ANGULAR metric, which discards magnitudes.")
    print("out    = max||x|| / median||x||.")
    print()
    print("READ THESE TOGETHER. A star graph is a tree, so a few outlier tokens")
    print("drive d_eucl to ~0 for a DEGENERATE reason. If d_eucl << d_ang and")
    print("out is large, the 'tree-likeness' is a rogue-dimension artefact, not")
    print("linguistic hierarchy. Only trust a layer where d_ang is ALSO low.")
    if best:
        best.sort()
        print()
        print("MOST TREE-LIKE BY THE ROBUST (ANGULAR) MEASURE:")
        for dang, dl, cvc, out, mn, li in best[:6]:
            flag = "  <-- d_eucl is an outlier artefact" if dl < 0.3 * dang else ""
            print(f"   d_ang={dang:.4f}  d_eucl={dl:.4f}  cv_ctr={cvc:.3f}  "
                  f"out={out:.1f}  {mn} L{li}{flag}")


if __name__ == "__main__":
    main()
