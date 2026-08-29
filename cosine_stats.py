"""Token-cosine distribution and the edge density GLOT's tau=0.6 actually yields.

WHY
---
GLOT thresholds edges at an ABSOLUTE cosine tau (default 0.6 in our configs).
That number is implicitly calibrated on BERT. If another backbone's token-cosine
distribution sits elsewhere, the same tau produces a near-empty or near-complete
graph -- and this project has already been burned once by exactly that confound
(absolute tau=0.4 gave density 0.96 on the stress data while a quantile rule gave
0.025, and the "win" turned out to be sparsity, not geometry).

ModernBERT layer 4 is well conditioned (mean/median norm 1.01, max/median 1.26)
yet still scores 0.13 MCC vs BERT's 0.30, so feature SCALE cannot explain it.
Edge density is the next suspect, and unlike scale it is trivially measurable.

Prints, per layer: the cosine quantiles and the density that tau=0.6 gives.
Density near 0 or near 1 means the graph carries no information and the backbone
must be run with --tau_quantile (density-matched) instead of an absolute --tau.
"""
import argparse

import torch

SENTENCES = [
    "the cat that the dog chased ran away",
    "colorless green ideas sleep furiously",
    "the man who the woman that the child saw knew left",
    "she gave the book to the student who needed it most",
    "although it rained the match continued until the light failed",
    "john said that mary believed that the report was accurate",
]


def cosine_profile(model_name, layers, tau=0.6, max_length=64):
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()

    enc = tok(SENTENCES, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt")
    with torch.no_grad():
        out = mdl(**enc, output_hidden_states=True)
    hs, mask = out.hidden_states, enc["attention_mask"].bool()

    n_layers = getattr(cfg, "num_hidden_layers", len(hs) - 1)
    sel = [l for l in layers if 0 <= l <= n_layers] or [n_layers]

    print(f"\n=== {model_name} ===")
    print(f"{'L':>3} {'cos p10':>9} {'cos p50':>9} {'cos p90':>9} "
          f"{'density@tau=%.2f' % tau:>17}  verdict")
    for li in sel:
        dens, qs = [], []
        for b in range(hs[li].size(0)):
            x = hs[li][b][mask[b]].float()
            if x.size(0) < 4:
                continue
            xn = torch.nn.functional.normalize(x, dim=-1)
            cos = (xn @ xn.t()).clamp(-1, 1)
            n = cos.size(0)
            off = ~torch.eye(n, dtype=torch.bool)
            v = cos[off]
            qs.append(torch.quantile(v, torch.tensor([0.10, 0.50, 0.90])))
            dens.append((v > tau).float().mean())
        q = torch.stack(qs).mean(0)
        d = torch.stack(dens).mean().item()
        if d < 0.02:
            verdict = "NEAR-EMPTY graph -- tau is mis-calibrated"
        elif d > 0.90:
            verdict = "NEAR-COMPLETE graph -- tau is mis-calibrated"
        else:
            verdict = "ok"
        print(f"{li:>3} {q[0]:>9.3f} {q[1]:>9.3f} {q[2]:>9.3f} {d:>17.4f}  {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["bert-base-uncased", "answerdotai/ModernBERT-base"])
    p.add_argument("--layers", nargs="+", type=int, default=[4, 12, 22])
    p.add_argument("--tau", type=float, default=0.6)
    a = p.parse_args()
    for m in a.models:
        try:
            cosine_profile(m, a.layers, a.tau)
        except Exception as e:
            print(f"{m} FAILED: {type(e).__name__}: {e}")
    print()
    print("If density is degenerate, the backbone must be run with")
    print("--tau_quantile (keeps a fixed FRACTION of pairs) instead of --tau.")
