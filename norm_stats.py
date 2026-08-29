"""Token-norm distribution per backbone/layer: mean vs median vs tail.

WHY THIS MATTERS
----------------
`--input_scale_norm=rms` rescales so the MEAN token norm hits a target. That is
the right statistic only if the norm distribution is well behaved. ModernBERT
has massive activations (max/median = 84, cv = 3.3), so its mean is dragged up
by a handful of attention-sink tokens. Rescaling the mean to 15 would then put
the TYPICAL token far below 15 while the sinks sit far above it -- the bulk of
the distribution would still be mis-scaled, just in the other direction.

This prints mean, median, p90, p99, max and the ratios that decide which
statistic to normalise by:
    mean/median  ~1  -> distribution is clean, mean-based scaling is fine
    mean/median >>1  -> mean is outlier-dominated, use median-based scaling
Run on CPU; it is forward passes only.
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


def stats(model_name, layers, max_length=64):
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
    hs = out.hidden_states
    mask = enc["attention_mask"].bool()

    n_layers = getattr(cfg, "num_hidden_layers", len(hs) - 1)
    sel = [l for l in layers if 0 <= l <= n_layers] or [n_layers]

    print(f"\n=== {model_name} ({n_layers} layers) ===")
    print(f"{'L':>3} {'mean':>10} {'median':>10} {'p90':>10} {'p99':>10} "
          f"{'max':>10} {'mean/med':>9} {'max/med':>9}")
    for li in sel:
        x = hs[li][mask].float()               # (n_valid_tokens, d)
        n = x.norm(dim=-1)
        med = n.median()
        q = torch.quantile(n, torch.tensor([0.90, 0.99]))
        print(f"{li:>3} {n.mean():>10.2f} {med:>10.2f} {q[0]:>10.2f} "
              f"{q[1]:>10.2f} {n.max():>10.2f} "
              f"{(n.mean() / med):>9.2f} {(n.max() / med):>9.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["bert-base-uncased", "answerdotai/ModernBERT-base"])
    p.add_argument("--layers", nargs="+", type=int, default=[4, 12, 22])
    a = p.parse_args()
    for m in a.models:
        try:
            stats(m, a.layers)
        except Exception as e:
            print(f"{m} FAILED: {type(e).__name__}: {e}")
    print()
    print("mean/median ~1 -> mean-based rescaling is fine (use rms).")
    print("mean/median >>1 -> the mean is outlier-dominated; normalise by the")
    print("                   MEDIAN so typical tokens land on the target scale.")
