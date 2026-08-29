#!/usr/bin/env python3
"""Measure token-cosine distribution and GLOT edge density, independent of training.

WHY THIS EXISTS
    Table `tab:density` was populated from training-time telemetry, which used a
    density whose numerator counted self-loops but whose denominator did not
    (fixed in hyperbolic_graph.py). Two different measurement paths also
    disagreed for RoBERTa (0.992 vs 1.09). Edge density is a property of
    (backbone, data, tau) alone -- it has nothing to do with training -- so it
    should be measured once, directly, by a script a reader can rerun.

    Density here = fraction of OFF-DIAGONAL ordered token pairs with
    cos(x_i, x_j) > tau, averaged over sentences. Padding is excluded via the
    attention mask. Self-loops are never counted.

    usage: measure_density.py [n_sentences] [max_length]
"""
import sys

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

MODELS = [
    ("BERT-base", "bert-base-uncased"),
    ("RoBERTa-base", "roberta-base"),
    ("ModernBERT-base", "answerdotai/ModernBERT-base"),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M"),
    ("TinyLlama-1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]
TAUS = [0.1, 0.3, 0.6]


def sentences(n):
    ds = load_dataset("nyu-mll/glue", "cola", split="validation")
    return [s for s in ds["sentence"][:n]]


@torch.no_grad()
def measure(model_id, sents, max_len, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    cos_pool, dens = [], {t: [] for t in TAUS}
    for i in range(0, len(sents), 16):
        batch = sents[i:i + 16]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        h = model(**enc).last_hidden_state          # (B, L, d)
        mask = enc["attention_mask"].bool()

        for b in range(h.size(0)):
            x = h[b][mask[b]]                       # (n, d) real tokens only
            n = x.size(0)
            if n < 2:
                continue
            xn = torch.nn.functional.normalize(x, dim=-1)
            sim = xn @ xn.t()                       # (n, n)
            off = ~torch.eye(n, dtype=torch.bool, device=sim.device)
            v = sim[off]                            # n(n-1) off-diagonal pairs
            cos_pool.append(v.float().cpu())
            for t in TAUS:
                dens[t].append((v > t).float().mean().item())

    del model
    torch.cuda.empty_cache()
    allc = torch.cat(cos_pool)
    q = torch.quantile(allc, torch.tensor([0.10, 0.50, 0.90]))
    return q.tolist(), {t: sum(v) / len(v) for t, v in dens.items()}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    # CPU by default: the GPU is busy with the campaign and this is cheap.
    device = torch.device("cpu")

    sents = sentences(n)
    print(f"\n  {n} CoLA validation sentences, max_length={max_len}, "
          f"off-diagonal pairs only, self-loops excluded\n")
    hdr = f"  {'Backbone':<18}{'p10':>8}{'p50':>8}{'p90':>8}" + \
          "".join(f"{'t=' + str(t):>9}" for t in TAUS)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for name, mid in MODELS:
        try:
            (p10, p50, p90), d = measure(mid, sents, max_len, device)
        except Exception as e:                       # noqa: BLE001
            print(f"  {name:<18} FAILED: {type(e).__name__}: {e}")
            continue
        line = f"  {name:<18}{p10:>8.3f}{p50:>8.3f}{p90:>8.3f}"
        line += "".join(f"{d[t]:>9.3f}" for t in TAUS)
        print(line, flush=True)
        rows.append((name, p10, p50, p90, d))

    print("\n  LaTeX rows for tab:density\n")
    for name, p10, p50, p90, d in rows:
        cells = " & ".join(f"{d[t]:.3f}" for t in TAUS)
        print(f"    {name:<18}& {p10:.3f} & {p50:.3f} & {p90:.3f} & {cells} \\\\")
    print()


if __name__ == "__main__":
    main()
