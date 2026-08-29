"""Smoke-test that a backbone loads and forwards through GLOT's own loader.

WHY: queuing a multi-hour chain behind a running campaign only to have the
backbone fail to load wastes the whole GPU slot. This checks the three things
that actually break for non-BERT models, using main.py's REAL code path:

  1. tokenizer construction  -- upstream hardcodes use_fast=False, which
     hard-fails for fast-only tokenizers (ModernBERT BPE, DeBERTa-v3).
  2. extra tokenizer outputs -- BERT emits token_type_ids, most newer models
     do not accept them; a mismatch throws inside forward().
  3. hidden_layer indexing   -- confirms hidden_states has enough entries and
     that layer L really differs from the last layer (a silently clamped index
     would make every layer identical and every layer comparison meaningless).

Runs on CPU so it can be executed while a GPU job holds the card.
"""
import argparse
import sys

import torch

import main as glot


def check(model_name, layers, max_length=64):
    print(f"\n=== {model_name} ===")
    try:
        backbone, _ = glot.load_backbone(model_name, max_length=max_length,
                                         task="glue", hidden_layer=-1)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        return False

    tok = backbone.tokenizer
    print(f"  tokenizer   : {type(tok).__name__} (fast={tok.is_fast})")
    print(f"  n_layers    : {backbone.config.num_hidden_layers}")
    print(f"  hidden_size : {backbone.config.hidden_size}")

    sents = ["the cat that the dog chased ran away",
             "colorless green ideas sleep furiously"]
    batch = tok(sents, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt")
    print(f"  tok keys    : {sorted(batch.keys())}")

    backbone.model.eval()
    ref = None
    ok = True
    for L in layers:
        backbone.hidden_layer = L
        try:
            with torch.no_grad():
                hid, _ = glot.forward_hidden(backbone, dict(batch))
        except Exception as e:
            print(f"  L{L:<3} FORWARD FAILED: {type(e).__name__}: {e}")
            ok = False
            continue
        n = hid.norm(dim=-1)
        tag = ""
        if ref is not None:
            # If the layer index were silently clamped, every layer would be
            # bit-identical and no layer comparison could ever mean anything.
            same = torch.allclose(hid, ref, atol=1e-6)
            tag = "  <-- IDENTICAL TO PREVIOUS LAYER (index not wired!)" if same else ""
        ref = hid.clone()
        print(f"  L{L:<3} shape={tuple(hid.shape)} mean|x|={n.mean():.2f} "
              f"max/med={(n.max() / n.median()).item():.1f}{tag}")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["answerdotai/ModernBERT-base"])
    p.add_argument("--layers", nargs="+", type=int, default=[4, 8, 12, 16, 20, 22])
    a = p.parse_args()
    allok = all(check(m, a.layers) for m in a.models)
    print()
    print("ALL OK" if allok else "FAILURES ABOVE -- do not queue the chain")
    sys.exit(0 if allok else 1)
