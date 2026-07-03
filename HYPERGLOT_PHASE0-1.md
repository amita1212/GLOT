# HyperGLOT — Phase 0 & 1 Run Guide

This folder is a clone of [GLOT](https://github.com/ipsitmantri/GLOT) plus the
**Stage A** (hyperbolic graph construction) additions for the HyperGLOT project.
See `../Hyperbolic-GLOT-Research-Report.tex` for the full plan.

## What was added (Stage A)

| File | Purpose |
|------|---------|
| `hyperbolic_graph.py` | Stand-alone Stage A module: builds the token graph by **Poincaré (hyperbolic) distance** instead of cosine. Drop-in for GLOT's `build_pyg_graphs`. |
| `verify_stage_a.py` | Verification suite (no GPU needed). Proves correctness, the *kNN ⇒ cosine* equivalence, magnitude-sensitivity, and that the Stage A graph flows through the **real** GLOT GNN+readout. |
| `main.py` (patched) | New, **backward-compatible** flags: `--graph_metric {cosine,poincare}`, `--curvature`, `--rho`, `--knn_k`, `--feature_norm`, and `--graph_adj` now also accepts `knn`. Defaults reproduce stock GLOT exactly. |

The only conceptual change vs GLOT:

```
GLOT (Euclidean):     edge(i,j) if  cosine(x_i, x_j)              >  tau
Stage A (hyperbolic): edge(i,j) if  d_Poincare(exp0(x_i),exp0(x_j)) <  rho
```

Everything downstream (Token-GNN, attention readout, classifier) is untouched.

## Environment

A CPU-only virtual environment is enough for **verification** and for Phase 0/1
on **BERT** with cached hidden states. It was created at
`%USERPROFILE%\hyperglot-venv` with:

```powershell
python -m venv "$env:USERPROFILE\hyperglot-venv"
& "$env:USERPROFILE\hyperglot-venv\Scripts\python.exe" -m pip install `
    torch --index-url https://download.pytorch.org/whl/cpu
& "$env:USERPROFILE\hyperglot-venv\Scripts\python.exe" -m pip install `
    geoopt torch-geometric numpy scikit-learn transformers datasets sentencepiece
```

> For **full training runs** (Phase 0 reproduction, decoder backbones) install the
> pinned stack in `requirements.txt` (needs `mteb`, `wandb`, `peft`,
> `torch-scatter`, and a GPU). The frozen-backbone design means the LLM is run
> **once** to cache hidden states; everything after that trains cheaply.

## Step 1 — Verify Stage A (do this first, ~10 s, CPU)

```powershell
cd GLOT
& "$env:USERPROFILE\hyperglot-venv\Scripts\python.exe" verify_stage_a.py
```

Expected: checks `[1]`–`[5]` print `OK`. Highlights:
- `[2]` hyperbolic-kNN == cosine-kNN for several curvatures → **GLOT is a special
  case of Stage A**.
- `[3]` on magnitude-varying features the edge sets differ → Stage A genuinely
  uses the hierarchy signal cosine discards.
- `[4]` the hyperbolic graph runs through the **real** GLOT head → drop-in works.

## Step 2 — Phase 0: reproduce GLOT (the control)

Needs the full `requirements.txt` env + a HuggingFace token in `main.py`.
Cheap, hierarchy-heavy GLUE tasks on frozen BERT:

```powershell
foreach ($task in @("cola","stsb","rte")) {
  python main.py --model_name_or_path="bert-base-uncased" --task=$task `
    --pooling_method=glot --gnn_type=gat --num_layers=2 --jk_mode=cat `
    --gat_hidden_dim=256 --scorer_hidden=128 --proj_dim=256 `
    --graph_adj=threshold --tau=0.6 `
    --max_length=128 --epochs=3 --batch_size=32 --lr=2e-4 --seed=42 `
    --precompute_hidden_states=1 --finetune_backbone=0
}
```

Deliverable: a table matching GLOT's reported CoLA/STS-B/RTE numbers within noise.

## Step 3 — Phase 1: Stage A sweep (cosine vs hyperbolic)

Same command, but switch the graph metric and sweep `rho` (the hyperbolic analogue
of `tau`) and curvature `c`. Everything else is identical, so any change is
attributable to graph construction alone.

```powershell
foreach ($task in @("cola","stsb","rte")) {
  foreach ($c in @(0.5,1.0,2.0)) {
    foreach ($rho in @(1.0,2.0,3.0)) {
      python main.py --model_name_or_path="bert-base-uncased" --task=$task `
        --pooling_method=glot --gnn_type=gat --num_layers=2 --jk_mode=cat `
        --gat_hidden_dim=256 --scorer_hidden=128 --proj_dim=256 `
        --graph_adj=threshold --graph_metric=poincare --curvature=$c --rho=$rho `
        --max_length=128 --epochs=3 --batch_size=32 --lr=2e-4 --seed=42 `
        --precompute_hidden_states=1 --finetune_backbone=0
    }
  }
}
```

Tips:
- Start with `--curvature=1.0` and sweep `--rho` first; `rho` controls sparsity the
  way `tau` does. (Toy models can need larger `rho` before edges appear — Step 1's
  tiny-BERT shows `edges=0` at `rho=2.0`, which is expected.)
- Optionally try `--graph_adj=knn --knn_k=8` for a sparsity-controlled variant, and
  `--feature_norm=1` to test the normalized regime.

**Go/no-go signal:** if `graph_metric=poincare` beats `cosine` on CoLA/STS-B/RTE
(or on the negation stress test in `diagnostic_stress_test.py`), Stage A validates
the hypothesis and you proceed to Stage B/C. If not, you've learned it cheaply.
