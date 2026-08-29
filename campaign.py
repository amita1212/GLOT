#!/usr/bin/env python
"""
Full HyperGLOT campaign: every arm, equal-budget random search, GLUE + stress.

Why random search with an equal budget
--------------------------------------
The arms do not share a hyper-parameter space -- Stage A tunes graph density and
curvature, Stage B tunes readout clipping, Stage C tunes the GNN entry scale --
so a grid cannot be "the same size" for all of them in any meaningful way, and
the earlier ablation's asymmetry (baseline pinned at one tau while hyperbolic
arms got their own knobs) is exactly how the original inflated result happened.
Random search with an identical number of trials per arm is the standard fair
protocol for heterogeneous spaces (Bergstra & Bengio, JMLR 2012) and is what is
used here. The sampler is seeded per arm, so the campaign is reproducible and
fully resumable after a spot preemption.

Selection bias is real and is corrected for downstream: with ~12 trials and a
CoLA seed-std of 0.81, the expected maximum is inflated by roughly +1.3 MCC even
if every arm is identical to the baseline. That is why the tuning stage and the
reporting stage use DIFFERENT seeds, and why `analyze_geometry.py` quotes deltas
in units of the noise floor.

Arms
----
  baseline           original GLOT
  A                  hyperbolic token graph (Stage A)
  B                  hyperbolic gyro-midpoint readout (Stage B)
  C                  hyperbolic Token-GNN (Stage C)
  AB, AC, BC, ABC    all combinations
  W_hyp              GLOT's cosine topology + hyperbolic edge weights
  W_depth            GLOT's cosine topology + signed hyperbolic depth gap
  AW                 Stage A topology + depth-gap edge weights

Targets
-------
  --target glue    -> main.py on a GLUE task
  --target stress  -> diagnostic_stress_test.py (relational needle-in-haystack)
"""

from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_runner import ResultsCSV, PAPER_METRIC, PAPER_DEFAULTS  # noqa: E402

MAIN = os.path.join(HERE, "hyperglot", "main.py")
STRESS = os.path.join(HERE, "hyperglot", "diagnostic_stress_test.py")
if not os.path.exists(MAIN):
    MAIN = os.path.join(HERE, "main.py")
    STRESS = os.path.join(HERE, "diagnostic_stress_test.py")

RE_STRESS = re.compile(r"Eval Accuracy=([\d.]+)")
RE_STS = re.compile(r"epoch (\d+) MSE [\d.]+ Spearman ([-\d.]+) Pearson ([-\d.]+)")
RE_PAIR = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) f1 ([-\d.]+)")
RE_SINGLE = re.compile(r"epoch (\d+) loss [\d.]+ acc ([-\d.]+) mcc ([-\d.]+)")
RE_DENSITY = re.compile(r"\[graph\] density=([\d.]+) empty_frac=([\d.]+)")


# --------------------------------------------------------------------------- #
# Search spaces. Each entry maps a flag to the list of values sampled from.
# Values that are FIXED for an arm are given as a single-element list, so the
# effective number of free choices differs but the TRIAL BUDGET does not.
# --------------------------------------------------------------------------- #
# Every arm draws its sparsity from the SAME grid, expressed as the fraction of
# token pairs kept. Cosine arms use `tau_quantile`, Poincare arms use
# `rho_quantile`; both mean "keep the closest q of pairs". This is what makes the
# comparison about GEOMETRY rather than about sparsity, and it matters a lot: on
# the stress data an absolute tau=0.4 already gives density 0.96 while a Poincare
# arm at rho_quantile=0.05 gives 0.058 -- and the sparse arm wins by ~3 points
# regardless of its metric, purely because a near-complete token graph is noise.
DENSITY_Q = [0.025, 0.05, 0.10, 0.20, 0.38]
# r_eff = sqrt(c)*mean||x||; with a *_unit feature_mode mean||x||=1 so c=r_eff^2.
R_EFF = [0.5, 1.0, 1.5, 2.0, 3.0]
FEATURE_MODES = ["center_unit", "cls_root_unit", "unit", "center"]

STAGE_A = {
    "graph_metric": ["poincare"],
    "rho_quantile": DENSITY_Q,
    "graph_curvature": [r * r for r in R_EFF],
    "feature_mode": FEATURE_MODES,
}
STAGE_B = {
    "hyperbolic_readout": [1],
    "readout_clip": [0.0, 0.3, 0.7, 1.0],
    "readout_scale": [0, 1],
    "learnable_curvature": [0, 1],
    "curvature": [0.25, 1.0, 4.0],
}
STAGE_C = {
    "hyperbolic_gnn": [1],
    "hyp_gnn_type": ["gcn", "gat"],
    "gnn_input_clip": [0.0, 0.3, 0.7, 1.0],
    "gnn_input_scale": [0, 1],
    "curvature": [0.25, 1.0, 4.0],
}
COSINE = {"graph_metric": ["cosine"], "tau_quantile": DENSITY_Q}

# --------------------------------------------------------------------------- #
# THE BUDGET ASYMMETRY, AND HOW TO REMOVE IT
# --------------------------------------------------------------------------- #
# `--trials N` gives every arm the same number of DRAWS, but not the same number
# of distinct CONFIGS. Stage A spans density x curvature x feature_mode = 100
# points, while the cosine baseline spans only len(DENSITY_Q) = 5. So at N=10 the
# hyperbolic arms draw 10 fresh configs and the baseline EXHAUSTS its space at 5.
# Reporting best-of-10 against best-of-5 flatters the larger space even when the
# two are identical in truth: for scores with sd ~1.7 (measured on CoLA L8) the
# expected maxima differ by 1.7 * (1.539 - 1.163) ~ 0.64 MCC from budget alone,
# which is most of the gap actually observed. That is a lottery-ticket effect,
# not a geometry effect.
#
# DENSITY_Q_FINE refines the SAME interval to 10 points so the baseline can spend
# a full 10 draws on distinct configs. It is opt-in via --fine_baseline precisely
# so previously recorded runs stay comparable: the coarse grid is a subset, so no
# earlier result is invalidated, and run_key dedup means already-completed points
# are reused rather than recomputed.
DENSITY_Q_FINE = [0.0125, 0.025, 0.0375, 0.05, 0.075,
                  0.10, 0.15, 0.20, 0.28, 0.38]


def _merge(*spaces):
    out = {}
    for s in spaces:
        out.update(s)
    return out


# --------------------------------------------------------------------------- #
# THE OPTIMIZER / ARCHITECTURE AXIS  (--wide)
# --------------------------------------------------------------------------- #
# Until now every arm searched ONLY graph-construction knobs (tau_quantile,
# rho_quantile, curvature, feature_mode, ...) while lr, weight_decay, depth,
# width, jumping-knowledge and projection width stayed pinned at PAPER_DEFAULTS.
# Those defaults are the ones the ORIGINAL authors selected -- for a Euclidean
# pooler. That is a confound, and it points the wrong way for us:
#
#   Stage B replaces the readout with a Frechet-style mean and Stage C replaces
#   message passing with Mobius operations. Non-Euclidean / structured layers are
#   known to underperform badly when trained at learning rates tuned for a dense
#   Euclidean layer, and to need structure-aware LR and init before any benefit
#   appears. Our sharpest "significant" result is exactly the shape that failure
#   produces: AB -3.56 and ABC -4.51 on TinyLlama with 0/15 seeds positive, i.e.
#   uniformly bad in every single seed.
#
# So "hyperbolic readout and GNN are harmful" is not yet supportable. What the
# data currently supports is "harmful AT THE BASELINE'S LEARNING RATE". WIDE adds
# the paper's own Table 6 optimizer/architecture grid so that claim can be
# settled either way.
#
# WIDE is applied to EVERY arm including the cosine baseline. Giving the
# hyperbolic arms a richer space than the baseline would manufacture exactly the
# selection bias this campaign exists to avoid.
#
# jk_mode: the paper's Table 6 states {cat, max, mean, none}, but the released
# code declares choices=["cat","lstm","max"] and `lstm` accepts then crashes at
# runtime. The reproducible space is therefore {cat, max}; `mean` and `none`
# cannot be run at all. Reported as a reproducibility gap rather than silently
# substituted.
WIDE = {
    "lr": [1e-3, 2e-4, 2e-5],
    "weight_decay": [0.0, 1e-5, 5e-5],
    "num_layers": [2, 4],
    "gat_hidden_dim": [64, 128, 256],
    "jk_mode": ["cat", "max"],
    "proj_dim": [128, 256, 512],
    "scorer_hidden": [128, 256],
}
# 3*3*2*3*2*3*2 = 648 optimizer/architecture points, multiplied by each arm's
# own graph grid. Random search over this needs far more than the 10 trials used
# for the graph-only space; --wide therefore also raises the default trial count.


# --------------------------------------------------------------------------- #
# WORD-ORDER STRUCTURE
# --------------------------------------------------------------------------- #
# Every arm above changes the GEOMETRY of the token graph. None of them changes
# what INFORMATION it carries, and all of them leave it permutation-invariant:
# edges depend only on feature similarity, so permuting the tokens permutes the
# graph and leaves the pooled embedding unchanged.
#
# That predicts the pattern actually observed. Stage A wins on STS-B (+0.223
# Spearman, 15/15 seeds), whose objective IS geometric -- Spearman on cosine
# similarity between pooled vectors. It is flat on CoLA, RTE and MRPC, which are
# classification tasks where a linear head can absorb a differently-shaped
# embedding. CoLA is the sharpest case: it scores linguistic ACCEPTABILITY,
# which is very largely word order, and an order-invariant structure cannot
# represent it at all.
#
# POSITIONAL adds edges between tokens within w sequence positions, which is the
# only thing here that makes the topology order-aware. POS_ONLY is the required
# ablation: if POSITIONAL wins, POS_ONLY says whether the gain is word order or
# merely a denser graph.
POSITIONAL = {"graph_metric": ["cosine"], "tau_quantile": DENSITY_Q,
              "positional_window": [1, 2, 3, 5]}
POS_ONLY = {"graph_metric": ["cosine"], "tau": [0.999],
            "positional_window": [1, 2, 3, 5], "positional_only": [1]}
# kNN fixes each token's DEGREE instead of a global similarity threshold, so it
# is robust to the per-sentence anisotropy that makes a single tau mis-calibrated
# (the same effect that gave ModernBERT a 0.996-density graph). Implemented in
# hyperbolic_graph.py but never swept. NOTE the CLI flag is --graph_adj, not
# --adjacency; campaign.py forwards dict keys verbatim as --key=value, so the
# key here must match main.py's argparse name exactly or the run dies at launch.
KNN = {"graph_metric": ["cosine"], "graph_adj": ["knn"], "knn_k": [2, 4, 8, 16]}


ARMS = {
    "baseline": COSINE,
    # Essential control: tau=0.999 keeps only self-loops, so the GNN runs with
    # NO token graph at all. A previous "Stage A wins the stress test" headline
    # turned out to be "removing the graph wins" once the empty-graph bug was
    # found, so no arm may be credited with a win until it also beats this.
    "no_graph": {"graph_metric": ["cosine"], "tau": [0.999]},
    # --- reproduce-the-paper vs calibrate-the-paper --------------------------
    # PAPER_TAU pins tau=0.6, exactly as published. On BERT that is a sensible
    # 0.149 density; on RoBERTa it is 0.992 -- a near-complete graph. In fact
    # EVERY tau in the paper's grid {0.1, 0.3, 0.6} gives RoBERTa >= 0.99,
    # because its 10th-percentile token cosine (0.701) already exceeds the
    # largest tau searched. DENSITY_FIX keeps the identical cosine metric but
    # picks the threshold by QUANTILE, so sparsity is comparable across
    # backbones. The pair separates "the published setting" from "the published
    # setting, calibrated", which is the only way to tell whether GLOT's
    # RoBERTa numbers reflect relational structure or just a complete graph.
    "paper_tau":   {"graph_metric": ["cosine"], "tau": [0.6]},
    "density_fix": {"graph_metric": ["cosine"], "tau_quantile": [0.05, 0.10, 0.15]},
    "A":        _merge(STAGE_A),
    # --- word-order / degree structure (see the note above) ------------------
    "POS":      _merge(POSITIONAL),
    "POS_ONLY": _merge(POS_ONLY),
    "A_POS":    _merge(STAGE_A, {"positional_window": [1, 2, 3, 5]}),
    "KNN":      _merge(KNN),
    "B":        _merge(COSINE, STAGE_B),
    "C":        _merge(COSINE, STAGE_C),
    "AB":       _merge(STAGE_A, STAGE_B),
    "AC":       _merge(STAGE_A, STAGE_C),
    "BC":       _merge(COSINE, STAGE_B, STAGE_C),
    "ABC":      _merge(STAGE_A, STAGE_B, STAGE_C),
    "W_hyp":    _merge(COSINE, {"edge_weight_mode": ["hyp", "hyp_z"],
                                "graph_curvature": [r * r for r in R_EFF],
                                "feature_mode": FEATURE_MODES,
                                "edge_temp": [0.25, 0.5, 1.0]}),
    "W_depth":  _merge(COSINE, {"edge_weight_mode": ["depth"],
                                "feature_mode": FEATURE_MODES}),
    "AW":       _merge(STAGE_A, {"edge_weight_mode": ["depth", "hyp_z"]}),
}

# `curvature` appears in both STAGE_B and STAGE_C; when an arm has Stage A too,
# `graph_curvature` keeps the graph geometry independent of the ball used by the
# hyperbolic layers, which otherwise silently couples two unrelated choices.


def sample_configs(arm: str, n: int, seed: int):
    """n distinct random configs for `arm`, reproducible and order-stable."""
    space = ARMS[arm]
    rng = random.Random(hash((arm, seed)) & 0xFFFFFFFF)
    keys = sorted(space)
    seen, out = set(), []
    # Cap at the space size so small arms are not asked for impossible variety.
    total = 1
    for k in keys:
        total *= len(space[k])
    for _ in range(n * 200):
        if len(out) >= min(n, total):
            break
        cfg = {k: rng.choice(space[k]) for k in keys}
        sig = tuple(sorted(cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


# --------------------------------------------------------------------------- #
def build_cmd(target, cfg, task, seed, model, stress_kw, hidden_layer=-1):
    if target == "glue":
        full = dict(PAPER_DEFAULTS)
        full.update(cfg)
        full.update(task=task, seed=seed, model_name_or_path=model,
                    hidden_layer=hidden_layer)
        full["max_length"] = 512 if task == "imdb" else 128
        return [sys.executable, MAIN] + [f"--{k}={v}" for k, v in full.items()]

    full = {
        "model_name_or_path": model, "seed": seed, "verbose": 1,
        "pooling_method": "glot", "jk_mode": "cat",
        "gat_hidden_dim": 128, "num_layers": 2, "scorer_hidden": 128,
        "epochs": 3, "batch_size": 32, "eval_batch_size": 64, "lr": 1e-4,
        "num_train_samples": 2000, "num_eval_samples": 1000, "max_length": 128,
        "signal_position": "random",
    }
    full.update(stress_kw)
    full.update(cfg)
    return [sys.executable, STRESS] + [f"--{k}={v}" for k, v in full.items()]


def run_one(target, cfg, task, seed, model, stress_kw, hidden_layer=-1):
    cmd = build_cmd(target, cfg, task, seed, model, stress_kw, hidden_layer)
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    vals, tail, dens = [], [], []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line)
        if len(tail) > 30:
            tail.pop(0)
        md = RE_DENSITY.search(line)
        if md:
            dens.append((float(md.group(1)), float(md.group(2))))
            continue
        if target == "stress":
            m = RE_STRESS.search(line)
            if m:
                vals.append(float(m.group(1)))
            continue
        for rx, key in ((RE_STS, "spearman"), (RE_PAIR, None), (RE_SINGLE, None)):
            m = rx.search(line)
            if m:
                metric = PAPER_METRIC[task]
                g = {"spearman": 2, "pearson": 3, "acc": 2,
                     "f1": 3, "mcc": 3}[metric]
                vals.append(float(m.group(g)))
                break
    proc.wait()
    elapsed = round(time.time() - t0, 2)
    if proc.returncode != 0 or not vals:
        print(f"  [FAIL rc={proc.returncode}] {cfg}", flush=True)
        for ln in tail[-12:]:
            print("    | " + ln.rstrip(), flush=True)
        return None
    return {
        "score": round(max(vals) * 100.0, 3),
        "elapsed_sec": elapsed,
        "mean_density": round(sum(d for d, _ in dens) / len(dens), 5) if dens else "",
        "empty_frac": round(sum(e for _, e in dens) / len(dens), 5) if dens else "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["glue", "stress"], default="stress")
    p.add_argument("--task", default="cola")
    p.add_argument("--model", default="bert-base-uncased")
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--trials", type=int, default=8, help="equal budget per arm")
    p.add_argument("--tune_seed", type=int, default=42)
    p.add_argument("--confirm_seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--stage", choices=["tune", "confirm", "both"], default="both")
    p.add_argument("--distractor_ratio", type=float, default=0.9)
    p.add_argument("--relational_distance", type=int, default=20)
    p.add_argument("--hidden_layer", type=int, default=-1,
                   help="Transformer layer for token features (-1 = last, GLOT's choice). "
                        "bert-base delta-hyperbolicity: L1 .079 vs L12 .196, so the early "
                        "layers are far more tree-like and are where hyperbolic geometry "
                        "has a mechanism to help. Requires a pre-warmed cache for that "
                        "layer (gcp/prewarm_model.sh).")
    p.add_argument("--fine_baseline", action="store_true",
                   help="Give the cosine arms a 10-point density grid instead of "
                        "5, so best-of-N is compared against best-of-N rather than "
                        "best-of-10 vs best-of-5. See DENSITY_Q_FINE.")
    p.add_argument("--wide", action="store_true",
                   help="Also search the paper's Table 6 optimizer/architecture grid "
                        "(lr, weight_decay, num_layers, gat_hidden_dim, jk_mode, "
                        "proj_dim, scorer_hidden) on TOP of each arm's graph grid. "
                        "Applied identically to every arm, baseline included. "
                        "Without this, lr is pinned at a value the original authors "
                        "tuned for a EUCLIDEAN pooler, which confounds any negative "
                        "result about the hyperbolic readout/GNN. See WIDE.")
    p.add_argument("--out", default=os.path.join(HERE, "results", "campaign.csv"))
    args = p.parse_args()

    if args.fine_baseline:
        # Mutate in place so every arm built from COSINE picks this up.
        COSINE["tau_quantile"] = DENSITY_Q_FINE
        ARMS["baseline"] = COSINE

    if args.wide:
        # EVERY arm, so the extra freedom cannot favour one of them.
        for _arm in list(ARMS):
            ARMS[_arm] = _merge(ARMS[_arm], WIDE)

    stress_kw = dict(distractor_ratio=args.distractor_ratio,
                     relational_distance=args.relational_distance)
    setting = (f"r{args.distractor_ratio}_d{args.relational_distance}"
               if args.target == "stress" else args.task)
    if args.hidden_layer >= 0:
        setting += f"_L{args.hidden_layer}"
    # run_key is built from `setting`, so tagging it keeps the wide campaign in a
    # separate namespace. Without this, an existing graph-only row would satisfy
    # the dedup check for a wide trial and the optimizer axis would never run.
    if args.wide:
        setting += "_wide"
    res = ResultsCSV(args.out)

    def record(arm, cfg, seed, stage, trial):
        key = (f"{args.target}|{args.model}|{setting}|{arm}|t{trial}|s{seed}|{stage}")
        if res.has(key):
            return
        print(f"[{stage}] {arm:9s} trial={trial:<3} seed={seed:<3} {cfg}", flush=True)
        r = run_one(args.target, cfg, args.task, seed, args.model, stress_kw,
                    args.hidden_layer)
        if not r:
            return
        row = dict(run_key=key, target=args.target, model=args.model,
                   setting=setting, task=args.task, arm=arm, stage=stage,
                   trial=trial, seed=seed,
                   metric="acc" if args.target == "stress" else PAPER_METRIC[args.task],
                   detail=";".join(f"{k}={v}" for k, v in sorted(cfg.items())),
                   **cfg, **r)
        res.append(row)
        print(f"    -> {r['score']:.2f}  density={r['mean_density']} "
              f"[{r['elapsed_sec']}s]", flush=True)

    # ---- tuning: equal number of random trials per arm ----
    if args.stage in ("tune", "both"):
        for arm in args.arms:
            for i, cfg in enumerate(sample_configs(arm, args.trials, args.tune_seed)):
                record(arm, cfg, args.tune_seed, "tune", i)

    # ---- confirmation: best config per arm, on DIFFERENT seeds ----
    if args.stage in ("confirm", "both"):
        rows = [r for r in ResultsCSV(args.out).rows
                if r.get("stage") == "tune" and r.get("setting") == setting
                and r.get("model") == args.model and r.get("target") == args.target]
        best = {}
        for r in rows:
            a = r["arm"]
            if a not in args.arms:
                continue
            if a not in best or float(r["score"]) > float(best[a]["score"]):
                best[a] = r
        for arm, r in sorted(best.items()):
            cfg = {}
            for kv in r["detail"].split(";"):
                if not kv:
                    continue
                k, v = kv.split("=", 1)
                cfg[k] = v
            print(f"confirming {arm}: tune={r['score']} {r['detail']}", flush=True)
            for seed in args.confirm_seeds:
                record(arm, cfg, seed, "confirm", int(r["trial"]))


if __name__ == "__main__":
    main()
