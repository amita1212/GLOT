#!/usr/bin/env python
"""HyperGLOT full experiment orchestrator.

Runs the complete ablation grid from the research plan across every task and
backbone used in the GLOT article, saving each finished result to CSV (and
optionally committing/pushing it) immediately so the sweep is fully resumable.

Ablation arms (each maps to the three orthogonal HyperGLOT switches):

    baseline : original GLOT                 (cosine graph, Euclidean GNN + readout)
    A        : hyperbolic graph construction  (Poincare distance edges)   -- Stage A
    C        : hyperbolic Token-GNN           (HGCN message passing)       -- Stage C
    AC       : hyperbolic graph + Token-GNN   (Stage A + C, Euclidean readout)
    ABC      : all three combined             (Stage A + B + C)

Every arm except the baseline is run in two graph-adjacency variants:
``threshold`` and ``knn``. That gives 1 baseline + 4 arms x 2 = **9 configs**
per (task, model).

Tasks (all evaluated in the article):
    * GLUE    : cola, sst2, stsb, mrpc, qqp, mnli, qnli, rte, wnli
    * IMDB    : long-text classification
    * stress  : the negation "needle in a haystack" diagnostic (distractor sweep)
    * mteb    : MTEB tasks (opt-in; see --with_mteb)

Backbones (all reported in the article): encoders (BERT, RoBERTa) and
decoder-only LLMs (TinyLlama, SmolLM2, Llama-3.2-3B, Mistral-7B).

Usage examples
--------------
    # Everything, BERT only, default seed:
    python run_all_experiments.py --models bert-base-uncased

    # A specific subset, multiple seeds, dry-run to preview commands:
    python run_all_experiments.py --models bert-base-uncased \
        --tasks cola stsb --configs baseline A_threshold ABC_knn \
        --seeds 41 42 43 --dry_run

    # Full grid across all backbones, committing each result to git:
    python run_all_experiments.py --all --git_push

The heavy lifting (training, metric computation, CSV row writing) is done by
``main.py`` / ``diagnostic_stress_test.py`` -- this script only builds the right
command lines, skips already-finished runs, and handles bookkeeping.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_CSV = os.path.join(HERE, "results", "hyperglot_all_results.csv")
DEFAULT_STRESS_CSV = os.path.join(HERE, "results", "hyperglot_stress_results.csv")


# --------------------------------------------------------------------------- #
# The 9 configurations (baseline + 4 arms x {threshold, knn}).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    name: str
    graph_metric: str        # cosine | poincare
    graph_adj: str           # threshold | knn
    hyperbolic_gnn: int      # 0 | 1   (Stage C)
    hyperbolic_readout: int  # 0 | 1   (Stage B)


def build_configs() -> Dict[str, Config]:
    cfgs: Dict[str, Config] = {
        "baseline": Config("baseline", "cosine", "threshold", 0, 0),
    }
    # (arm label, graph_metric, hyp_gnn, hyp_readout)
    arms = [
        ("A", "poincare", 0, 0),   # Stage A: hyperbolic graph construction
        ("C", "cosine", 1, 0),     # Stage C: hyperbolic Token-GNN
        ("AC", "poincare", 1, 0),  # Stage A + C: hyperbolic graph + hyperbolic GNN
        ("ABC", "poincare", 1, 1), # all three combined
    ]
    for arm, metric, hgnn, hread in arms:
        for adj in ("threshold", "knn"):
            name = f"{arm}_{adj}"
            cfgs[name] = Config(name, metric, adj, hgnn, hread)
    return cfgs


CONFIGS = build_configs()


# --------------------------------------------------------------------------- #
# Backbones used in the article.
# --------------------------------------------------------------------------- #
@dataclass
class Model:
    name: str
    is_decoder: bool = False
    # optional per-model overrides (e.g. smaller batch for 7B models)
    overrides: Dict[str, str] = field(default_factory=dict)


MODELS: Dict[str, Model] = {
    # Encoders
    "bert-base-uncased": Model("bert-base-uncased", is_decoder=False),
    "FacebookAI/roberta-base": Model("FacebookAI/roberta-base", is_decoder=False),
    # Decoder-only LLMs (frozen). Batch sizes shrink as models grow.
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": Model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", is_decoder=True,
        overrides={"batch_size": "16", "eval_batch_size": "32"}),
    "HuggingFaceTB/SmolLM2-1.7B": Model(
        "HuggingFaceTB/SmolLM2-1.7B", is_decoder=True,
        overrides={"batch_size": "16", "eval_batch_size": "32"}),
    "meta-llama/Llama-3.2-3B": Model(
        "meta-llama/Llama-3.2-3B", is_decoder=True,
        overrides={"batch_size": "8", "eval_batch_size": "16"}),
    "mistralai/Mistral-7B-v0.1": Model(
        "mistralai/Mistral-7B-v0.1", is_decoder=True,
        overrides={"batch_size": "4", "eval_batch_size": "8"}),
}


# --------------------------------------------------------------------------- #
# Task groups + their (article) hyper-parameters.
# --------------------------------------------------------------------------- #
GLUE_SINGLE = ["cola", "sst2"]
GLUE_PAIR = ["mrpc", "qqp", "mnli", "qnli", "rte", "wnli"]
GLUE_STS = ["stsb"]
GLUE_TASKS = GLUE_SINGLE + GLUE_STS + GLUE_PAIR
DOC_TASKS = ["imdb"]
STRESS_DISTRACTORS = [0.2, 0.5, 0.8, 0.9]
DEFAULT_MTEB_TASKS = [
    "Banking77Classification", "STS12", "STS13", "SciFact",
    "ArguAna", "TwentyNewsgroupsClustering", "SprintDuplicateQuestions",
]

# Shared GLUE/IMDB hyper-parameters (from README).
GLUE_HP = {
    "epochs": "3", "batch_size": "32", "eval_batch_size": "64", "lr": "2e-4",
    "weight_decay": "0.0", "gnn_type": "gat", "scorer_hidden": "128",
    "gat_hidden_dim": "256", "num_layers": "2", "jk_mode": "cat",
    "tau": "0.8", "rho": "1.0", "curvature": "1.0", "knn_k": "8",
    "proj_dim": "256", "precompute_hidden_states": "1", "finetune_backbone": "0",
    "adaptive_length": "0",
}


def glue_max_length(task: str) -> str:
    return "512" if task == "imdb" else "128"


# --------------------------------------------------------------------------- #
# Resume bookkeeping: read done-keys from the CSV.
# --------------------------------------------------------------------------- #
def load_done_keys(path: str, key_fields: List[str]) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(tuple(str(row.get(k, "")) for k in key_fields))
    return done


def maybe_git_push(paths: List[str], message: str, push: bool) -> None:
    """Stage, commit and (optionally) push the given result files."""
    try:
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            return
        subprocess.run(["git", "add", *existing], cwd=HERE, check=False)
        commit = subprocess.run(
            ["git", "commit", "-m", message], cwd=HERE,
            capture_output=True, text=True)
        if commit.returncode != 0 and "nothing to commit" in (commit.stdout + commit.stderr):
            return
        if push:
            subprocess.run(["git", "push"], cwd=HERE, check=False)
    except Exception as exc:  # never let bookkeeping abort the sweep
        print(f"[git] skipped ({exc})", flush=True)


# --------------------------------------------------------------------------- #
# Command builders.
# --------------------------------------------------------------------------- #
def build_main_cmd(python: str, model: Model, task: str, cfg: Config, seed: int,
                   results_csv: str, mteb_task: Optional[str] = None) -> List[str]:
    hp = dict(GLUE_HP)
    hp.update(model.overrides)
    cmd = [
        python, os.path.join(HERE, "main.py"),
        f"--model_name_or_path={model.name}",
        f"--decoder_cls_last_token={1 if model.is_decoder else 0}",
        f"--task={task}",
        f"--max_length={glue_max_length(task)}",
        f"--seed={seed}",
        "--verbose=1",
        "--pooling_method=glot",
        f"--graph_adj={cfg.graph_adj}",
        f"--graph_metric={cfg.graph_metric}",
        f"--hyperbolic_gnn={cfg.hyperbolic_gnn}",
        f"--hyperbolic_readout={cfg.hyperbolic_readout}",
        f"--arm={cfg.name}",
        f"--results_csv={results_csv}",
        "--run_tag=hyperglot",
    ]
    for k, v in hp.items():
        cmd.append(f"--{k}={v}")
    if task == "mteb" and mteb_task:
        cmd.append(f"--mteb_task={mteb_task}")
    return cmd


def build_stress_cmd(python: str, model: Model, cfg: Config, distractor: float,
                     seed: int, stress_csv: str) -> List[str]:
    ov = model.overrides
    return [
        python, os.path.join(HERE, "diagnostic_stress_test.py"),
        f"--model_name_or_path={model.name}",
        f"--decoder_cls_last_token={1 if model.is_decoder else 0}",
        f"--distractor_ratio={distractor}",
        "--epochs=3",
        f"--batch_size={ov.get('batch_size', '32')}",
        f"--eval_batch_size={ov.get('eval_batch_size', '32')}",
        "--gat_hidden_dim=64",
        "--scorer_hidden=256",
        "--num_layers=4",
        "--tau=0.6",
        "--lr=1e-4",
        f"--seed={seed}",
        "--pooling_method=glot",
        f"--graph_adj={cfg.graph_adj}",
        f"--graph_metric={cfg.graph_metric}",
        f"--hyperbolic_gnn={cfg.hyperbolic_gnn}",
        f"--hyperbolic_readout={cfg.hyperbolic_readout}",
        f"--arm={cfg.name}",
        f"--results_csv={stress_csv}",
        "--run_tag=hyperglot",
    ]


def run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=HERE).returncode


# --------------------------------------------------------------------------- #
# Main driver.
# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="Run every model x task x config.")
    p.add_argument("--models", nargs="*", default=["bert-base-uncased"],
                   help=f"Subset of: {list(MODELS)}")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of GLUE/IMDB tasks (default: all GLUE+IMDB).")
    p.add_argument("--configs", nargs="*", default=None,
                   help=f"Subset of: {list(CONFIGS)} (default: all 9).")
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--with_stress", action="store_true", help="Also run the negation stress test.")
    p.add_argument("--stress_only", action="store_true", help="Run only the stress test.")
    p.add_argument("--with_mteb", action="store_true", help="Also run MTEB tasks (untrained-pooler eval).")
    p.add_argument("--mteb_tasks", nargs="*", default=DEFAULT_MTEB_TASKS)
    p.add_argument("--results_csv", default=DEFAULT_RESULTS_CSV)
    p.add_argument("--stress_csv", default=DEFAULT_STRESS_CSV)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--git_push", action="store_true",
                   help="git add+commit+push the CSV after each finished run.")
    p.add_argument("--git_commit", action="store_true",
                   help="git add+commit (no push) after each finished run.")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running.")
    p.add_argument("--stop_on_error", action="store_true")
    args = p.parse_args()

    models = list(MODELS.values()) if args.all else [MODELS[m] for m in args.models if m in MODELS]
    unknown = [m for m in args.models if m not in MODELS] if not args.all else []
    for u in unknown:
        print(f"[warn] unknown model '{u}' (add it to MODELS); skipping.")
    if not models:
        print("No valid models selected.")
        return 2

    tasks = args.tasks if args.tasks is not None else (GLUE_TASKS + DOC_TASKS)
    if args.all:
        tasks = GLUE_TASKS + DOC_TASKS
    cfg_names = args.configs if args.configs is not None else list(CONFIGS)
    configs = [CONFIGS[c] for c in cfg_names if c in CONFIGS]
    do_commit = args.git_commit or args.git_push

    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.stress_csv)), exist_ok=True)

    main_key_fields = ["model", "task", "mteb_task", "arm", "seed"]
    stress_key_fields = ["model", "arm", "distractor_ratio", "seed"]

    n_run, n_skip, n_fail = 0, 0, 0

    # ---- Standard tasks (GLUE, IMDB, optionally MTEB) --------------------- #
    if not args.stress_only:
        for model in models:
            for task in tasks:
                for cfg in configs:
                    for seed in args.seeds:
                        done = load_done_keys(args.results_csv, main_key_fields)
                        key = (model.name, task, "", cfg.name, str(seed))
                        if key in done:
                            n_skip += 1
                            continue
                        cmd = build_main_cmd(args.python, model, task, cfg, seed, args.results_csv)
                        rc = run_cmd(cmd, args.dry_run)
                        if rc == 0:
                            n_run += 1
                            if do_commit and not args.dry_run:
                                maybe_git_push([args.results_csv],
                                               f"results: {model.name} {task} {cfg.name} seed{seed}",
                                               push=args.git_push)
                        else:
                            n_fail += 1
                            print(f"[fail rc={rc}] {model.name} {task} {cfg.name} seed{seed}")
                            if args.stop_on_error:
                                return rc

        if args.with_mteb:
            for model in models:
                for mteb_task in args.mteb_tasks:
                    for cfg in configs:
                        for seed in args.seeds:
                            done = load_done_keys(args.results_csv, main_key_fields)
                            key = (model.name, "mteb", mteb_task, cfg.name, str(seed))
                            if key in done:
                                n_skip += 1
                                continue
                            cmd = build_main_cmd(args.python, model, "mteb", cfg, seed,
                                                 args.results_csv, mteb_task=mteb_task)
                            rc = run_cmd(cmd, args.dry_run)
                            if rc == 0:
                                n_run += 1
                                if do_commit and not args.dry_run:
                                    maybe_git_push([args.results_csv],
                                                   f"results(mteb): {model.name} {mteb_task} {cfg.name}",
                                                   push=args.git_push)
                            else:
                                n_fail += 1
                                if args.stop_on_error:
                                    return rc

    # ---- Negation stress test -------------------------------------------- #
    if args.with_stress or args.stress_only or args.all:
        for model in models:
            for cfg in configs:
                for distractor in STRESS_DISTRACTORS:
                    for seed in args.seeds:
                        done = load_done_keys(args.stress_csv, stress_key_fields)
                        key = (model.name, cfg.name, str(distractor), str(seed))
                        if key in done:
                            n_skip += 1
                            continue
                        cmd = build_stress_cmd(args.python, model, cfg, distractor, seed, args.stress_csv)
                        rc = run_cmd(cmd, args.dry_run)
                        if rc == 0:
                            n_run += 1
                            if do_commit and not args.dry_run:
                                maybe_git_push([args.stress_csv],
                                               f"stress: {model.name} {cfg.name} d{distractor} seed{seed}",
                                               push=args.git_push)
                        else:
                            n_fail += 1
                            if args.stop_on_error:
                                return rc

    print(f"\n=== done. ran={n_run} skipped={n_skip} failed={n_fail} ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
