#!/usr/bin/env bash
# Resolve the two omissions precisely.
#
# (a) --tasks default is None. If None means "all GLUE", then item 4, which
#     omits --tasks, would launch the entire GLUE ablation grid -- including
#     QQP and MNLI, which the paper itself estimates at 38 and 41 DAYS.
# (b) --mteb_train_file default is os.path.join(HERE, ...) where HERE is the
#     DRIVER's directory (hyperglot_new/), not the shared data dir under
#     ~/glot. If so the default path does not exist and the MS MARCO stage
#     cannot run.
set -u
cd /home/t-amitalfasi/glot || exit 1
D=hyperglot_new/run_all_experiments.py

echo "=== how is args.tasks resolved when None? ==="
grep -n 'args.tasks' "$D"

echo
echo "=== HERE definition in the driver ==="
grep -n '^HERE' "$D"

echo
echo "=== would the DEFAULT train file exist? ==="
ls -la hyperglot_new/data/msmarco-triplets.jsonl 2>/dev/null \
    || echo "  MISSING -> default is wrong; --mteb_train_file must be passed explicitly"

echo
echo "=== authoritative: ask argparse itself what item 4's args would become ==="
CUDA_VISIBLE_DEVICES= ~/glotenv/bin/python - <<'PY'
import importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location(
    "drv", "/home/t-amitalfasi/glot/hyperglot_new/run_all_experiments.py")
m = importlib.util.module_from_spec(spec)
sys.argv = ["run_all_experiments.py"]
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
p = m.build_parser() if hasattr(m, "build_parser") else None
if p is None:
    # fall back: find the parser factory by name
    cands = [n for n in dir(m) if "pars" in n.lower()]
    print("no build_parser(); candidates:", cands)
else:
    a = p.parse_args([
        "--with_mteb", "--models", "bert-base-uncased",
        "--configs", "baseline", "A", "C", "AC",
        "--seeds", "1", "--mteb_tasks", "STS12",
    ])
    print("tasks          =", a.tasks)
    print("mteb_train_file=", a.mteb_train_file)
    print("configs        =", a.configs)
    import os
    print("train file exists?", os.path.exists(a.mteb_train_file))
    cfgs = m.build_configs()
    print("of those configs, which EXIST:", [c for c in a.configs if c in cfgs])
    print("which are silently DROPPED :", [c for c in a.configs if c not in cfgs])
PY
