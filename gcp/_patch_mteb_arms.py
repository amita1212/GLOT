"""Add the two arms the MTEB driver is missing: B alone and ABC.

`build_configs()` in run_all_experiments.py defines A, C, AC and AB, but:
  * there is no standalone B  (Stage B was only ever reachable inside AB), and
  * ABC is present only as a COMMENTED-OUT line.

The MTEB Table-3 campaign needs both, so this inserts them next to the others.
Idempotent: running it twice is a no-op.
"""
import re
import sys

PATH = "/home/t-amitalfasi/glot/hyperglot_new/run_all_experiments.py"

NEW_ARMS = (
    '        ("B", "cosine", 0, 1),    # Stage B alone: cosine graph + Einstein readout\n'
    '        ("ABC", "poincare", 1, 1),  # all three stages\n'
)

with open(PATH, encoding="utf-8") as fh:
    src = fh.read()

if '("B", "cosine", 0, 1)' in src and '("ABC", "poincare", 1, 1)' in src:
    print("already patched -- no change")
    sys.exit(0)

anchor = '        ("AB", "poincare", 0, 1),'
idx = src.find(anchor)
if idx < 0:
    print("ANCHOR NOT FOUND -- refusing to guess", file=sys.stderr)
    sys.exit(1)

eol = src.index("\n", idx) + 1
src = src[:eol] + NEW_ARMS + src[eol:]

with open(PATH, "w", encoding="utf-8") as fh:
    fh.write(src)

print("patched: added B and ABC")

# Prove the registry now contains what the campaign will ask for.
sys.path.insert(0, "/home/t-amitalfasi/glot/hyperglot_new")
import run_all_experiments as rae  # noqa: E402

want = ["baseline", "A_threshold", "B_threshold",
        "C_threshold", "AC_threshold", "ABC_threshold"]
print("\narm                graph_metric  adj        hyp_gnn  hyp_readout")
ok = True
for name in want:
    cfg = rae.CONFIGS.get(name)
    if cfg is None:
        print(f"{name:<18} *** MISSING ***")
        ok = False
    else:
        print(f"{name:<18}{cfg.graph_metric:<14}{cfg.graph_adj:<11}"
              f"{cfg.hyperbolic_gnn:<9}{cfg.hyperbolic_readout}")
sys.exit(0 if ok else 1)
