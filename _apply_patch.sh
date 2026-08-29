#!/usr/bin/env bash
set -e
cd /home/t-amitalfasi/glot
sed -i 's/\r$//' patch_density.py
cp -n hyperglot/hyperbolic_graph.py hyperglot/hyperbolic_graph.py.bak || true
cp -n hyperglot/main.py hyperglot/main.py.bak || true
~/glotenv/bin/python patch_density.py
echo "--- syntax check ---"
~/glotenv/bin/python - <<'PY'
import ast
for p in ("hyperglot/hyperbolic_graph.py", "hyperglot/main.py"):
    ast.parse(open(p, encoding="utf-8").read())
    print("OK", p)
PY
echo "--- unit check of the new formula ---"
~/glotenv/bin/python - <<'PY'
import sys, torch
sys.path.insert(0, "hyperglot")
from hyperbolic_graph import _GraphStats

def d(n, pairs):
    ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    s = _GraphStats(report_after=10**9)
    s.observe(n, ei)
    return s.sum_density / s.n_graphs

n = 4
complete = [(i, j) for i in range(n) for j in range(n)]          # incl self-loops
selfonly = [(i, i) for i in range(n)]
half     = [(i, i) for i in range(n)] + [(0, 1), (1, 0), (2, 3), (3, 2)]
print(f"  complete+selfloops  -> {d(n, complete):.4f}  (expect 1.0000)")
print(f"  self-loops only     -> {d(n, selfonly):.4f}  (expect 0.0000)")
print(f"  4 of 12 off-diag    -> {d(n, half):.4f}  (expect 0.3333)")
PY
