#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo "=== is the density-metric patch live on the VM? ==="
grep -n -A12 "def observe" hyperglot/hyperbolic_graph.py | head -24
echo
echo "=== unit test of the live metric ==="
~/glotenv/bin/python - <<'EOF'
import sys, torch
sys.path.insert(0, "hyperglot")
from hyperbolic_graph import _GraphStats
n = 12
# complete graph incl. self loops
ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
comp = torch.stack([ii.reshape(-1), jj.reshape(-1)])
loops = torch.stack([torch.arange(n), torch.arange(n)])
four = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
for name, ei, want in [("complete", comp, 1.0),
                       ("self-loops only", loops, 0.0),
                       ("4 of 132 off-diag", four, 4 / (n * (n - 1)))]:
    s = _GraphStats(enabled=True)
    try:
        s.observe(n, ei)
    except TypeError:
        s.observe(n, ei.size(1))          # unpatched signature
    print(f"  {name:20s} got {s.sum_density:.4f}  want {want:.4f}  "
          f"{'OK' if abs(s.sum_density - want) < 1e-4 else '*** WRONG ***'}")
EOF
