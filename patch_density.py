#!/usr/bin/env python3
"""Patch _GraphStats.observe so density counts and denominator agree.

BUG: the numerator counted every edge INCLUDING self-loops, while the
denominator n(n-1) counts only OFF-DIAGONAL ordered pairs. For a complete
graph with self-loops that gives n^2 / (n^2 - n) = n/(n-1) > 1, which is not a
density at all -- observed at 1.09 on RoBERTa (n ~ 11). For a self-loop-only
graph it gives 1/(n-1) ~ 0.10 rather than 0.

FIX: subtract the self-loops from the numerator, so density is the fraction of
off-diagonal ordered pairs that carry an edge, in [0, 1]. `observe` now takes
the edge_index so it can count self-loops exactly rather than assuming they are
always present; the int form is still accepted for backwards compatibility.

Density is telemetry only and never touches the graph, so applying this while a
campaign is running cannot change any score.
"""
import re
import sys

PATH = "hyperglot/hyperbolic_graph.py"

OLD = '''    def observe(self, n_nodes: int, n_edges: int):
        if n_nodes < 2:
            return
        self.n_graphs += 1
        self.sum_density += n_edges / (n_nodes * (n_nodes - 1))
        self.n_empty += int(n_edges == 0)'''

NEW = '''    def observe(self, n_nodes: int, edges):
        """Accumulate OFF-DIAGONAL edge density.

        `edges` may be an edge_index tensor (preferred: self-loops are counted
        exactly) or an int edge count (legacy: self-loops are assumed present,
        one per node, which is what every adjacency in this file emits when
        cfg.self_loops is set).

        The previous version divided a self-loop-inclusive count by n(n-1) and
        so returned n/(n-1) > 1 for a complete graph. Densities logged before
        this fix are inflated by 1/(n-1) and must not be compared with ones
        logged after it.
        """
        if n_nodes < 2:
            return
        if hasattr(edges, "size"):          # edge_index tensor (2, E)
            n_edges = int(edges.size(1))
            n_self = int((edges[0] == edges[1]).sum()) if n_edges else 0
        else:                                # legacy int
            n_edges = int(edges)
            n_self = min(n_nodes, n_edges)
        n_off = max(0, n_edges - n_self)
        self.n_graphs += 1
        self.sum_density += n_off / (n_nodes * (n_nodes - 1))
        self.n_empty += int(n_off == 0)'''

src = open(PATH, encoding="utf-8").read()
if NEW.split("\n")[0] in src and "off-diagonal" in src.lower():
    print("already patched")
    sys.exit(0)
if OLD not in src:
    print("PATTERN NOT FOUND -- aborting, file unchanged")
    sys.exit(1)
open(PATH, "w", encoding="utf-8").write(src.replace(OLD, NEW))
print(f"patched {PATH}")

# Pass edge_index rather than a count at both call sites.
for path, old, new in [
    ("hyperglot/hyperbolic_graph.py",
     "_GRAPH_STATS.observe(x_b.size(0), edge_index.size(1))",
     "_GRAPH_STATS.observe(x_b.size(0), edge_index)"),
    ("hyperglot/main.py",
     "_GRAPH_STATS.observe(n, edge_index.size(1))",
     "_GRAPH_STATS.observe(n, edge_index)"),
]:
    s = open(path, encoding="utf-8").read()
    if new in s:
        print(f"  {path}: call site already updated")
    elif old in s:
        open(path, "w", encoding="utf-8").write(s.replace(old, new))
        print(f"  {path}: call site updated")
    else:
        print(f"  {path}: CALL SITE NOT FOUND")
