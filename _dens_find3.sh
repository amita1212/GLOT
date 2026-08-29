#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
F=hyperglot/hyperbolic_graph.py
echo "=== $F : density / stats ==="
grep -n 'density\|observe\|class .*Stats\|_GRAPH_STATS\|self_loop\|arange' $F | head -40
echo
echo '=== stats class body ==='
L=$(grep -n 'class .*Stats' $F | head -1 | cut -d: -f1)
if [ -n "$L" ]; then sed -n "${L},$((L+50))p" $F; else echo "no Stats class"; fi
echo
echo '=== edge construction (threshold) ==='
grep -n -B3 -A12 'edge_index = ' $F | head -60
