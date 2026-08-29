#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo '=== _GRAPH_STATS class ==='
grep -n '_GRAPH_STATS\|class .*GraphStats\|def observe' hyperglot/main.py
echo
echo '=== class body ==='
L=$(grep -n 'class .*GraphStats' hyperglot/main.py | head -1 | cut -d: -f1)
sed -n "${L},$((L+45))p" hyperglot/main.py
echo
echo '=== how self-loops are added (threshold branch) ==='
grep -n 'add_self_loops\|self_loop\|eye(\|arange(n)' hyperglot/main.py | head -20
