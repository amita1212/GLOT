#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo '=== no_graph arm definition in campaign.py ==='
grep -n -A6 '"no_graph"' campaign.py
echo
echo '=== how adjacency none/self is built in hyperglot/main.py ==='
grep -n -i 'no_graph\|self_loop\|self-loop\|eye(\|identity' hyperglot/main.py | head -30
echo
echo '=== graph_adj choices ==='
grep -n -- '--graph_adj' hyperglot/main.py
grep -n -B2 -A10 'graph_adj ==' hyperglot/main.py | head -60
