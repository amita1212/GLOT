#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
sed -i 's/\r$//' chain_mrpc_rte.sh
nohup bash chain_mrpc_rte.sh > logs/chain_mrpc_rte.log 2>&1 &
echo "chain launched pid $!"
