#!/bin/bash
# Resume the BERT sweep after dropping the Stage B (ABC) arm.
# The orchestrator is resumable: it skips (model,task,arm,seed) rows already
# in the CSV, so all completed work (cola..mnli non-ABC) is preserved and it
# continues at qnli with only baseline/A/C/AC arms.
cd ~/GLOT
source ~/env.sh
python3 run_all_experiments.py --models bert-base-uncased --seeds 42 >> ~/sweep_bert.log 2>&1
