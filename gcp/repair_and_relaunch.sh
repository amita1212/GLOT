#!/usr/bin/env bash
# Repair the three defects found mid-run, then relaunch the pipeline.
#
#  1. ablation_fair.py had a syntax error (a stray join of two statements),
#     so STAGE 1 crashed instantly and produced zero rows.
#  2. sweep_paper_grid.py searched jk_mode in {cat,max,mean,none} per the
#     paper's Table 6, but the released code only accepts {cat,lstm,max};
#     "mean"/"none" made argparse exit rc=2 and silently killed ~25% of trials.
#     The grid is now {cat,max,lstm}, so the previously-written trial indices no
#     longer correspond to the same configs -> the old CSV must be discarded.
#  3. diagnostic_stress_test.py imports matplotlib, which setup_vm.sh never
#     installed, so STAGE 2 failed on every single run.
set -uo pipefail
cd /home/t-amitalfasi/glot

echo "=== 3. install the missing plotting deps ==="
/home/t-amitalfasi/glotenv/bin/pip install -q matplotlib seaborn
/home/t-amitalfasi/glotenv/bin/python -c "import matplotlib, seaborn; print('matplotlib', matplotlib.__version__)"

echo
echo "=== stop the running pipeline ==="
bash stop_pipeline.sh

echo
echo "=== 2. discard sweep CSVs written under the old jk_mode grid ==="
for f in results/sweep_cola.csv results/sweep_rte.csv results/sweep_stsb.csv; do
    if [ -f "$f" ]; then
        mv "$f" "${f%.csv}_OLD_jkmode.csv"
        echo "  archived $f"
    fi
done

echo
echo "=== sanity: every script parses ==="
/home/t-amitalfasi/glotenv/bin/python -m py_compile \
    ablation_fair.py sweep_paper_grid.py repro_paper.py && echo "  SYNTAX_OK"

echo
echo "=== relaunch ==="
nohup bash run_everything.sh > logs_pipeline.txt 2>&1 &
echo "PIPELINE_PID=$!"
sleep 20
tail -15 logs_pipeline.txt
