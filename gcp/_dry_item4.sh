#!/usr/bin/env bash
# Authoritative check: run the driver in --dry_run with EXACTLY the arguments
# queue item 4 uses, and count what it would launch. Nothing executes.
set -u
cd /home/t-amitalfasi/glot || exit 1
D=hyperglot_new/run_all_experiments.py
PY=~/glotenv/bin/python

echo "=== how args.tasks is resolved ==="
grep -n 'args\.tasks' "$D"

echo
echo "=== DRY RUN with item 4's exact arguments ==="
CUDA_VISIBLE_DEVICES= "$PY" -u "$D" --with_mteb --dry_run \
    --models bert-base-uncased \
    --configs baseline A C AC \
    --seeds 1 2 3 4 5 \
    --mteb_tasks Banking77Classification STS12 STS13 SciFact ArguAna \
                 TwentyNewsgroupsClustering SprintDuplicateQuestions \
    --mteb_ckpt_dir checkpoints_mteb \
    --results_csv /tmp/_dry_item4.csv 2>&1 > /tmp/_dry_item4.log

echo "  distinct GLUE tasks it would train on:"
grep -o -- '--task=[a-z0-9]*' /tmp/_dry_item4.log | sort -u | tr '\n' ' '; echo
echo "  distinct arms it would use:"
grep -o -- '--arm=[A-Za-z0-9_]*' /tmp/_dry_item4.log | sort -u | tr '\n' ' '; echo
echo "  total commands it would run: $(grep -c 'main.py' /tmp/_dry_item4.log)"
echo "  tail of dry-run summary:"
tail -4 /tmp/_dry_item4.log

echo
echo "=== DRY RUN with the CORRECTED arguments ==="
CUDA_VISIBLE_DEVICES= "$PY" -u "$D" --with_mteb --dry_run \
    --models bert-base-uncased \
    --configs baseline A_threshold C_threshold AC_threshold \
    --seeds 1 2 3 4 5 \
    --mteb_tasks EmotionClassification SciFact RedditClustering \
                 AskUbuntuDupQuestions STS12 TwitterSemEval2015 SummEval \
                 Banking77Classification STS13 ArguAna \
                 TwentyNewsgroupsClustering SprintDuplicateQuestions \
    --mteb_train_file /home/t-amitalfasi/glot/data/msmarco-triplets.jsonl \
    --mteb_ckpt_dir /home/t-amitalfasi/glot/checkpoints_mteb \
    --results_csv /tmp/_dry_item4b.csv \
    --tasks 2>&1 > /tmp/_dry_item4b.log

echo "  distinct GLUE tasks: [$(grep -o -- '--task=[a-z0-9]*' /tmp/_dry_item4b.log | sort -u | tr '\n' ' ')]"
echo "  distinct arms: $(grep -o -- '--arm=[A-Za-z0-9_]*' /tmp/_dry_item4b.log | sort -u | tr '\n' ' ')"
echo "  total commands: $(grep -c 'main.py' /tmp/_dry_item4b.log)"
tail -4 /tmp/_dry_item4b.log
