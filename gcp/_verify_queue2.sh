#!/usr/bin/env bash
# Prove that queue_rest2.sh differs from the live queue in item 4 ONLY, and
# that item 4's new argument list is actually accepted by the driver.
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== items 1,2,3,5 must be byte-identical between the two queues ==="
for tag in "1\. RoBERTa" "2\. Stage C" "3\. Stage A" "5\. Decoder"; do
    a=$(awk "/---- $tag/,/^# ---- /" queue_rest.sh  | md5sum | cut -d' ' -f1)
    b=$(awk "/---- $tag/,/^# ---- /" queue_rest2.sh | md5sum | cut -d' ' -f1)
    [ "$a" = "$b" ] && echo "  item $tag  IDENTICAL" || echo "  item $tag  ***DIFFERS***"
done

echo
echo "=== item 4: old vs new argument lists ==="
echo "--- OLD ---"
sed -n '/START mteb-trained/,/results_csv/p' queue_rest.sh  | grep -E 'configs|mteb_tasks|^ ' | head -8
echo "--- NEW ---"
sed -n '/START mteb-trained/,/results_csv/p' queue_rest2.sh | grep -E 'configs|mteb_tasks|^ ' | head -8

echo
echo "=== does the driver ACCEPT the new config + task names? (dry run, CPU) ==="
CUDA_VISIBLE_DEVICES= ~/glotenv/bin/python -u run_all_experiments.py --with_mteb --dry_run \
    --models bert-base-uncased \
    --configs baseline A_threshold C_threshold AC_threshold \
    --seeds 1 \
    --mteb_tasks EmotionClassification SciFact RedditClustering \
                 AskUbuntuDupQuestions STS12 TwitterSemEval2015 SummEval \
                 Banking77Classification STS13 ArguAna \
                 TwentyNewsgroupsClustering SprintDuplicateQuestions \
    --mteb_ckpt_dir checkpoints_mteb \
    --results_csv /tmp/_dryrun_mteb.csv 2>&1 | tail -25

echo
echo "=== and does the OLD argument list still abort, as expected? ==="
CUDA_VISIBLE_DEVICES= ~/glotenv/bin/python -u run_all_experiments.py --with_mteb --dry_run \
    --models bert-base-uncased --configs baseline A C AC --seeds 1 \
    --mteb_tasks STS12 --results_csv /tmp/_dryrun_old.csv 2>&1 | tail -5
