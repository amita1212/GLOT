#!/usr/bin/env bash
# Live progress + ETA for whichever machine this runs on.
cd ~/glot || exit 1
now=$(date -u '+%F %T UTC')
echo "host=$(hostname)  $now"
echo

# ---------------- BGU: Table-1 queue + chain --------------------------------
if [ -f results/campaign_t1_tl_cola.csv ]; then
  C=$(awk -F, 'NR>1 && $0 ~ /confirm/' results/campaign_t1_tl_cola.csv | wc -l)
  T=$(awk -F, 'NR>1 && $0 ~ /tune/'    results/campaign_t1_tl_cola.csv | wc -l)
  echo "=== GPU1: TinyLlama CoLA ==="
  echo "  tune    $T/90"
  echo "  confirm $C/135"
  # mean seconds per confirm run so far
  AVG=$(awk -F, 'NR>1 && $0 ~ /confirm/ {for(i=1;i<=NF;i++) if($i+0>60 && $i+0<3000) {s+=$i; n++; break}} END{if(n)printf "%.0f", s/n}' results/campaign_t1_tl_cola.csv)
  [ -n "$AVG" ] && echo "  ~${AVG}s per confirm run" && \
    echo "  remaining: $((135-C)) runs  ~$(( (135-C)*AVG/3600 ))h"
  echo
  echo "  last log line:"; tail -1 logs/t1_tl_cola.log | cut -c1-110 | sed 's/^/    /'
  echo
  echo "=== GPU1: chain state ==="
  if [ -f .chain_gpu1.lock ] && [ -d "/proc/$(cat .chain_gpu1.lock)" ]; then
    echo "  chain alive (pid $(cat .chain_gpu1.lock)), waiting or running"
  else
    echo "  chain NOT running"
  fi
  tail -3 logs/chain_gpu1.log 2>/dev/null | sed 's/^/    /'
  echo
  echo "  queue tail:"; tail -2 logs/queue_table1_gap.log | sed 's/^/    /'
  echo
  echo "  corrective factorial rows: $( [ -f results/factorial_geom_cola_parity.csv ] && wc -l < results/factorial_geom_cola_parity.csv || echo 'not started' )/261"
  echo "  matched decoder rows     : $( [ -f results/campaign_decoder_stsb_matched.csv ] && wc -l < results/campaign_decoder_stsb_matched.csv || echo 'not started' )/46"
fi

# ---------------- L4: MTEB ---------------------------------------------------
if [ -d .mteb3_done ]; then
  echo "=== GPU2: MTEB Table-3 ==="
  for m in bert-base-uncased FacebookAI_roberta-base TinyLlama_TinyLlama-1.1B-Chat-v1.0; do
    n=$(find .mteb3_done -name "${m}__*" 2>/dev/null | wc -l)
    printf '  %-42s %3d/90\n' "$m" "$n"
  done
  R=$(find .mteb3_done -name "FacebookAI_roberta-base__*" 2>/dev/null | wc -l)
  echo
  echo "  roberta units done: $R/90"
  # per-unit minutes from the BERT block, for the ETA
  echo "  BERT block took: $(grep -c '^--- ' logs/mteb_table3.log) headers logged"
  if [ "$R" -gt 0 ]; then
    echo "  remaining roberta: $((90-R)) units"
  fi
  echo
  echo "  current unit:"; grep -E '^--- 20' logs/mteb_table3.log | tail -1 | sed 's/^/    /'
  echo "  gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/    /'
fi
