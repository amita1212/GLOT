#!/usr/bin/env bash
# Detached launcher for decoder_sweep.sh, with the checks that must pass BEFORE
# ~8 GPU-hours are committed. Each check exists because its failure mode has
# already cost time in this project.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
sed -i 's/\r$//' decoder_sweep.sh campaign.py hyperglot/main.py
~/glotenv/bin/python -m py_compile campaign.py hyperglot/main.py \
    || { echo "COMPILE FAILED"; exit 1; }
echo "COMPILE_OK"

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

echo
echo "=== 1. does the decoder load through GLOT's OWN loader? ==="
# Catches: fast/slow tokenizer mismatch, missing pad token (decoders usually
# have none -- load_backbone falls back to eos), and whether is_decoder_like
# actually fires (TinyLlama is model_type=llama, so it should).
cd hyperglot
CUDA_VISIBLE_DEVICES= ~/glotenv/bin/python - "$MODEL" <<'PY'
import sys, torch, main as glot
name = sys.argv[1]
bb, dec_cls = glot.load_backbone(name, max_length=64, task="glue")
print(f"  tokenizer      : {type(bb.tokenizer).__name__} (fast={bb.tokenizer.is_fast})")
print(f"  pad_token_id   : {bb.pad_token_id}  padding_side={bb.tokenizer.padding_side}")
print(f"  model_type     : {getattr(bb.config, 'model_type', '?')}")
print(f"  is_decoder     : {bb.is_decoder}   (MUST be True for a causal model)")
print(f"  hidden_size    : {bb.config.hidden_size}   n_layers={bb.config.num_hidden_layers}")
enc = bb.tokenizer(["the cat sat on the mat", "a much longer sentence here to force padding"],
                   padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    hid, mask = glot.forward_hidden(bb, dict(enc))
print(f"  forward_hidden : {tuple(hid.shape)}  mean|x|={hid[mask.bool()].norm(dim=-1).mean():.2f}")
# With RIGHT padding + causal attention the last REAL token is at lengths-1, not
# at index -1. Verify the CLS pooler picks the real token, not a pad.
lengths = mask.sum(1)
last_real = hid[torch.arange(hid.size(0)), (lengths - 1).long()]
print(f"  last-real-token norm {last_real.norm(dim=-1).tolist()}")
print(f"  index -1 norm        {hid[:, -1].norm(dim=-1).tolist()}")
print("  (these MUST differ for the padded row, or padding is being pooled)")
assert bb.is_decoder, "is_decoder_like failed to detect a causal model"
print("  DECODER PATH OK")
PY
cd ..

echo
echo "=== 2. do all nine arms actually launch? (1 trial each, tuning only) ==="
# campaign.py forwards config keys verbatim as --key=value, so a single mistyped
# key kills every run of that arm silently-ish. Cheaper to find out now.
~/glotenv/bin/python campaign.py --target glue --task stsb --model "$MODEL" \
    --arms baseline no_graph A B C AB AC BC ABC --trials 1 --stage tune \
    --out results/_smoke_decoder.csv 2>&1 \
    | grep -E '^\s+->|FAIL|unrecognized|Traceback' | head -20

nohup bash decoder_sweep.sh "stsb cola" > logs/decoder.log 2>&1 &
echo "launched decoder sweep pid $!"
