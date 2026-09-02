#!/usr/bin/env python
"""Do the two decoder campaigns agree on the knobs the arms did NOT search?

If lr / depth / width differ between the baseline campaign and the B/C
campaign, the matched rerun must pin them explicitly, or the 'matched' campaign
would differ in more than the stage under test.
"""
import csv
import os

ROOT = os.path.expanduser("~/glot/results")
KEYS = ["lr", "num_layers", "gat_hidden_dim", "proj_dim", "scorer_hidden",
        "weight_decay", "jk_mode", "epochs", "batch_size", "max_length"]

for f in ("campaign_decoder_stsb.csv", "campaign_decoder_stsb_BC.csv"):
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print(f"{f}: missing")
        continue
    rd = csv.DictReader(open(p, encoding="utf-8", errors="ignore"))
    cols = rd.fieldnames or []
    row = next((r for r in rd if r.get("stage") == "confirm"), None)
    print(f"--- {f} ---")
    for k in KEYS:
        if k not in cols:
            print(f"    {k:<16} <no such column>")
        else:
            print(f"    {k:<16} = {row.get(k)!r}")
    print()
