#!/usr/bin/env python
"""Full confirmed TinyLlama STS-B configs for baseline, B and C, ready to paste."""
import csv
import os

ROOT = os.path.expanduser("~/glot")
FILES = ["results/campaign_decoder_stsb.csv", "results/campaign_decoder_stsb_BC.csv"]

KEYS = ["graph_metric", "jk_mode", "lr", "num_layers", "gat_hidden_dim",
        "proj_dim", "scorer_hidden", "weight_decay", "tau_quantile",
        "rho_quantile", "feature_mode", "graph_curvature", "curvature",
        "hyperbolic_gnn", "hyp_gnn_type", "gnn_input_clip", "gnn_input_scale",
        "hyperbolic_readout", "readout_clip", "readout_scale",
        "learnable_curvature"]

want = ("baseline", "B", "C")
rows = {}
for f in FILES:
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p, encoding="utf-8", errors="ignore")):
        if r.get("stage") == "confirm" and r.get("arm") in want:
            rows.setdefault(r["arm"], (os.path.basename(f), r))

for arm in want:
    if arm not in rows:
        print(f"# {arm}: NOT FOUND\n")
        continue
    src, r = rows[arm]
    print(f'    # {arm}, transcribed from {src}')
    print(f'    "{arm}": {{')
    for k in KEYS:
        v = (r.get(k) or "").strip()
        if v in ("", "-1.0", "-1", "None", "nan"):
            continue
        if k in ("hyperbolic_gnn", "hyperbolic_readout", "gnn_input_scale",
                 "readout_scale", "learnable_curvature") and v in ("0", "0.0", "False"):
            continue
        if k in ("gnn_input_clip", "readout_clip"):
            try:
                if float(v) == 0.0:
                    continue
            except ValueError:
                pass
        print(f'        "{k}": "{v}",')
    print("    },")
