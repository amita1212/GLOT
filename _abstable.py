"""Absolute confirmed scores: arm x task x backbone, mean +/- sd over 15 seeds.

This is the analogue of the original paper's Table 1 (rows = method,
columns = task, blocks = backbone), but with OUR arms as the rows.
"""
import csv
import os
from collections import defaultdict
from math import sqrt

BERT = [("CoLA", "results/campaign_wide_cola.csv"),
        ("STS-B", "results/campaign_wide_stsb.csv"),
        ("MRPC", "results/campaign_wide_mrpc.csv"),
        ("RTE", "results/campaign_wide_rte.csv")]
ROB = [("CoLA", "results/campaign_rob_cola.csv"),
       ("STS-B", "results/campaign_rob_stsb.csv")]

def load(path):
    per = defaultdict(list)
    if not os.path.exists(path):
        return per
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r.get("stage") != "confirm":
            continue
        try:
            per[r["arm"]].append(float(r["score"]))
        except (ValueError, KeyError):
            pass
    return per

def cell(vals):
    if not vals:
        return None
    n = len(vals)
    m = sum(vals) / n
    sd = sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd, n

bert = {t: load(p) for t, p in BERT}
rob = {t: load(p) for t, p in ROB}

arms = ["baseline", "no_graph", "paper_tau", "density_fix",
        "A", "B", "C", "AB", "AC", "BC", "ABC"]

print("=" * 108)
print("ABSOLUTE CONFIRMED SCORES  (mean +/- sd, n=15 shared seeds)")
print("CoLA=MCC  STS-B=Spearman  MRPC=F1  RTE=accuracy")
print("=" * 108)
hdr = f"{'arm':12s}"
for t, _ in BERT:
    hdr += f" {'BERT/' + t:>16s}"
for t, _ in ROB:
    hdr += f" {'RoB/' + t:>16s}"
print(hdr)
print("-" * 108)
for arm in arms:
    line = f"{arm:12s}"
    any_cell = False
    for t, _ in BERT:
        c = cell(bert[t].get(arm, []))
        if c:
            any_cell = True
            line += f" {c[0]:9.2f}+-{c[1]:<5.2f}"
        else:
            line += f" {'--':>16s}"
    for t, _ in ROB:
        c = cell(rob[t].get(arm, []))
        if c:
            any_cell = True
            line += f" {c[0]:9.2f}+-{c[1]:<5.2f}"
        else:
            line += f" {'--':>16s}"
    if any_cell:
        print(line)

print()
print("=" * 108)
print("LATEX ROWS (BERT block)")
print("=" * 108)
NICE = {"baseline": r"\textsc{Glot} (baseline)", "no_graph": r"no graph",
        "paper_tau": r"published $\tau$", "density_fix": r"quantile $\tau$"}
for arm in arms:
    cells = []
    ok = False
    for t, _ in BERT:
        c = cell(bert[t].get(arm, []))
        if c:
            ok = True
            cells.append(f"{c[0]:.2f}\\,$\\pm${c[1]:.2f}")
        else:
            cells.append("---")
    for t, _ in ROB:
        c = cell(rob[t].get(arm, []))
        cells.append(f"{c[0]:.2f}\\,$\\pm${c[1]:.2f}" if c else "---")
    if ok:
        name = NICE.get(arm, arm)
        print(f"{name} & " + " & ".join(cells) + r" \\")
