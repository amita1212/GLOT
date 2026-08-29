"""Compare our results against GLOT's published Table 1, per task and backbone.

Paper: "Graph-based Latent Optimal Transport ... " (ICLR 2026), Table 1 -- a
comparison of pooling methods on GLUE across six frozen backbones. Metrics are
MCC for CoLA, Spearman for STS-B, F1 for MRPC/QQP, Accuracy for the rest, x100.

Our numbers are read from the campaign CSVs so this table can never drift from
the recorded runs. Confirmation-stage rows only (held-out seeds), never the
tuning stage, whose maximum is inflated by the search itself.
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

# --------------------------------------------------------------------------- #
# GLOT paper, Table 1. Only the two ENCODER backbones are transcribed: those are
# the ones we can run on a single L4. The decoder backbones (SmolLM2, TinyLlama,
# LLaMA-3B, Mistral-7B) are out of reach on this hardware.
# --------------------------------------------------------------------------- #
PAPER = {
    "BERT": {
        "[CLS]":   {"cola": 22.66, "stsb": 61.08, "mrpc": 79.58, "rte": 50.90},
        "Mean":    {"cola": 19.55, "stsb": 74.96, "mrpc": 80.28, "rte": 51.62},
        "Max":     {"cola": 15.79, "stsb": 74.12, "mrpc": 81.64, "rte": 51.98},
        "AdaPool": {"cola": 29.20, "stsb": 80.01, "mrpc": 77.99, "rte": 51.62},
        "GLOT":    {"cola": 47.49, "stsb": 83.86, "mrpc": 82.58, "rte": 59.21},
    },
    "RoBERTa": {
        "[CLS]":   {"cola":  6.92, "stsb": 52.87, "mrpc": 81.22, "rte": 52.34},
        "Mean":    {"cola": 23.69, "stsb": 70.55, "mrpc": 81.92, "rte": 54.63},
        "Max":     {"cola": 22.06, "stsb": 66.39, "mrpc": 81.52, "rte": 52.22},
        "AdaPool": {"cola": 26.80, "stsb": 71.12, "mrpc": 80.78, "rte": 50.45},
        "GLOT":    {"cola": 56.08, "stsb": 85.27, "mrpc": 81.95, "rte": 56.68},
    },
}

METRIC = {"cola": "MCC", "stsb": "Spearman", "mrpc": "F1", "rte": "Accuracy"}
TASKS = ["cola", "stsb", "mrpc", "rte"]
MODEL_LABEL = {"bert-base-uncased": "BERT", "roberta-base": "RoBERTa"}

# Edge density that the paper's OWN tau grid {0.1, 0.3, 0.6} produces, measured
# by cosine_stats.py on layer 12. RoBERTa's 10th-percentile cosine is 0.701,
# above every tau in the published search space, so none of its reported numbers
# can have come from a sparse graph.
PAPER_TAU_DENSITY = {
    "BERT":    {0.1: 0.850, 0.3: 0.638, 0.6: 0.149},
    "RoBERTa": {0.1: 1.000, 0.3: 1.000, 0.6: 0.992},
}


def load_confirm(paths):
    """(model, task, arm) -> list of held-out-seed scores.

    IMPORTANT FILTERS, both learned by getting this wrong:
      * target == "glue". The stress campaign stores task="cola" (campaign.py's
        default --task) even though it is the synthetic needle-in-haystack task
        scored by ACCURACY ~95. Mixing those into CoLA's MCC ~47 produced an
        impossible "CoLA 71.77" in the first version of this table.
      * one SETTING at a time. CoLA has runs at both layer 8 and layer 12; those
        are different experiments and averaging them would be meaningless, so
        any task spanning several settings is reported loudly rather than
        silently averaged.
    """
    out = defaultdict(list)
    settings = defaultdict(set)
    for p in paths:
        if "ABORTED" in p or "_smoke" in p or "stress" in os.path.basename(p):
            continue
        try:
            rows = list(csv.DictReader(open(p)))
        except OSError:
            continue
        for r in rows:
            if r.get("stage") != "confirm" or r.get("target") != "glue":
                continue
            key = (r.get("model", ""), r.get("task", ""), r.get("arm", ""))
            try:
                out[key].append(float(r["score"]))
            except (KeyError, TypeError, ValueError):
                continue
            settings[(key[0], key[1])].add(r.get("setting", ""))
    for k, v in sorted(settings.items()):
        if len(v) > 1:
            print(f"  !! {k[1]} spans multiple settings {sorted(v)} -- different "
                  f"experiments; pass --results explicitly to pick one")
    return out


def mean(v):
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=None)
    a = ap.parse_args()
    paths = a.results or sorted(glob.glob("results/campaign_*.csv"))
    data = load_confirm(paths)

    models = sorted({m for (m, _, _) in data})
    print("Sources:", ", ".join(os.path.basename(p) for p in paths))
    print()

    for model in models:
        label = MODEL_LABEL.get(model, model)
        paper = PAPER.get(label)
        print(f"================ {label}  ({model}) ================")
        hdr = f"{'method':<26}" + "".join(f"{t.upper():>12}" for t in TASKS)
        print(hdr)
        print(f"{'metric':<26}" + "".join(f"{METRIC[t]:>12}" for t in TASKS))
        print("-" * len(hdr))

        if paper:
            for meth in ["[CLS]", "Mean", "Max", "AdaPool", "GLOT"]:
                row = "".join(f"{paper[meth].get(t, float('nan')):>12.2f}" for t in TASKS)
                tag = f"paper {meth}" + (" (baseline)" if meth == "GLOT" else "")
                print(f"{tag:<26}{row}")
            print("-" * len(hdr))

        arms = sorted({arm for (m, _, arm) in data if m == model})
        # baseline first, then the rest by name
        arms = (["baseline"] if "baseline" in arms else []) + \
               [x for x in arms if x != "baseline"]
        for arm in arms:
            cells = ""
            for t in TASKS:
                v = mean(data.get((model, t, arm)))
                cells += f"{v:>12.2f}" if v is not None else f"{'-':>12}"
            n = max((len(data.get((model, t, arm), [])) for t in TASKS), default=0)
            print(f"{'ours ' + arm + f' (n={n})':<26}{cells}")

        if paper:
            print("-" * len(hdr))
            base = {t: mean(data.get((model, t, "baseline"))) for t in TASKS}
            cells = ""
            for t in TASKS:
                if base[t] is None:
                    cells += f"{'-':>12}"
                else:
                    cells += f"{base[t] - paper['GLOT'][t]:>+12.2f}"
            print(f"{'ours baseline - paper GLOT':<26}{cells}")
        print()

    print("=" * 72)
    print("EDGE DENSITY PRODUCED BY THE PAPER'S OWN tau GRID (layer 12)")
    print(f"{'backbone':<12}" + "".join(f"{'tau=' + str(t):>12}" for t in (0.1, 0.3, 0.6)))
    for label, d in PAPER_TAU_DENSITY.items():
        print(f"{label:<12}" + "".join(f"{d[t]:>12.3f}" for t in (0.1, 0.3, 0.6)))
    print()
    print("RoBERTa's 10th-percentile token cosine is 0.701, above EVERY tau in the")
    print("paper's search space {0.1, 0.3, 0.6}. So no setting in that space gives")
    print("RoBERTa a sparse token graph: its published numbers, including the")
    print("paper's best CoLA result (56.08), come from a near-complete graph in")
    print("which the GNN receives no selective relational structure.")


if __name__ == "__main__":
    main()
