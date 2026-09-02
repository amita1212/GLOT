"""Corrected GLUE cost model.

The old _taskcost.py scaled linearly from full train-set size and ignored two
things that matter a lot:
  1. The original paper trains QQP/QNLI/MNLI on train[:20000], not the full set
     (their Appendix B.3). Following their protocol makes those tasks CHEAP.
  2. Pair tasks embed two sentences per example, so they cost ~2x a single-
     sentence task of the same size, and the DEV set is evaluated every epoch
     (QQP dev is 40k examples -- bigger than its capped train set).

Fits cost = c0 + c1 * P  where  P = epochs * (n_train_eff + n_dev) * (2 if pair)
on the four tasks we have measured, then projects the rest.
"""
import numpy as np

EPOCHS = 2
SUBSAMPLE = 20000          # original's cap ...
CAPPED = {"qqp", "qnli", "mnli"}   # ... applies to these three ONLY (their B.3).
# SST-2 is NOT capped: it is reported at its full 67,349 training examples.

# n_train, n_dev, is_pair
GLUE = {
    "wnli":  (635,    71,    True),
    "rte":   (2490,   277,   True),
    "mrpc":  (3668,   408,   True),
    "stsb":  (5749,   1500,  True),
    "cola":  (8551,   1043,  False),
    "sst2":  (67349,  872,   False),
    "qnli":  (104743, 5463,  True),
    "qqp":   (363846, 40430, True),
    "mnli":  (392702, 9815 + 9832, True),
}

# measured seconds per run, from results_vm campaign CSVs (BERT-base, L=128)
MEASURED = {"cola": 138.4, "stsb": 223.5, "mrpc": 165.5, "rte": 124.0}
DECODER_X = {"cola": 548.4 / 138.4, "stsb": 808.0 / 223.5}  # TinyLlama / BERT


def used_train(task):
    n_tr = GLUE[task][0]
    return min(n_tr, SUBSAMPLE) if task in CAPPED else n_tr


def passes(task):
    _, n_dev, pair = GLUE[task]
    return EPOCHS * (used_train(task) + n_dev) * (2 if pair else 1)


X = np.array([[1.0, passes(t)] for t in MEASURED])
y = np.array([MEASURED[t] for t in MEASURED])
c0, c1 = np.linalg.lstsq(X, y, rcond=None)[0]
pred = X @ np.array([c0, c1])

print("FIT  cost(s) = %.1f + %.3e * passes      (BERT-base, max_len 128)" % (c0, c1))
print(f"{'task':6s} {'passes':>9s} {'measured':>9s} {'fitted':>8s} {'err':>7s}")
for t, p, m in zip(MEASURED, X[:, 1], y):
    f = c0 + c1 * p
    print(f"{t:6s} {p:9,.0f} {m:9.1f} {f:8.1f} {100*(f-m)/m:6.1f}%")

print()
print("PROJECTED COST, following the ORIGINAL's protocol (train[:20000] for "
      "qqp/qnli/mnli)")
print("-" * 96)
print(f"{'task':6s} {'n_train':>8s} {'used':>7s} {'n_dev':>7s} {'s/run':>7s} "
      f"{'full 9-arm':>11s} {'reduced':>9s} {'confirm-only':>13s} {'cache':>8s}")
print(f"{'':6s} {'':>8s} {'':>7s} {'':>7s} {'':>7s} {'495 runs':>11s} "
      f"{'4x55=220':>9s} {'9x15=135':>13s} {'(GB)':>8s}")
tot_full = tot_missing = 0.0
for t in ("wnli", "rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"):
    n_tr, n_dev, pair = GLUE[t]
    used = used_train(t)
    s = c0 + c1 * passes(t)
    full, red, conf = 495 * s / 3600, 220 * s / 3600, 135 * s / 3600
    gb = used * 128 * 768 * 2 / 1e9 * (2 if pair else 1)
    done = t in ("cola", "stsb", "mrpc", "rte")
    mark = "  <- done" if done else ""
    if not done:
        tot_missing += full
    tot_full += full
    print(f"{t:6s} {n_tr:8,d} {used:7,d} {n_dev:7,d} {s:7.0f} "
          f"{full:9.1f} h {red:7.1f} h {conf:11.1f} h {gb:8.1f}{mark}")

print()
print(f"  all six missing GLUE tasks, full 9-arm design : {tot_missing:6.1f} h "
      f"= {tot_missing/24:5.1f} days of one L4")
print(f"  all ten GLUE tasks, full 9-arm design         : {tot_full:6.1f} h "
      f"= {tot_full/24:5.1f} days")

print()
print("Same six tasks at a REDUCED design (baseline,B,C,BC; 40 trials + 15 seeds):")
red_tot = sum(220 * (c0 + c1 * passes(t)) / 3600
              for t in ("sst2", "qnli", "qqp", "mnli", "wnli"))
print(f"  {red_tot:.1f} h = {red_tot/24:.1f} days")

print()
print("CONFIRM-ONLY design (replay CoLA-selected configs, 9 arms x 15 seeds, no tuning):")
conf_tot = sum(135 * (c0 + c1 * passes(t)) / 3600
               for t in ("sst2", "qnli", "qqp", "mnli", "wnli"))
print(f"  {conf_tot:.1f} h = {conf_tot/24:.1f} days   <-- cheapest way to cover all 10 GLUE tasks")

print()
print("DECODER multiplier, measured (TinyLlama / BERT): "
      f"cola {DECODER_X['cola']:.1f}x, stsb {DECODER_X['stsb']:.1f}x")
print("Larger backbones scale with hidden size and layer count; Mistral-7B was")
print("never run here, so any Mistral number would be a guess and is not given.")
