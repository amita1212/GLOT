"""Cache size per GLUE task, from MEASURED bytes-per-example, not a formula.

_costmodel.py's cache column is wrong twice over: it assumes 2 bytes per
activation (the caches are fp32, so 4) and it omits the dev split (which is
cached too, and for QQP the dev set is TWICE the capped train set). Both errors
push the same way, so the figure it prints is roughly 4x too small.

Calibration against real directories on the BGU machine, RoBERTa/RTE:
    train 2,490 pairs -> 1.9 GB observed ; 2,490 * 0.748 MB = 1.86 GB
    val     277 pairs -> 209 MB observed ;   277 * 0.748 MB = 207 MB
so 0.748 MB per PAIR example and 0.374 MB per SINGLE example at max_len 128
reproduce reality to within 3%.
"""
MB_SINGLE = 0.374          # measured, max_len 128, fp32
MB_PAIR = 0.748

EPOCHS, SUBSAMPLE = 2, 20000
CAPPED = {"qqp", "qnli", "mnli"}          # original's B.3; SST-2 is NOT capped

GLUE = {                                   # n_train, n_dev, is_pair
    "rte":  (2490,   277,          True),
    "mrpc": (3668,   408,          True),
    "stsb": (5749,   1500,         True),
    "cola": (8551,   1043,         False),
    "sst2": (67349,  872,          False),
    "qnli": (104743, 5463,         True),
    "qqp":  (363846, 40430,        True),
    "mnli": (392702, 9815 + 9832,  True),
}

print(f"{'task':6s} {'train used':>11s} {'dev':>8s} {'MB/ex':>7s} {'cache GB':>9s}")
print("-" * 48)
gb = {}
for t, (n_tr, n_dev, pair) in GLUE.items():
    used = min(n_tr, SUBSAMPLE) if t in CAPPED else n_tr
    per = MB_PAIR if pair else MB_SINGLE
    g = (used + n_dev) * per / 1000.0
    gb[t] = g
    print(f"{t:6s} {used:11,d} {n_dev:8,d} {per:7.3f} {g:9.1f}")

print()
print(f"  QQP + MNLI together      : {gb['qqp'] + gb['mnli']:.1f} GB"
      "   (paper says 149 GB)")
print(f"  SST-2 alone              : {gb['sst2']:.1f} GB")
print(f"  QNLI alone               : {gb['qnli']:.1f} GB")
print(f"  all four uncovered tasks : "
      f"{gb['sst2'] + gb['qnli'] + gb['qqp'] + gb['mnli']:.1f} GB")

# SST-2 under the reduced design the paper actually describes:
# four arms, confirmation seeds only, reusing the CoLA-selected configuration.
S_SST2 = 787.0     # fitted s/run from _costmodel.py
for label, runs in (("4 arms x 15 confirm seeds", 4 * 15),
                    ("4 arms x (40 trials + 15)", 4 * 55),
                    ("9 arms x (40 trials + 15)", 9 * 55)):
    print(f"  SST-2, {label:26s}: {runs:4d} runs = {runs * S_SST2 / 3600:6.1f} h")
