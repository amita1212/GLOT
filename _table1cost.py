"""What would 'complete Table 1' actually cost?

Runtime model from _costmodel.py (fitted on measured BERT runs).
Cache model from MEASURED cache directories on the VM:
    bert cola  train  8,551 single -> 3.2 GB  => 0.374 MB/example
    bert sts   train  5,749 pair   -> 4.3 GB  => 0.748 MB/example
    bert mrpc  train  3,668 pair   -> 2.7 GB  => 0.736 MB/example
    tinyllama cola train 8,551 single -> 8.4 GB => 0.982 MB/example
    tinyllama sts  train 5,749 pair   -> 12  GB => 2.087 MB/example
"""
import numpy as np

EPOCHS = 2
CAPPED = {"qqp", "qnli", "mnli"}
SUB = 20000

# n_train, n_dev, is_pair
GLUE = {
    "cola": (8551, 1043, False), "sst2": (67349, 872, False),
    "stsb": (5749, 1500, True), "mrpc": (3668, 408, True),
    "qqp": (363846, 40430, True), "mnli": (392702, 9815 + 9832, True),
    "qnli": (104743, 5463, True), "rte": (2490, 277, True),
    "wnli": (635, 71, True),
}
DONE = {"bert": {"cola", "stsb", "mrpc", "rte"},
        "roberta": {"cola", "stsb"},
        "tinyllama": {"stsb"}}
# runtime multiplier vs BERT, measured
XRUN = {"bert": 1.00, "roberta": 1.05, "tinyllama": 3.80}
# MB per cached example, measured
MB = {"bert": (0.374, 0.748), "roberta": (0.374, 0.748),
      "tinyllama": (0.982, 2.087)}

C0, C1 = 62.6, 5.307e-3   # from _costmodel.py fit


def used(t):
    return min(GLUE[t][0], SUB) if t in CAPPED else GLUE[t][0]


def secs(t, bb):
    _, dev, pair = GLUE[t]
    p = EPOCHS * (used(t) + dev) * (2 if pair else 1)
    return (C0 + C1 * p) * XRUN[bb]


def cache_gb(t, bb):
    _, dev, pair = GLUE[t]
    per = MB[bb][1 if pair else 0]
    return (used(t) + dev) * per / 1024


FULL, CONF = 495, 135     # runs per task: 9 arms x (40 tune + 15 seeds) / 15 seeds
print(f"{'backbone':10s} {'task':6s} {'s/run':>7s} {'full 495':>9s} "
      f"{'conf 135':>9s} {'cache GB':>9s}")
tot = {}
for bb in ("bert", "roberta", "tinyllama"):
    tf = tc = tg = 0.0
    for t in GLUE:
        if t in DONE[bb]:
            continue
        s = secs(t, bb)
        f, c, g = FULL * s / 3600, CONF * s / 3600, cache_gb(t, bb)
        tf += f; tc += c; tg += g
        print(f"{bb:10s} {t:6s} {s:7.0f} {f:8.1f}h {c:8.1f}h {g:8.1f}")
    tot[bb] = (tf, tc, tg)
    print(f"{bb:10s} {'TOTAL':6s} {'':7s} {tf:8.1f}h {tc:8.1f}h {tg:8.1f}")
    print()

F = sum(v[0] for v in tot.values())
C = sum(v[1] for v in tot.values())
G = sum(v[2] for v in tot.values())
print("=" * 64)
print(f"ALL THREE BACKBONES, every task not yet run:")
print(f"  full 9-arm design (40 tuning trials + 15 seeds): {F:7.0f} h = {F/24:5.1f} days")
print(f"  confirmation-only (replay tuned configs, 15 s):  {C:7.0f} h = {C/24:5.1f} days")
print(f"  NEW hidden-state cache required:                 {G:7.0f} GB")
print(f"  free disk on hyperglot-l4 right now:                 118 GB")
print()
print("Largest single-task cache (sets the peak if we delete between tasks):")
peak = max((cache_gb(t, bb), bb, t) for bb in tot for t in GLUE if t not in DONE[bb])
print(f"  {peak[1]} {peak[2]}: {peak[0]:.0f} GB")
