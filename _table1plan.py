"""Schedule for completing Table 1 with the six arms the user chose.

Arms: baseline, A, B, C, AC, BC  (6 of the 9; AB, ABC and no_graph dropped)
  full design        = 6 * (40 tune + 15 confirm) = 330 runs/task
  confirmation-only  = 6 * 15                     =  90 runs/task

Runtime fit and cache MB/example are both measured -- see _costmodel.py and the
`du -sh data/*` listing on hyperglot-l4.
"""
EPOCHS, CAPPED, SUB = 2, {"qqp", "qnli", "mnli"}, 20000
C0, C1 = 62.6, 5.307e-3
GLUE = {
    "wnli": (635, 71, True), "rte": (2490, 277, True),
    "mrpc": (3668, 408, True), "stsb": (5749, 1500, True),
    "cola": (8551, 1043, False), "sst2": (67349, 872, False),
    "qnli": (104743, 5463, True), "mnli": (392702, 9815 + 9832, True),
    "qqp": (363846, 40430, True),
}
DONE = {"bert": {"cola", "stsb", "mrpc", "rte"},
        "roberta": {"cola", "stsb"}, "tinyllama": {"stsb"}}
XRUN = {"bert": 1.00, "roberta": 1.05, "tinyllama": 3.80}
MB = {"bert": (0.374, 0.748), "roberta": (0.374, 0.748),
      "tinyllama": (0.982, 2.087)}
FULL, CONF = 330, 90


def used(t):
    return min(GLUE[t][0], SUB) if t in CAPPED else GLUE[t][0]


def secs(t, bb):
    _, dev, pair = GLUE[t]
    return (C0 + C1 * EPOCHS * (used(t) + dev) * (2 if pair else 1)) * XRUN[bb]


def gb(t, bb):
    _, dev, pair = GLUE[t]
    return (used(t) + dev) * MB[bb][1 if pair else 0] / 1024


jobs = [(bb, t) for bb in ("bert", "roberta", "tinyllama")
        for t in GLUE if t not in DONE[bb]]
jobs.sort(key=lambda j: CONF * secs(j[1], j[0]))

print("SCHEDULE  (cheapest first; cache built then DELETED before the next task)")
print("-" * 84)
print(f"{'#':>2} {'backbone':10s} {'task':5s} {'s/run':>6s} {'full':>8s} "
      f"{'confirm':>8s} {'peak GB':>8s} {'cum conf':>9s}")
cf = cc = 0.0
for i, (bb, t) in enumerate(jobs, 1):
    s = secs(t, bb)
    f, c = FULL * s / 3600, CONF * s / 3600
    cf += f; cc += c
    flag = "  <-- exceeds 118 GB free" if gb(t, bb) > 118 else ""
    print(f"{i:2d} {bb:10s} {t:5s} {s:6.0f} {f:7.1f}h {c:7.1f}h "
          f"{gb(t, bb):7.1f} {cc:8.1f}h{flag}")

print("-" * 84)
print(f"   TOTAL, 6 arms          full {cf:7.0f} h = {cf/24:5.1f} d")
print(f"                       confirm {cc:7.0f} h = {cc/24:5.1f} d")
print(f"   peak disk with delete-between-tasks: {max(gb(t,b) for b,t in jobs):.0f} GB")
print()
print("Split by backbone (confirmation-only):")
for bb in ("bert", "roberta", "tinyllama"):
    h = sum(CONF * secs(t, bb) / 3600 for b, t in jobs if b == bb for _ in [0])
    h = sum(CONF * secs(t, bb) / 3600 for b, t in jobs if b == bb)
    print(f"   {bb:10s} {h:6.0f} h = {h/24:4.1f} d")
print()
print("If TinyLlama qqp+mnli+sst2 are dropped (the three that need >65 GB):")
skip = {("tinyllama", "qqp"), ("tinyllama", "mnli"), ("tinyllama", "sst2")}
h = sum(CONF * secs(t, b) / 3600 for b, t in jobs if (b, t) not in skip)
g = max(gb(t, b) for b, t in jobs if (b, t) not in skip)
print(f"   confirm {h:.0f} h = {h/24:.1f} d ; peak disk {g:.0f} GB")
