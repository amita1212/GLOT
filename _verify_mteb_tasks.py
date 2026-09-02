"""Verify the original GLOT paper's Table 3 MTEB task names against the mteb
version actually installed on this machine, and report each task's main metric.

Rationale: the queued MTEB item used the ORIGINAL PAPER'S APPENDIX Table 12
task list, and two of those names do not exist in mteb 2.x at all (they were
renamed to .v2 revisions). Guessing names off-line is how that defect got in,
so this resolves every name against the installed registry instead.

CPU only -- safe to run while a GPU campaign is in flight.
"""
import mteb

print("mteb version:", getattr(mteb, "__version__", "?"))

# --- discover how to enumerate task names in THIS version -------------------
names = set()
for attempt in ("registry", "get_tasks"):
    try:
        if attempt == "registry":
            from mteb.overview import TASKS_REGISTRY  # type: ignore
            names = set(TASKS_REGISTRY.keys())
        else:
            if names:
                break
            names = {t.metadata.name for t in mteb.get_tasks()}
        if names:
            print(f"enumerated {len(names)} task names via {attempt}")
            break
    except Exception as e:                                    # noqa: BLE001
        print(f"  ({attempt} unavailable: {type(e).__name__})")

# Table 3 of the original paper, as printed there.
TABLE3 = [
    "EmotionClassification",
    "SciFact",
    "RedditClustering",
    "AskUbuntuDupQuestions",
    "STS12",
    "TwitterSemEval2015",
    "SummEval",
]
# What the queue currently asks for (appendix Table 12 set).
QUEUED = [
    "Banking77Classification", "STS12", "STS13", "SciFact",
    "ArguAna", "TwentyNewsgroupsClustering", "SprintDuplicateQuestions",
]


def resolve(name):
    """Return (resolved_name, main_metric) or (None, reason)."""
    for cand in (name, name + ".v2", name + "Summarization.v2"):
        try:
            t = mteb.get_task(cand)
            return cand, t.metadata.main_score
        except Exception:                                     # noqa: BLE001
            continue
    near = sorted(n for n in names if n.lower().startswith(name.lower()[:8]))
    return None, f"NOT FOUND; near: {near[:6]}"


for label, lst in (("TABLE 3 (target)", TABLE3), ("QUEUED (current)", QUEUED)):
    print(f"\n=== {label} ===")
    for n in lst:
        got, info = resolve(n)
        if got:
            flag = "exact" if got == n else f"RENAMED -> {got}"
            print(f"  {n:<26} {flag:<34} metric={info}")
        else:
            print(f"  {n:<26} {info}")
