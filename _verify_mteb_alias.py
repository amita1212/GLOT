"""Does mteb.get_task() silently alias a deprecated name to a .v2 revision?

If it does, the name "resolves" but the DATASET is a revised one, which changes
what the numbers mean when comparing against the original paper's Table 3.
Print the requested name beside the resolved task's actual metadata name.
"""
import mteb

NAMES = [
    # original paper Table 3
    "EmotionClassification", "SciFact", "RedditClustering",
    "AskUbuntuDupQuestions", "STS12", "TwitterSemEval2015", "SummEval",
    # currently queued (appendix Table 12 set)
    "Banking77Classification", "STS13", "ArguAna",
    "TwentyNewsgroupsClustering", "SprintDuplicateQuestions",
]

print(f"{'requested':<28}{'resolved':<32}{'':<9}{'type':<16}metric")
for n in NAMES:
    try:
        t = mteb.get_task(n)
        actual = t.metadata.name
        flag = "ALIASED" if actual != n else "same"
        print(f"{n:<28}{actual:<32}{flag:<9}{t.metadata.type:<16}{t.metadata.main_score}")
    except Exception as e:                                    # noqa: BLE001
        print(f"{n:<28}{'<FAILED>':<32}{type(e).__name__}: {e}")
