from datasets import load_dataset
try:
    ds = load_dataset("glue", "cola")
    print("BARE 'glue' OK:", {k: len(v) for k, v in ds.items()})
except Exception as e:
    print("BARE 'glue' FAILED:", type(e).__name__, str(e)[:300])
