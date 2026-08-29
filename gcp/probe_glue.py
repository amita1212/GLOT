from datasets import load_dataset

for name in ["glue", "nyu-mll/glue"]:
    try:
        ds = load_dataset(name, "cola")
        print(f"{name:16s} OK  ->", {k: len(v) for k, v in ds.items()})
    except Exception as e:
        print(f"{name:16s} FAILED -> {type(e).__name__}: {str(e)[:200]}")
