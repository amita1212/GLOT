import importlib
import sys

MODS = [
    "torch", "torch_scatter", "torch_sparse", "torch_geometric",
    "transformers", "datasets", "sentence_transformers", "mteb",
    "peft", "wandb", "sklearn", "numpy", "pandas", "tqdm", "accelerate",
]

missing = []
for m in MODS:
    try:
        mod = importlib.import_module(m)
        print(f"{m:24s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:24s} MISSING ({type(e).__name__}: {e})")
        missing.append(m)

try:
    import torch
    print(f"{'cuda.is_available':24s} {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{'gpu':24s} {torch.cuda.get_device_name(0)}")
except Exception as e:
    print("torch check failed:", e)

print()
print("RESULT:", "ALL OK" if not missing else f"MISSING -> {missing}")
sys.exit(1 if missing else 0)
