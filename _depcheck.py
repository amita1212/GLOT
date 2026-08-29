import importlib
for m in ["torch","transformers","datasets","torch_geometric","sklearn","scipy","numpy","tqdm","geoopt","torch_scatter"]:
    try:
        mod = importlib.import_module(m)
        print(m.ljust(20), getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m.ljust(20), "MISSING", type(e).__name__)
import torch
print("cuda available:", torch.cuda.is_available())
