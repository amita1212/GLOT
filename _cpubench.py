"""Measure whether the IMDB pooler-training phase is viable on CPU.

The backbone is frozen and its states are cached to disk (precompute_hidden_states),
so a CPU-only worker would only ever run THIS: pooler fwd+bwd on cached
(batch, 512, 768) hidden states. That is what we time here.

Synthetic inputs are legitimate because cost depends on tensor shape and graph
density, not on the values -- and we pin density via tau_quantile.
"""
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import GLOT  # noqa: E402

SEQ = 512          # IMDB max_length
DIM = 768          # BERT-base hidden
BATCH = 32
WARMUP = 2
ITERS = 6

# Baseline CoLA-selected configuration (Table: Confirmed configurations).
CFG = dict(
    in_dim=DIM, hidden_dim=256, num_layers=4, jk_mode="max",
    conv="gat", adjacency="threshold", tau_quantile=0.10,
    use_edge_weight=True,
)


def make_batch(device):
    torch.manual_seed(0)
    hidden = torch.randn(BATCH, SEQ, DIM, device=device)
    mask = torch.ones(BATCH, SEQ, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (BATCH,), device=device)
    return hidden, mask, labels


def time_step(device_str, threads=None):
    if threads is not None:
        torch.set_num_threads(threads)
    device = torch.device(device_str)
    pooler = GLOT(device=device, **CFG).to(device)
    out_dim = CFG["hidden_dim"] * CFG["num_layers"] if CFG["jk_mode"] == "cat" else CFG["hidden_dim"]
    head = nn.Linear(out_dim, 2).to(device)
    opt = torch.optim.Adam(list(pooler.parameters()) + list(head.parameters()), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    hidden, mask, labels = make_batch(device)

    def one():
        opt.zero_grad(set_to_none=True)
        z = pooler(hidden, mask)
        loss = lossf(head(z), labels)
        loss.backward()
        opt.step()

    for _ in range(WARMUP):
        one()
    if device_str == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        one()
    if device_str == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS


def main():
    print(f"host cores={os.cpu_count()}  loadavg={os.getloadavg()}")
    print(f"shape: batch={BATCH} seq={SEQ} dim={DIM}  cfg={CFG}\n")

    results = {}
    for th in (1, 2, 4):
        try:
            dt = time_step("cpu", threads=th)
            results[f"cpu-{th}t"] = dt
            print(f"CPU  {th} thread(s): {dt*1000:9.1f} ms/batch")
        except Exception as e:  # noqa: BLE001
            print(f"CPU  {th} thread(s): FAILED {type(e).__name__}: {e}")

    if torch.cuda.is_available():
        try:
            dt = time_step("cuda")
            results["cuda"] = dt
            print(f"GPU (L4, shared): {dt*1000:9.1f} ms/batch")
        except Exception as e:  # noqa: BLE001
            print(f"GPU: FAILED {type(e).__name__}: {e}")

    # Projections for the real campaign.
    train_n, epochs = 22500, 2
    batches = (train_n + BATCH - 1) // BATCH * epochs
    runs = 4 * 5  # arms baseline/A/C/AC x 5 seeds, confirmation only
    print(f"\nIMDB train={train_n} epochs={epochs} -> {batches} batches/run, {runs} runs")
    for k, dt in results.items():
        per_run_h = batches * dt / 3600
        print(f"  {k:10s}: {per_run_h:6.2f} h/run   {per_run_h*runs:7.1f} h for {runs} runs")

    cache_gb = 50000 * SEQ * DIM * 4 / 1e9
    print(f"\nIMDB cache (train+test, fp32): {cache_gb:.0f} GB")


if __name__ == "__main__":
    main()
