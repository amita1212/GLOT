import os
import json
import csv
import time
import random
import argparse
import string
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from hyperbolic_layers import HyperbolicGCNConv, HyperbolicGATConv, hyperbolic_readout
from hyperbolic_graph import HyperbolicGraphConfig, build_pyg_graphs_hyper

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

from transformers import AutoTokenizer, AutoConfig, AutoModel
from datasets import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch, dense_to_sparse, softmax
from torch_scatter import scatter_add

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import numpy as np
from matplotlib.patches import Rectangle



HF_TOKEN = "<>"
# -------------------------
# Boilerplate & Utilities from your original script
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_decoder_like(config):
    if getattr(config, "is_decoder", False): return True
    mt = getattr(config, "model_type", "") or ""
    if mt in {"gpt2", "gptj", "gpt_neo", "llama", "mpt", "gemma", "falcon"}: return True
    arch = getattr(config, "architectures", None)
    if arch and any(("CausalLM" in a) for a in arch): return True
    return False

def masked_mean(x, mask, dim):
    mask = mask.to(x.dtype)
    s = (x * mask.unsqueeze(-1)).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1e-6).unsqueeze(-1)
    return s / denom

def masked_max(x, mask, dim):
    very_small = torch.finfo(x.dtype).min
    mask_exp = mask.unsqueeze(-1).to(x.dtype)
    x_masked = x * mask_exp + (1 - mask_exp) * very_small
    return x_masked.max(dim=dim).values

def accuracy(preds, labels):
    return float((preds == labels).mean())

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Backbone:
    tokenizer: AutoTokenizer
    model: AutoModel
    config: AutoConfig
    is_decoder: bool
    model_name: str

def load_backbone(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    is_dec = is_decoder_like(config)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return Backbone(tokenizer=tokenizer, model=model, config=config, is_decoder=is_dec, model_name=model_name_or_path)

def forward_hidden(backbone: Backbone, batch_inputs):
    with torch.no_grad():
        outputs = backbone.model(**batch_inputs, return_dict=True, output_hidden_states=True)
        hidden = outputs.hidden_states[-1] if backbone.is_decoder else outputs.last_hidden_state
    return hidden, batch_inputs["attention_mask"]

# -------------------------
# Pooling modules (Copied from your script)
# -------------------------

class MeanPooler(nn.Module):
    def forward(self, hidden, attention_mask): return masked_mean(hidden, attention_mask, dim=1)

class MaxPooler(nn.Module):
    def forward(self, hidden, attention_mask): return masked_max(hidden, attention_mask, dim=1)

class CLSPooler(nn.Module):
    def __init__(self, use_last_token_for_decoder=True):
        super().__init__()
        self.use_last_token_for_decoder = use_last_token_for_decoder
    def forward(self, hidden, attention_mask, is_decoder):
        if is_decoder and self.use_last_token_for_decoder:
            lengths = attention_mask.sum(dim=1)
            idx = (lengths - 1).clamp_min(0).long()
            b_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[b_idx, idx]
        else:
            return hidden[:, 0, :]

class AdaPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    def forward(self, hidden_states, mask):
        scores = self.score_layer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        return torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)


def pairwise_cosine_single(h, mask):
    """
    h: (L, d)  single sequence (no batch)
    mask: (L,)  1=valid, 0=pad
    returns sim: (L, L) in [-1, 1], pads zeroed out
    """
    valid_idx = mask.nonzero(as_tuple=True)[0]
    h = h[valid_idx]  # (n, d)
    sim = F.cosine_similarity(h.unsqueeze(1), h.unsqueeze(0), dim=-1)
    return sim

def _threshold_edges(sim, tau):
    """
    Binary threshold: edges where sim >= tau, undirected, no self loops.
    """
    A = (sim > tau).float()
    edge_index, edge_weights = dense_to_sparse(A)
    return edge_index, edge_weights

@torch.no_grad()
def build_pyg_graphs(
    hidden,
    attention_mask,
    adjacency="threshold",
    tau=0.3,
    device=None,
):
    assert hidden.dim() == 3 and attention_mask.dim() == 2, "Bad input shapes"
    B, L, d = hidden.shape
    device = device or hidden.device
    graphs: List[Data] = []

    for b in range(B):
        mask_b = attention_mask[b].to(dtype=torch.bool)
        x_b = hidden[b, mask_b]  # (n, d)
        token_idx = torch.arange(L, device=device)[mask_b]  # (n,)
        n = x_b.size(0)

        sim = pairwise_cosine_single(x_b, mask_b)  # (n, n)

        if adjacency == "threshold":
            edge_index, edge_weight = _threshold_edges(sim, tau=tau)
            data = Data(x=x_b, edge_index=edge_index, edge_attr=edge_weight).to(device)
        else:
            raise ValueError(f"Unknown adjacency: {adjacency}")

        data.token_idx = token_idx
        graphs.append(data)

    return Batch.from_data_list(graphs)

class GLOT(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_layers=2,
        jk_mode="cat",  
        conv=GATConv,
        adjacency="threshold",
        tau=0.3,
        device=None,
        # --- Stage A (HyperGLOT) options; defaults preserve original GLOT ---
        graph_metric="cosine",   # {"cosine", "poincare"}
        curvature=1.0,
        rho=1.0,
        knn_k=8,
        feature_norm=False,
        # --- Stage B / C (HyperGLOT) options; defaults preserve original GLOT ---
        hyperbolic_gnn=False,
        hyperbolic_readout=False,
        hyp_gnn_type="gcn",       # Stage C conv when hyperbolic_gnn: {"gcn", "gat"}
        readout_clip=0.0,         # Stage B: clip feature norm before exp map (0=off)
        readout_scale=False,      # Stage B: learnable input scale before the exp map
        learnable_curvature=False,# trainable Poincare ball curvature
        gnn_input_clip=0.0,       # Stage C: clip token norm before the entry exp map (0=off)
        gnn_input_scale=False,    # Stage C: learnable input scale before the entry exp map
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.adjacency = adjacency
        self.tau = tau
        self.device = device
        self.graph_metric = graph_metric
        self.curvature = curvature
        self.rho = rho
        self.knn_k = knn_k
        self.feature_norm = feature_norm
        self.hyperbolic_gnn = bool(hyperbolic_gnn)
        self.hyperbolic_readout = bool(hyperbolic_readout)
        self.hyp_gnn_type = str(hyp_gnn_type)
        self.readout_clip = float(readout_clip)
        self.learnable_curvature = bool(learnable_curvature)
        self.gnn_input_clip = float(gnn_input_clip)

        self.ball = None
        if self.hyperbolic_gnn or self.hyperbolic_readout:
            self.ball = geoopt.PoincareBall(c=self.curvature, learnable=self.learnable_curvature)

        # Optional learnable input scales (fix the boundary-saturation failure).
        self.readout_scale = None
        if self.hyperbolic_readout and bool(readout_scale):
            self.readout_scale = nn.Parameter(torch.tensor(0.1))
        self.gnn_input_scale = None
        if self.hyperbolic_gnn and bool(gnn_input_scale):
            self.gnn_input_scale = nn.Parameter(torch.tensor(0.1))

        # Build conv stack
        self.convs = nn.ModuleList()
        last_dim = in_dim
        for _ in range(num_layers):
            if self.hyperbolic_gnn:
                if self.hyp_gnn_type == "gat":
                    layer = HyperbolicGATConv(last_dim, hidden_dim, self.ball)
                else:
                    layer = HyperbolicGCNConv(last_dim, hidden_dim, self.ball)
            else:
                layer = conv(last_dim, hidden_dim, edge_dim=1)
            self.convs.append(layer)
            last_dim = hidden_dim

        if jk_mode == "cat":
            self.out_dim = in_dim + num_layers * hidden_dim
        else:
            self.out_dim = hidden_dim
        
        self.score_layer = nn.Sequential(
            nn.Linear(self.out_dim, max(128, self.out_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(128, self.out_dim // 2), 1)
        )
        

    def forward(self, hidden, attention_mask):
        """
        hidden: (B, L, d)  token features
        attention_mask: (B, L)  1=valid, 0=pad

        Returns:
          z: (B, D) pooled embeddings
        """
        device = self.device or hidden.device
        B, L, d = hidden.shape

        # Same four-way routing as main.py: any poincare metric OR any knn
        # adjacency uses the Stage A builder; cosine+threshold stays original.
        use_hyperbolic_builder = (self.graph_metric == "poincare") or (self.adjacency == "knn")
        if use_hyperbolic_builder:
            hyperbolic_config = HyperbolicGraphConfig(
                graph_metric=self.graph_metric,
                adjacency=self.adjacency,
                tau=self.tau,
                rho=self.rho,
                k=self.knn_k,
                curvature=self.curvature,
                feature_norm=self.feature_norm,
            )
            batch = build_pyg_graphs_hyper(
                hidden, attention_mask, hyperbolic_config, device=device
            )
        else:
            batch = build_pyg_graphs(
                hidden, attention_mask, adjacency=self.adjacency,
                tau=self.tau, device=device
            )

        batch = batch.to(device)
        x, edge_index = batch.x, batch.edge_index
        edge_weight = getattr(batch, "edge_attr", None)

        h_list = [x]
        if self.hyperbolic_gnn:
            # Stage C: hyperbolic Token-GNN (see hyperbolic_layers.py). Stabilise
            # the entry lift so raw token norms don't saturate expmap0 at the
            # ball boundary (mirrors the Stage B readout fix).
            x_in = x
            if self.gnn_input_scale is not None:
                x_in = x_in * self.gnn_input_scale
            if self.gnn_input_clip and self.gnn_input_clip > 0:
                norm = x_in.norm(dim=-1, keepdim=True).clamp_min(1e-5)
                x_in = x_in * torch.clamp(self.gnn_input_clip / norm, max=1.0)
            h_ball = self.ball.projx(self.ball.expmap0(x_in))
            for conv in self.convs:
                h_ball = conv(h_ball, edge_index, edge_weight)
                h_list.append(self.ball.logmap0(h_ball))
        else:
            h = x
            for conv in self.convs:
                h = conv(h, edge_index, edge_attr=edge_weight)
                h = F.relu(h)
                h_list.append(h)

        if self.jk_mode == "cat":
            h_all = torch.cat(h_list, dim=-1)
        elif self.jk_mode == "max":
            h_all = torch.stack(h_list[1:], dim=-1).max(dim=-1).values
        elif self.jk_mode == "mean":
            h_all = torch.stack(h_list[1:], dim=-1).mean(dim=-1)
        else:
            raise ValueError("Unknown JK mode") 

        scores = self.score_layer(h_all).squeeze(-1)
        weights = softmax(scores, batch.batch)
        if self.hyperbolic_readout:
            # Stage B: Poincare gyro-midpoint readout (see hyperbolic_layers.py).
            readout_c = self.ball.c if self.learnable_curvature else self.curvature
            pooled = hyperbolic_readout(
                h_all, weights, batch.batch, self.ball, readout_c,
                num_graphs=int(batch.batch.max().item()) + 1,
                scale=self.readout_scale, clip=self.readout_clip,
            )
        else:
            pooled = scatter_add(weights.unsqueeze(-1) * h_all, batch.batch, dim=0)
        
        return pooled

class SingleClassifier(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(dim, num_labels)
    def forward(self, z): return self.classifier(z)

def build_pooler(name: str, hidden_size: int, args) -> nn.Module:
    name = name.lower()
    if name == "mean": return MeanPooler()
    elif name == "max": return MaxPooler()
    elif name == "cls": return CLSPooler(use_last_token_for_decoder=args.decoder_cls_last_token)
    elif name == "adapool": return AdaPool(in_dim=hidden_size, hidden_dim=args.scorer_hidden)
    elif name == "glot":
        return GLOT(
            in_dim=hidden_size, hidden_dim=args.gat_hidden_dim,
            num_layers=args.num_layers, jk_mode=args.jk_mode, tau=args.tau,
            adjacency=getattr(args, "graph_adj", "threshold"),
            graph_metric=getattr(args, "graph_metric", "cosine"),
            curvature=getattr(args, "curvature", 1.0),
            rho=getattr(args, "rho", 1.0),
            knn_k=getattr(args, "knn_k", 8),
            feature_norm=bool(getattr(args, "feature_norm", 0)),
            hyperbolic_gnn=bool(getattr(args, "hyperbolic_gnn", 0)),
            hyperbolic_readout=bool(getattr(args, "hyperbolic_readout", 0)),
            hyp_gnn_type=getattr(args, "hyp_gnn_type", "gcn"),
            readout_clip=float(getattr(args, "readout_clip", 0.0)),
            readout_scale=bool(getattr(args, "readout_scale", 0)),
            learnable_curvature=bool(getattr(args, "learnable_curvature", 0)),
            gnn_input_clip=float(getattr(args, "gnn_input_clip", 0.0)),
            gnn_input_scale=bool(getattr(args, "gnn_input_scale", 0)))
    else: raise ValueError(f"Unknown pooling method: {name}")

def pool_hidden(pooler, hidden, mask, is_decoder, name):
    return pooler(hidden, mask, is_decoder) if isinstance(pooler, CLSPooler) else pooler(hidden, mask)

# -------------------------
#  Data Generation
# -------------------------

# Using a fixed list of common words for reproducibility as the "haystack"
NOISE_WORDS = "the of and to a in for is on that by this with i you it not or be are from at as your all have new more an was we will home can us about if page my has search free but our one other do no information time they site he up may what which their news out use any there see only so his when contact here business who web also now help get pm view online first am been would how were me services some these click its like service x than find date top yet".split()

def generate_dataset(
    num_samples: int,
    seq_len: int,
    distractor_ratio: float,
    signal_position: str,
    relational_distance: int,
) -> List[dict]:
    
    dataset = []
    TARGET_NOUNS = ['keys', 'reports', 'files', 'tickets', 'documents', 'alerts']
    MODIFIERS = ['not', 'never', 'without', 'excluding']
    CONTEXTS = ['the delivery contains', 'the folder includes', 'in the box are']

    for _ in range(num_samples):
        # 1. Choose signal and label
        is_positive = random.choice([True, False])
        target_noun = random.choice(TARGET_NOUNS)
        context = random.choice(CONTEXTS)
        
        # 2. Construct the signal phrase (the "needle")
        if is_positive:
            signal_phrase_words = [context, target_noun]
            label = 1
        else:
            modifier = random.choice(MODIFIERS)
            distractor_words_between = random.choices(NOISE_WORDS, k=relational_distance)
            signal_phrase_words = [context, modifier] + distractor_words_between + [target_noun]
            label = 0
        
        # 3. Construct the noise (the "haystack")
        num_signal_words = len(signal_phrase_words)
        num_noise_words = int(seq_len * distractor_ratio)
        num_total_words = num_noise_words + num_signal_words
        if num_total_words > seq_len: # adjust if ratio makes it too long
            num_noise_words -= (num_total_words - seq_len)
            
        noise = random.choices(NOISE_WORDS, k=num_noise_words)
        
        # 4. Inject needle into haystack
        signal_start_idx = 0
        if signal_position == 'middle':
            signal_start_idx = len(noise) // 2
        elif signal_position == 'end':
            signal_start_idx = len(noise)
        elif signal_position == 'random':
            signal_start_idx = random.randint(0, len(noise))
            
        final_words = noise[:signal_start_idx] + signal_phrase_words + noise[signal_start_idx:]
        final_words = final_words[:seq_len] # Ensure exact length
        
        dataset.append({
            "text": " ".join(final_words),
            "label": label,
            "query": f"Are {target_noun} present?" # For context, not used by model
        })
        
    return dataset

# -------------------------
# Main Experiment Logic
# -------------------------

def _arm_name(args) -> str:
    """Ablation-arm label from the three HyperGLOT switches (A/B/C)."""
    if getattr(args, "arm", ""):
        return args.arm
    a = (getattr(args, "graph_metric", "cosine") == "poincare")
    b = bool(getattr(args, "hyperbolic_readout", 0))
    c = bool(getattr(args, "hyperbolic_gnn", 0))
    if not (a or b or c):
        return "baseline"
    if a and b and c:
        return "ABC"
    return "".join([n for n, f in (("A", a), ("B", b), ("C", c)) if f])


def _append_stress_csv(args, best_acc: float):
    """Append the negation stress-test result (accuracy) to a CSV."""
    path = args.results_csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": getattr(args, "run_tag", ""),
        "model": args.model_name_or_path,
        "task": "stress",
        "arm": _arm_name(args),
        "pooling": args.pooling_method,
        "graph_metric": args.graph_metric,
        "graph_adj": args.graph_adj,
        "hyperbolic_gnn": int(getattr(args, "hyperbolic_gnn", 0)),
        "hyperbolic_readout": int(getattr(args, "hyperbolic_readout", 0)),
        "distractor_ratio": args.distractor_ratio,
        "relational_distance": args.relational_distance,
        "signal_position": args.signal_position,
        "tau": args.tau,
        "rho": args.rho,
        "curvature": args.curvature,
        "knn_k": args.knn_k,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "acc": round(float(best_acc), 6),
    }
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[results] appended stress row -> {path}", flush=True)


def _git_push_results(path: str, message: str, branch: str = "", remote: str = "origin"):
    """Stage, commit and push the results CSV. Never raises (git problems must
    not abort a run)."""
    try:
        repo_dir = os.path.dirname(os.path.abspath(path))
        add = subprocess.run(["git", "add", os.path.abspath(path)],
                             cwd=repo_dir, capture_output=True, text=True)
        if add.returncode != 0:
            print(f"[git] add failed: {add.stderr.strip()}", flush=True)
            return
        commit = subprocess.run(["git", "commit", "-m", message],
                                cwd=repo_dir, capture_output=True, text=True)
        if commit.returncode != 0:
            if "nothing to commit" not in (commit.stdout + commit.stderr):
                print(f"[git] commit failed: {(commit.stdout + commit.stderr).strip()}", flush=True)
            return
        push_cmd = ["git", "push", remote] + ([branch] if branch else [])
        push = subprocess.run(push_cmd, cwd=repo_dir, capture_output=True, text=True)
        if push.returncode != 0:
            print(f"[git] push failed: {(push.stdout + push.stderr).strip()}", flush=True)
        else:
            print(f"[git] pushed results -> {remote} {branch}".rstrip(), flush=True)
    except Exception as exc:
        print(f"[git] skipped ({exc})", flush=True)


def run_experiment(backbone: Backbone, args, device):

    run = wandb.init(project="GLOT")
    wandb.config.update(args)

    print("--- Starting Experiment ---")
    print(f"Model: {args.model_name_or_path}, Pooling: {args.pooling_method}")
    print(f"Params: Distractor Ratio={args.distractor_ratio}, Relational Distance={args.relational_distance}, Signal Position={args.signal_position}")
    print("-" * 60)
    
    # 1. Generate Data
    print("Generating synthetic dataset...")
    train_data = generate_dataset(args.num_train_samples, args.max_length, args.distractor_ratio, args.signal_position, args.relational_distance)
    eval_data = generate_dataset(args.num_eval_samples, args.max_length, args.distractor_ratio, args.signal_position, args.relational_distance)
    
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(eval_data)
    
    # 2. Setup Model, Dataloaders, Optimizer
    pooler = build_pooler(args.pooling_method, backbone.config.hidden_size, args).to(device)
    
    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        labels = [int(ex["label"]) for ex in examples]
        batch = backbone.tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
        return batch

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Determine pooled dimension to setup classifier
    sample_batch = next(iter(val_loader))
    hidden, mask = forward_hidden(backbone, {"input_ids": sample_batch["input_ids"], "attention_mask": sample_batch["attention_mask"]})
    z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, args.pooling_method)
    dim = z.size(-1)

    classifier = SingleClassifier(dim=dim, num_labels=2).to(device)
    
    params = list(classifier.parameters())
    if any(p.requires_grad for p in pooler.parameters()):
        params.extend(pooler.parameters())
        
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # 3. Training Loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        pooler.train()
        classifier.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Training"):
            hidden, mask = forward_hidden(backbone, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
            z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, args.pooling_method)
            logits = classifier(z)
            loss = F.cross_entropy(logits, batch["labels"])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({
                "loss/step": loss
            })

        # 4. Evaluation Loop
        pooler.eval()
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Evaluating"):
                hidden, mask = forward_hidden(backbone, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
                z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, args.pooling_method)
                logits = classifier(z)
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(batch["labels"].cpu().numpy())
                
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = accuracy(all_preds, all_labels)
        best_acc = max(best_acc, acc)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Eval Accuracy={acc:.4f}")
        wandb.log({
            "metrics/acc": acc
        })

    print("-" * 60)
    print(f"Experiment Finished. Best Validation Accuracy: {best_acc:.4f}")
    print("-" * 60)

    # Persist the stress-test result (accuracy at this distractor ratio) so the
    # negation "needle in a haystack" sweep from the article is saved to CSV.
    if getattr(args, "results_csv", ""):
        _append_stress_csv(args, best_acc)
        # Push the CSV to GitHub after each run so results survive a crash.
        if getattr(args, "git_push", 0):
            _git_push_results(
                args.results_csv,
                message=f"stress: {args.model_name_or_path} {_arm_name(args)} d{args.distractor_ratio} seed{args.seed}",
                branch=getattr(args, "git_branch", ""),
                remote=getattr(args, "git_remote", "origin"),
            )
    print("RESULT_JSON " + json.dumps({
        "model": args.model_name_or_path, "task": "stress",
        "arm": _arm_name(args), "graph_metric": args.graph_metric,
        "graph_adj": args.graph_adj, "hyperbolic_gnn": int(getattr(args, "hyperbolic_gnn", 0)),
        "hyperbolic_readout": int(getattr(args, "hyperbolic_readout", 0)),
        "distractor_ratio": args.distractor_ratio, "seed": args.seed,
        "metrics": {"acc": best_acc},
    }, default=str), flush=True)


    INPUT_SENTENCE = eval_data[0]['text']
    all_vectors, all_labels, num_tokens = get_augmented_data(backbone.model_name, INPUT_SENTENCE, device)
    sample_batch = next(iter(val_loader))
    hidden, mask = forward_hidden(backbone, {"input_ids": sample_batch["input_ids"], "attention_mask": sample_batch["attention_mask"]})
    z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, args.pooling_method)
    print(z.shape)
    # The pooled GLOT vector only lives in the same space as the raw token
    # embeddings when its dimension matches (e.g. mean/max pooling). With a
    # projection head (jk_mode=cat/proj) the pooled dim differs, so appending it
    # to the token-similarity matrix would crash. Only include it when dims align.
    if z.size(-1) == all_vectors.size(-1):
        all_labels.append("[Token-GNN]")
        all_vectors = torch.cat(
            [all_vectors.detach().cpu(), z[0].unsqueeze(0).detach().cpu()], dim=0
        ).cpu().numpy()
    else:
        print(f"[stress] pooled dim {z.size(-1)} != token dim {all_vectors.size(-1)}; "
              f"omitting pooled vector from the similarity figure.")
        all_vectors = all_vectors.detach().cpu().numpy()
    similarity_matrix = compute_similarity_matrix(all_vectors)

    TARGET_NOUNS = ['keys', 'reports', 'files', 'tickets', 'documents', 'alerts']
    MODIFIERS = ['not', 'never', 'without', 'excluding']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(similarity_matrix, xticklabels=all_labels, yticklabels=all_labels, cmap="viridis", annot=False, ax=ax)
    ax.set_title(backbone.model_name, fontsize=32)
    ax.tick_params(axis='x', rotation=90)
    ax.tick_params(axis='y', rotation=0)

    needle_indices = [
        i for i, token in enumerate(all_labels[:num_tokens]) 
        if token.strip().lower() in TARGET_NOUNS + MODIFIERS
    ]

    highlight_color = 'red'
    pooler_color = 'blue'

    for idx in needle_indices:
        ax.get_xticklabels()[idx].set_color(highlight_color)
        ax.get_xticklabels()[idx].set_fontweight('bold')
        ax.get_yticklabels()[idx].set_color(highlight_color)
        ax.get_yticklabels()[idx].set_fontweight('bold')
        ax.add_patch(Rectangle((0, idx), len(all_labels), 1, fill=False, edgecolor=highlight_color, lw=2))
        ax.add_patch(Rectangle((idx, 0), 1, len(all_labels), fill=False, edgecolor=highlight_color, lw=2))

    # --- Highlighting Pooler Vectors (on pooler part of axis) ---
    for i in range(num_tokens, len(all_labels)):
        ax.get_xticklabels()[i].set_color(pooler_color)
        ax.get_xticklabels()[i].set_fontweight('bold')
        ax.get_yticklabels()[i].set_color(pooler_color)
        ax.get_yticklabels()[i].set_fontweight('bold')
    
    fig.suptitle("Augmented Similarity: Tokens vs. Pooling Methods", fontsize=20, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("augmented_similarity.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("augmented_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as augmented_similarity.pdf and .png")


# -------------------------
# CLI Argument Parser
# -------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Run the Relational Needle in a Haystack () experiment.")
    # --- Model & General ---
    p.add_argument("--model_name_or_path", type=str, required=True, help="HF model name or path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=1)
    
    # --- Data Generation ---
    p.add_argument("--num_train_samples", type=int, default=2000, help="Number of synthetic samples for training.")
    p.add_argument("--num_eval_samples", type=int, default=500, help="Number of synthetic samples for evaluation.")
    p.add_argument("--max_length", type=int, default=128, help="Sequence length for synthetic data.")
    p.add_argument("--distractor_ratio", type=float, default=0.8, help="Ratio of noise/haystack tokens (0.0 to 1.0).")
    p.add_argument("--signal_position", type=str, default="random", choices=["start", "middle", "end", "random"])
    p.add_argument("--relational_distance", type=int, default=10, help="Words between modifier ('not') and target.")
    
    # --- Training ---
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)

    # --- Pooling ---
    p.add_argument("--pooling_method", type=str, default="mean", help="[cls, adapool, max, mean, glot]")
    p.add_argument("--decoder_cls_last_token", default=0, type=int, help="For CLS pooling, use last token for decoders.")
    p.add_argument("--scorer_hidden", type=int, default=128, help="Hidden dim for AdaPool scorer.")
    p.add_argument("--gat_hidden_dim", type=int, default=128, help="Hidden dim for GLOT's GAT layers.")
    p.add_argument("--num_layers", type=int, default=2, help="Number of GAT layers in GLOT.")
    p.add_argument("--jk_mode", type=str, default="cat", choices=["cat", "lstm", "max"])
    p.add_argument("--tau", type=float, default=0.3, help="Cosine threshold for GLOT graph.")
    # --- Stage A (HyperGLOT) graph-construction options ---
    p.add_argument("--graph_adj", type=str, default="threshold", choices=["threshold", "knn"])
    p.add_argument("--graph_metric", type=str, default="cosine", choices=["cosine", "poincare"],
                   help="Edge metric: cosine (GLOT) or Poincare hyperbolic distance (Stage A).")
    p.add_argument("--curvature", type=float, default=1.0, help="Poincare ball curvature c.")
    p.add_argument("--rho", type=float, default=1.0, help="Hyperbolic-distance threshold for poincare+threshold.")
    p.add_argument("--knn_k", type=int, default=8, help="Neighbours when graph_adj=knn.")
    p.add_argument("--feature_norm", type=int, default=0, help="L2-normalise tokens before mapping into the ball.")
    # --- Stage B / C (HyperGLOT) options ---
    p.add_argument("--hyperbolic_gnn", type=int, default=0,
                   help="Stage C: replace the Euclidean Token-GNN with an HGCN stack.")
    p.add_argument("--hyperbolic_readout", type=int, default=0,
                   help="Stage B: replace the Euclidean readout with a Poincare gyro-midpoint.")
    p.add_argument("--hyp_gnn_type", type=str, default="gcn", choices=["gcn", "gat"],
                   help="Stage C hyperbolic conv type when hyperbolic_gnn=1: gcn (HGCN) or gat (attention).")
    p.add_argument("--readout_clip", type=float, default=0.0,
                   help="Stage B: clip Euclidean feature norm before the exp map (0=off).")
    p.add_argument("--readout_scale", type=int, default=0,
                   help="Stage B: use a learnable input scale before the exp map.")
    p.add_argument("--learnable_curvature", type=int, default=0,
                   help="Make the Poincare ball curvature c a trainable parameter.")
    p.add_argument("--gnn_input_clip", type=float, default=0.0,
                   help="Stage C: clip token norm before the entry exp map (0=off).")
    p.add_argument("--gnn_input_scale", type=int, default=0,
                   help="Stage C: use a learnable input scale before the entry exp map.")
    p.add_argument("--arm", type=str, default="", help="Optional ablation-arm label recorded in results CSV.")
    p.add_argument("--results_csv", type=str, default="", help="If set, append the stress-test result to this CSV.")
    p.add_argument("--run_tag", type=str, default="", help="Free-form label recorded in the results CSV.")
    p.add_argument("--git_push", type=int, default=0,
                   help="If 1, git add+commit+push the results CSV to GitHub after each run.")
    p.add_argument("--git_remote", type=str, default="origin", help="Git remote to push results to.")
    p.add_argument("--git_branch", type=str, default="", help="Git branch to push results to (default: current).")
    
    return p


def get_augmented_data(model_name, sentence, DEVICE):
    """
    Computes hidden states and also derives vectors for CLS, Mean, and Max pooling.
    Returns a single list of all vectors and a corresponding list of labels.
    """
    print(f"\nProcessing model: {model_name}")
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        token=HF_TOKEN
    ).to(DEVICE)
    model.eval()

    # --- Step 1: Get hidden states for sentence words ONLY ---
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    inputs_no_special = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(DEVICE)
    sentence_tokens = [tokenizer.decode(token_id) for token_id in inputs_no_special["input_ids"][0]]
    
    with torch.no_grad():
        outputs_no_special = model(**inputs_no_special)
        sentence_hidden_states = outputs_no_special.last_hidden_state[0] # (N, D)

    # --- Step 2: Calculate Mean and Max pool vectors ---
    mean_pool_vec = torch.mean(sentence_hidden_states, dim=0)
    max_pool_vec = torch.max(sentence_hidden_states, dim=0).values

    # --- Step 3: Get the CLS (or equivalent) vector ---
    inputs_with_special = tokenizer(sentence, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        outputs_with_special = model(**inputs_with_special)
        all_hidden_states = outputs_with_special.last_hidden_state[0] # (N+special, D)
    
    if is_decoder_like(config):
        # For decoders, use the last token's hidden state as the CLS equivalent
        cls_pool_vec = all_hidden_states[-1]
    else:
        # For encoders like BERT, use the first token's hidden state
        cls_pool_vec = all_hidden_states[0]
        
    # --- Step 4: Combine all vectors and labels ---
    all_vectors = torch.cat([
        sentence_hidden_states,
        cls_pool_vec.unsqueeze(0),
        mean_pool_vec.unsqueeze(0),
        max_pool_vec.unsqueeze(0),
    ], dim=0).cpu()

    all_labels = sentence_tokens + ['[CLS]', '[MEAN]', '[MAX]']

    return all_vectors, all_labels, len(sentence_tokens)

def compute_similarity_matrix(vectors):
    """Computes the cosine similarity matrix from a combined set of vectors."""
    return 1 - cdist(vectors, vectors, metric='cosine')

# --- NEW PLOTTING FUNCTION ---
def plot_augmented_heatmaps(results, needle_words, highlight_color, pooler_color):
    """
    Plots the augmented (N+P)x(N+P) heatmap with special highlighting.
    """
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(10 * num_models, 10))
    
    if num_models == 1: axes = [axes]
        
    print("\nGenerating augmented highlighted plot...")

    for ax, model_name in zip(axes, results.keys()):
        similarity_matrix, labels, num_sentence_tokens = results[model_name]
        
        sns.heatmap(similarity_matrix, xticklabels=labels, yticklabels=labels, cmap="viridis", annot=False, ax=ax)
        
        ax.set_title(model_name, fontsize=14)
        ax.tick_params(axis='x', rotation=90)
        ax.tick_params(axis='y', rotation=0)

        # --- Highlighting Needle Words (on token part of axis) ---
        needle_indices = [
            i for i, token in enumerate(labels[:num_sentence_tokens]) 
            if token.strip().lower() in needle_words
        ]
        for idx in needle_indices:
            ax.get_xticklabels()[idx].set_color(highlight_color)
            ax.get_xticklabels()[idx].set_fontweight('bold')
            ax.get_yticklabels()[idx].set_color(highlight_color)
            ax.get_yticklabels()[idx].set_fontweight('bold')
            ax.add_patch(Rectangle((0, idx), len(labels), 1, fill=False, edgecolor=highlight_color, lw=2))
            ax.add_patch(Rectangle((idx, 0), 1, len(labels), fill=False, edgecolor=highlight_color, lw=2))

        # --- Highlighting Pooler Vectors (on pooler part of axis) ---
        for i in range(num_sentence_tokens, len(labels)):
            ax.get_xticklabels()[i].set_color(pooler_color)
            ax.get_xticklabels()[i].set_fontweight('bold')
            ax.get_yticklabels()[i].set_color(pooler_color)
            ax.get_yticklabels()[i].set_fontweight('bold')

        # --- Draw a dividing line between tokens and poolers ---
        # ax.axhline(y=num_sentence_tokens, color='white', linewidth=3, linestyle='--')
        # ax.axvline(x=num_sentence_tokens, color='white', linewidth=3, linestyle='--')

    fig.suptitle("Augmented Similarity: Tokens vs. Pooling Methods", fontsize=20, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("augmented_similarity.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("augmented_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved as augmented_similarity.pdf and .png")


def main():
    args = build_argparser().parse_args()
    args.decoder_cls_last_token = 0
    if args.model_name_or_path in ["TinyLlama/TinyLlama_v1.1", "HuggingFaceTB/SmolLM2-360M", "meta-llama/Llama-3.2-3B", "mistralai/Mistral-7B-v0.1"]:
        args.decoder_cls_last_token = 1
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    backbone = load_backbone(args.model_name_or_path)
    backbone.model.to(device)

    run_experiment(backbone, args, device)

if __name__ == "__main__":
    main()