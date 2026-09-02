import os
from datetime import datetime
import json
import csv
import time
import random
import argparse
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import geoopt
from hyperbolic_layers import HyperbolicGCNConv, HyperbolicGATConv, hyperbolic_readout
from hyperbolic_graph import (
    HyperbolicGraphConfig, build_pyg_graphs_hyper, _GRAPH_STATS,
)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    # fallback if tqdm unavailable
    def tqdm(x, **kwargs):
        return x

from functools import partial
from transformers import AutoTokenizer, AutoConfig, AutoModel, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from datasets import load_dataset, Dataset
import mteb

from peft import LoraConfig, get_peft_model, TaskType
import wandb
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, MLP, GINConv, GINEConv, GCNConv
from torch_geometric.utils import to_dense_batch, dense_to_sparse, softmax
from torch_scatter import scatter_add

from torch.profiler import profile, ProfilerActivity, record_function



HF_TOKEN = "<>" # Place your huggingface token here




# Helper class to encode large sets of sentences efficiently
class Encoder:
    def __init__(self, model, tokenizer, pooler, pooler_name, is_decoder, device, batch_size=32, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.pooler = pooler
        self.pooler_name = pooler_name
        self.is_decoder = is_decoder
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.model.model = torch.compile(self.model.model)
        self.pooler = torch.compile(self.pooler)

    @torch.no_grad()
    def encode(self, sentences: List[str], **kwargs) -> np.ndarray:
        self.pooler.eval()
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Encoding sentences"):
            batch_sents = sentences[i:i+self.batch_size]
            inputs = self.tokenizer(
                batch_sents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            hidden, mask = forward_hidden(self.model, inputs)
            pooled = pool_hidden(self.pooler, hidden, mask, self.is_decoder, self.pooler_name)
            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_decoder_like(config):
    # Heuristic detection of decoder-only causal models
    if getattr(config, "is_decoder", False):
        return True
    mt = getattr(config, "model_type", "") or ""
    # common causal families
    if mt in {"gpt2", "gptj", "gpt_neo", "llama", "mpt", "gemma", "falcon"}:
        return True
    # architectures hint
    arch = getattr(config, "architectures", None)
    if arch:
        if any(("CausalLM" in a) for a in arch):
            return True
    return False

def masked_mean(x, mask, dim):
    mask = mask.to(x.dtype)
    s = (x * mask.unsqueeze(-1)).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1e-6).unsqueeze(-1)
    return s / denom

def masked_max(x, mask, dim):
    # Set masked positions to very small before max
    very_small = torch.finfo(x.dtype).min
    mask_exp = mask.unsqueeze(-1).to(x.dtype)
    x_masked = x * mask_exp + (1 - mask_exp) * very_small
    return x_masked.max(dim=dim).values

def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def masked_softmax(scores, mask, dim=-1):
    # scores: (..., L)
    # mask: (..., L), 1 for valid
    scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.softmax(scores, dim=dim)



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Pooling modules
# -------------------------

class MeanPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden, attention_mask):
        return masked_mean(hidden, attention_mask, dim=1)

class MaxPooler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, hidden, attention_mask):
        return masked_max(hidden, attention_mask, dim=1)

class CLSPooler(nn.Module):
    def __init__(self, use_last_token_for_decoder=True):
        super().__init__()
        self.use_last_token_for_decoder = use_last_token_for_decoder

    def forward(self, hidden, attention_mask, is_decoder):
        # hidden: (B, L, d)
        if is_decoder and self.use_last_token_for_decoder:
            # pick last non-pad token
            lengths = attention_mask.sum(dim=1)  # (B,)
            idx = (lengths - 1).clamp_min(0).long()
            b_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[b_idx, idx]
        else:
            # first token
            return hidden[:, 0, :]

class MLPPool(nn.Module):
    def __init__(self, inp_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        channel_list = [inp_dim] + [hidden_dim] * num_layers
        self.mlp = MLP(
            channel_list=channel_list
        )
    
    def forward(self, hidden, attention_mask):
        device = hidden.device
        data = build_pyg_graphs(
            hidden, attention_mask, device=device
        )

        data = data.to(device)
        out = self.mlp(data.x, data.batch)
        scores = torch.ones(out.shape[0], device=out.device, dtype=torch.float32).squeeze(-1)
        weights = softmax(scores, data.batch)
        pooled = scatter_add(weights.unsqueeze(-1) * out, data.batch, dim=0)
        return pooled
            
class AdaPool(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, hidden_states, mask):
        scores = self.score_layer(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * hidden_states, dim=1)
        return pooled


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
    adjacency="knn",
    tau=0.3,
    device=None,
):
    """
    Convert a batch of token sequences into a list of torch_geometric.data.Data graphs.

    Args
    -----
    hidden: (B, L, d) last-layer token features
    attention_mask: (B, L) 1=valid, 0=pad
    adjacency: "knn" | "threshold" | "soft" | "softmax"
    k, tau, temperature: graph hyper-params
    include_edge_weight: if True and a weighted adjacency is used, store as 'edge_weight'
    device: move Data.x, edge_index, edge_weight to this device (defaults to hidden.device)

    Returns
    -------
    graphs: List[Data], each with fields:
        x: (n, d) token features (valid tokens only)
        edge_index: (2, E)
        edge_weight: (E,) optional for weighted graphs
        num_nodes: n
        token_idx: (n,) original positions within the L-length sequence
    """
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

        # Telemetry only -- does not touch the graph. The baseline must be as
        # auditable as the hyperbolic arms, otherwise a density mismatch between
        # them is invisible (which is how the empty-graph generation survived).
        _GRAPH_STATS.observe(n, edge_index)

        data.token_idx = token_idx
        graphs.append(data)

    return Batch.from_data_list(graphs)

class GLOT(nn.Module):
    """
    A pooling head that:
      (i) builds PyG graphs from token features,
      (ii) applies ANY chosen PyG conv stack with Jumping Knowledge,
      (iii) pools tokens with an adaptive scorer.
    """
    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        num_layers=2,
        jk_mode="cat",  
        conv="gat",
        adjacency="threshold",
        tau=0.3,
        use_edge_weight=True,
        device=None,
        # --- Stage A (HyperGLOT) options; defaults preserve original GLOT ---
        graph_metric="cosine",   # {"cosine", "poincare"}
        curvature=1.0,            # Poincare ball curvature c (graph_metric=="poincare")
        rho=1.0,                  # ABSOLUTE hyperbolic-distance threshold (see rho_quantile)
        rho_quantile=-1.0,        # if in (0,1), threshold at this quantile of observed
                                  # pairwise distances instead of using `rho`
        tau_quantile=-1.0,        # cosine twin of rho_quantile: keeps the most-similar
                                  # fraction q of pairs. Needed to density-match the
                                  # BASELINE against the hyperbolic arms.
        knn_k=8,                  # neighbours if adjacency=="knn"
        feature_norm=False,       # L2-normalise tokens before mapping into the ball
        feature_mode="none",      # geometry conditioning before expmap0:
                                  # {"none","l2","unit","center","center_unit"}
        graph_scale=1.0,          # extra feature multiplier before expmap0
        graph_curvature=-1.0,     # curvature used ONLY for graph construction;
                                  # <0 falls back to `curvature`. Stage A needs a
                                  # far smaller c than Stage B/C, so sharing one
                                  # value silently couples two unrelated choices.
        edge_weight_mode="binary",# {"binary","soft","hyp"} -- graded edge_attr for the GAT
        edge_temp=0.25,           # soft-weight temperature (fraction of score spread)
        positional_window=0,      # >0: also connect tokens within w sequence positions.
                                  # GLOT's similarity graph is PERMUTATION-INVARIANT, so
                                  # message passing cannot see word order; CoLA is an
                                  # order task. 0 = off (upstream behaviour).
        positional_only=False,    # ablation: use ONLY positional edges, no similarity
        input_scale_norm="none",  # {"none","rms","median"} rescale token features
                                  # entering the pooler. See the note in forward().
        input_scale_target=15.0,  # target mean token norm when input_scale_norm="rms"
        # --- Stage B / C (HyperGLOT) options; defaults preserve original GLOT ---
        hyperbolic_gnn=False,     # Stage C: replace the Euclidean GNN with an HGCN stack
        hyperbolic_readout=False, # Stage B: replace the Euclidean readout with a gyro-midpoint
        hyp_gnn_type="gcn",       # Stage C conv when hyperbolic_gnn: {"gcn", "gat"}
        readout_clip=0.0,         # Stage B: clip Euclidean feature norm before exp map (0=off)
        readout_scale=False,      # Stage B: learnable input scale before the exp map
        learnable_curvature=False,# make the Poincare ball curvature a trainable parameter
        gnn_input_clip=0.0,       # Stage C: clip token norm before lifting onto the ball (0=off)
        gnn_input_scale=False,    # Stage C: learnable input scale before the entry exp map
        gat_edge_attr=True,       # PARITY KNOB. The Euclidean GATConv is built with
                                  # edge_dim=1 and fed edge_attr, while HyperbolicGATConv
                                  # accepts edge_weight and ignores it -- so a Euclidean
                                  # vs hyperbolic contrast varies the edge-attribute path
                                  # as well as the curvature. Even at edge_weight_mode=
                                  # "binary", where every attribute is exactly 1.0, the
                                  # path is not inert: GATConv adds alpha_edge BEFORE a
                                  # leaky_relu, which is not translation-equivariant, so
                                  # the constant does not cancel in the softmax (measured
                                  # relative output change 5.7e-3). Set False to drop the
                                  # path from the Euclidean arm and match the hyperbolic
                                  # one. True reproduces every run in the paper.
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.jk_mode = jk_mode
        self.adjacency = adjacency
        self.tau = tau
        self.use_edge_weight = use_edge_weight
        self.device = device
        self.graph_metric = graph_metric
        self.curvature = curvature
        self.rho = rho
        self.rho_quantile = rho_quantile
        self.tau_quantile = float(tau_quantile)
        self.knn_k = knn_k
        self.feature_norm = feature_norm
        self.feature_mode = str(feature_mode)
        self.graph_scale = float(graph_scale)
        self.graph_curvature = (float(graph_curvature) if float(graph_curvature) > 0
                                else float(curvature))
        self.edge_weight_mode = str(edge_weight_mode)
        self.edge_temp = float(edge_temp)
        self.positional_window = int(positional_window)
        self.positional_only = bool(positional_only)
        self.input_scale_norm = str(input_scale_norm)
        self.input_scale_target = float(input_scale_target)
        self.hyperbolic_gnn = bool(hyperbolic_gnn)
        self.hyperbolic_readout = bool(hyperbolic_readout)
        self.hyp_gnn_type = str(hyp_gnn_type)
        self.readout_clip = float(readout_clip)
        self.learnable_curvature = bool(learnable_curvature)
        self.gnn_input_clip = float(gnn_input_clip)
        self.gat_edge_attr = bool(gat_edge_attr)

        # A single Poincare ball is shared by Stage B (readout) and Stage C (GNN).
        # It is only instantiated when a hyperbolic component is requested, so the
        # original GLOT path never imports/creates any hyperbolic objects.
        self.ball = None
        if self.hyperbolic_gnn or self.hyperbolic_readout:
            self.ball = geoopt.PoincareBall(c=self.curvature, learnable=self.learnable_curvature)

        # Optional learnable input scale for the Stage B readout (fixes the
        # boundary-saturation failure by keeping lifted tokens away from the edge).
        self.readout_scale = None
        if self.hyperbolic_readout and bool(readout_scale):
            self.readout_scale = nn.Parameter(torch.tensor(0.1))

        # Optional learnable input scale for the Stage C GNN entry. Raw BERT
        # tokens have norms ~10-25, so expmap0 pushes them onto the ball boundary
        # (tanh saturates -> vanishing gradients). Scaling/clipping keeps the
        # lifted tokens in the ball interior so message passing sees real
        # curvature structure (same fix as Stage B, applied at the GNN input).
        self.gnn_input_scale = None
        if self.hyperbolic_gnn and bool(gnn_input_scale):
            self.gnn_input_scale = nn.Parameter(torch.tensor(0.1))

        # Build conv stack
        self.convs = nn.ModuleList()
        last_dim = in_dim
        for _ in range(num_layers):
            if self.hyperbolic_gnn:
                # Stage C: hyperbolic layer (see hyperbolic_layers.py). Either the
                # original HGCN (degree-normalised) or the attention-weighted GAT.
                if self.hyp_gnn_type == "gat":
                    layer = HyperbolicGATConv(last_dim, hidden_dim, self.ball)
                else:
                    layer = HyperbolicGCNConv(last_dim, hidden_dim, self.ball)
            elif conv == "gat":
                # edge_dim=1 only when the edge-attribute path is enabled; omitting
                # it also removes GATConv's lin_edge/att_edge parameters, so the
                # Euclidean and hyperbolic attention differ in curvature alone.
                layer = (GATConv(last_dim, hidden_dim, edge_dim=1)
                         if self.gat_edge_attr else GATConv(last_dim, hidden_dim))
            elif conv == "gcn":
                layer = GCNConv(last_dim, hidden_dim)
            elif conv == "gine":
                mlp = MLP([last_dim, hidden_dim, hidden_dim])
                layer = GINEConv(nn=mlp, train_eps=False, edge_dim=1)
            elif conv == "gin":
                mlp = MLP([last_dim, hidden_dim, hidden_dim])
                layer = GINConv(nn=mlp, train_eps=False)
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

        # ---- input scale normalisation (opt-in, default off) ----------------
        # GLOT feeds RAW hidden states into the GAT. That is fine for BERT, whose
        # token norms are ~15, but modern encoders exhibit massive activations /
        # attention sinks: measured mean||x|| is 874 for ModernBERT L12 and 2614
        # for L16, i.e. 60-180x larger. At a fixed lr of 2e-4 that saturates the
        # attention softmax, and ModernBERT duly scores 0.13-0.23 MCC on CoLA at
        # every layer versus BERT's 0.46 -- a broken pipeline, not a worse model.
        #
        # MEASURED CALIBRATION WARNING 1: rescaling to mean||x|| = 1 does NOT work.
        # It replaces a ~60x overshoot with a ~15x undershoot and the GAT cannot
        # learn at lr 2e-4 in 2 epochs. Measured: BERT 0.4554 -> 0.4082, and every
        # ModernBERT layer collapsed to MCC exactly 0.0 (majority class). The
        # target must be BERT's NATURAL scale, which is what the rest of GLOT's
        # hyperparameters were tuned around -- hence input_scale_target=15.0.
        # With target=15 the BERT control is neutral again (0.4554 -> 0.4532).
        #
        # MEASURED CALIBRATION WARNING 2: which STATISTIC to normalise by matters.
        # Measured token-norm mean/median: BERT 0.94-1.02 (clean), ModernBERT L4
        # 1.01 (clean) but L12 7.66 and L22 9.87, with max/median 100 and 105.
        # For those layers the MEAN is dragged up by attention sinks, so mean-based
        # rescaling puts the TYPICAL token at ~2 instead of 15. Use "median" for
        # backbones whose mean/median ratio is far from 1 (see norm_stats.py).
        #
        # NOTE: this does NOT fix ModernBERT on its own. Its token cosines sit at
        # 0.75-0.79 versus BERT's 0.25-0.40, so GLOT's absolute tau=0.6 yields a
        # density of 0.996 -- a near-complete graph carrying no structure. That is
        # a separate bug and is fixed with --tau_quantile, not here.
        #
        # This is a pure SCALE change: relative norms are preserved, so the depth
        # signal Stage A reads out of the radius survives (a per-token LayerNorm
        # would destroy exactly that and make the hyperbolic arms untestable).
        # The cosine graph is scale-invariant, so this only affects the GNN input.
        #
        # Default "none" so every recorded BERT number reproduces bit-for-bit.
        if self.input_scale_norm in ("rms", "median"):
            m = attention_mask.unsqueeze(-1).to(hidden.dtype)          # (B, L, 1)
            valid = attention_mask.bool()
            norms = hidden[valid].norm(dim=-1)                         # (n_valid,)
            if norms.numel() == 0:
                ref = hidden.new_tensor(self.input_scale_target)
            elif self.input_scale_norm == "median":
                ref = norms.median()
            else:
                ref = norms.mean()
            hidden = hidden * (self.input_scale_target / ref.clamp_min(1e-6))
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        #     with record_function("graph_construction"):

        # Choose how token edges are formed:
        #   "cosine"   -> original GLOT (edge if cosine similarity > tau)
        #   "poincare" -> Stage A / HyperGLOT (edge if hyperbolic distance < rho)
        # Choose how token edges are formed. Four configurations are supported:
        #   cosine   + threshold -> original GLOT (exact)            [else branch]
        #   cosine   + knn       -> Stage A builder in cosine mode   [hyperbolic builder]
        #   poincare + threshold -> Stage A hyperbolic (dist < rho)  [hyperbolic builder]
        #   poincare + knn       -> Stage A hyperbolic kNN           [hyperbolic builder]
        # The original build_pyg_graphs only implements cosine+threshold, so any
        # poincare metric OR any knn adjacency routes through the Stage A builder.
        # Soft edge weights also need the Stage A builder, because upstream's
        # `_threshold_edges` can only emit binary weights.
        use_hyperbolic_builder = (
            (self.graph_metric == "poincare")
            or (self.adjacency == "knn")
            or (self.edge_weight_mode != "binary")
            or (0.0 < getattr(self, "tau_quantile", -1.0) < 1.0)
            # Upstream's build_pyg_graphs has no notion of token position, so any
            # positional-edge request must route through the Stage A builder.
            or (getattr(self, "positional_window", 0) > 0)
        )

        if use_hyperbolic_builder:
            # Stage A token-graph construction (see hyperbolic_graph.py).
            hyperbolic_config = HyperbolicGraphConfig(
                graph_metric=self.graph_metric,   # "cosine" or "poincare"
                adjacency=self.adjacency,          # "threshold" or "knn"
                tau=self.tau,
                tau_quantile=getattr(self, "tau_quantile", -1.0),
                rho=self.rho,
                rho_quantile=getattr(self, "rho_quantile", -1.0),
                k=self.knn_k,
                curvature=getattr(self, "graph_curvature", self.curvature),
                feature_norm=self.feature_norm,
                feature_mode=getattr(self, "feature_mode", "none"),
                graph_scale=getattr(self, "graph_scale", 1.0),
                edge_weight_mode=getattr(self, "edge_weight_mode", "binary"),
                edge_temp=getattr(self, "edge_temp", 0.25),
                positional_window=getattr(self, "positional_window", 0),
                positional_only=getattr(self, "positional_only", False),
            )
            batch = build_pyg_graphs_hyper(
                hidden, attention_mask, hyperbolic_config, device=device
            )
        else:
            # Original GLOT path: cosine-similarity threshold graph (exact reproduction).
            batch = build_pyg_graphs(
                hidden, attention_mask, adjacency=self.adjacency,
                tau=self.tau, device=device,
            )

        batch = batch.to(device)
        x, edge_index = batch.x, batch.edge_index
        edge_weight = getattr(batch, "edge_attr", None)

        h_list = [x]
        if self.hyperbolic_gnn:
            # Stage C: hyperbolic Token-GNN. Node features live on the Poincare
            # ball; we keep a tangent-space copy of each layer's output so the
            # Jumping-Knowledge concatenation and the scorer are unchanged in dim.
            # Stabilise the entry lift so raw BERT norms (~10-25) don't saturate
            # expmap0 at the ball boundary (mirrors the Stage B readout fix).
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
            # Original GLOT path: Euclidean GNN stack.
            h = x
            for conv in self.convs:
                if isinstance(conv, GATConv):
                    h = conv(h, edge_index,
                             edge_attr=edge_weight if self.gat_edge_attr else None)
                elif isinstance(conv, GCNConv):
                    h = conv(h, edge_index, edge_weight=edge_weight.squeeze())
                elif isinstance(conv, GINConv):
                    h = conv(h, edge_index)
                elif isinstance(conv, GINEConv):
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

        # with record_function("readout"):
        scores = self.score_layer(h_all).squeeze(-1)
        weights = softmax(scores, batch.batch)
        if self.hyperbolic_readout:
            # Stage B: aggregate the refined tokens with a curvature-aware
            # Einstein / gyro midpoint in the Poincare ball (returns a tangent
            # vector so the Euclidean classifier/projection head is unchanged).
            readout_c = self.ball.c if self.learnable_curvature else self.curvature
            pooled = hyperbolic_readout(
                h_all, weights, batch.batch, self.ball, readout_c,
                num_graphs=int(batch.batch.max().item()) + 1,
                scale=self.readout_scale, clip=self.readout_clip,
            )
        else:
            # Original GLOT readout: Euclidean attention-weighted sum.
            pooled = scatter_add(weights.unsqueeze(-1) * h_all, batch.batch, dim=0)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        return pooled


# -------------------------
# Heads for training objectives
# -------------------------

class ProjectionHead(nn.Module):
    """Optional projection before cosine, e.g., identity by default."""
    def __init__(self, in_dim, out_dim, normalize=True):
        super().__init__()
        self.proj = None
        if out_dim is not None:
            self.proj = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        else:
            self.out_dim = in_dim
        self.normalize = normalize

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            z = self.proj(z)
        if self.normalize:
            z = l2_normalize(z)
        return z

class PairClassifier(nn.Module):
    """Linear classifier over r = [z1, z2]."""
    def __init__(self, dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(2 * dim, num_labels)

    def forward(self, z1, z2):
        r = torch.cat([z1, z2], dim=-1)
        return self.classifier(r)

class SingleClassifier(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.classifier = nn.Linear(dim, num_labels)

    def forward(self, z):
        return self.classifier(z)

def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples['input_texts'],
                           max_length=512,
                           padding=True,
                           truncation=True)

    return batch_dict

class CustomMTEBModel(mteb.EncoderProtocol):
    def __init__(self, model_name, revision, backbone, pooler, pooler_name, device, args):
        self.backbone = backbone
        model_name = self.backbone.model_name_or_path
        revision = None
        self.model_name = model_name
        self.pooler = pooler
        self.pooler_name = pooler_name
        self.tokenizer = backbone.tokenizer
        self.device = device
        self.args = args

        self.backbone.model.eval()

    @torch.no_grad
    def encode(
        self,
        inputs: torch.utils.data.DataLoader[mteb.types.BatchedInput],
        *,
        task_metadata: mteb.TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: mteb.types.PromptType | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        sentences = [text for batch in inputs for text in batch["text"]]
        total_sentences = len(sentences)
        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)
        
        try:
            first_batch = next(iter(data_loader))
            first_batch = {k: v.to(self.device) for k, v in first_batch.items()}
            hidden, mask = forward_hidden(self.backbone, first_batch)
            first_z = pool_hidden(self.pooler, hidden, mask, self.backbone.is_decoder, self.pooler_name)
            embedding_dim = first_z.shape[-1]
        except StopIteration:
            # Handle case where data_loader is empty
            return np.array([])
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True
        )

        concatenated_embeds_pt = torch.empty(
            (total_sentences, embedding_dim), 
            dtype=torch.float32, 
            device=self.device
        ) 
        current_index = 0

        encoded_embeds = []
        for batch_dict in tqdm(data_loader, desc="Encoding"):
            batch_dict = {k: v.to(self.device) for k,v in batch_dict.items()}
            hidden, mask = forward_hidden(self.backbone, batch_dict)
            z = pool_hidden(self.pooler, hidden, mask, self.backbone.is_decoder, self.pooler_name)
            z = F.normalize(z, p=2, dim=-1)

            batch_size = z.shape[0]
            concatenated_embeds_pt[current_index : current_index + batch_size] = z
            current_index += batch_size
           
        concatenated_embeds = concatenated_embeds_pt.cpu().numpy()
        
        return concatenated_embeds
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray):
        return embeddings1 @ embeddings2.T
    
    def similarity_pairwise(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        # Handle 1D case (single embedding)
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1[None, :]
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2[None, :]

        # Pairwise dot product = cosine (since normalized)
        return np.sum(embeddings1 * embeddings2, axis=1)

@dataclass
class Backbone:
    tokenizer: AutoTokenizer
    model: AutoModel
    config: AutoConfig
    is_decoder: bool
    pad_token_id: int
    model_name_or_path: str
    # Which transformer layer to read token features from. -1 = the last layer,
    # which is what GLOT uses. Measured Gromov delta-hyperbolicity of the token
    # metric space on bert-base (2*delta/diam, 0 = perfect tree):
    #     layer  1     2     3     8    12
    #     delta  .079  .087  .090  .151  .196
    # i.e. the LAST layer -- the only one GLOT ever uses -- is the LEAST
    # tree-like in the network, by a factor of 2.5. Hyperbolic geometry only
    # has a mechanism to help when the data is approximately tree-like, so the
    # early layers are where Stage A should have been applied all along.
    hidden_layer: int = -1

def load_backbone(model_name_or_path, max_length, decoder_cls_last_token=None, task="mteb",
                  hidden_layer=-1):
    config = AutoConfig.from_pretrained(model_name_or_path, token=HF_TOKEN)
    # Upstream GLOT hardcodes use_fast=False. That is fine for BERT (which ships a
    # slow tokenizer) but hard-fails for fast-only tokenizers such as
    # ModernBERT (BPE, no slow class) and DeBERTa-v3 (no sentencepiece->slow
    # converter). Try slow first so BERT behaviour is bit-identical to upstream,
    # then fall back to fast so newer backbones can be evaluated at all.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=HF_TOKEN, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=HF_TOKEN, use_fast=True)
    is_dec = is_decoder_like(config)

    # Ensure padding token & side
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if is_dec:
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(model_name_or_path, token=HF_TOKEN, torch_dtype=torch.float16 if task == "mteb" else torch.float32)

    # lora_config = LoraConfig(
    #     task_type=TaskType.SEQ_CLS,
    #     r=64,                       # rank = 64
    #     lora_alpha=64,              # scaling: alpha/r = 1 → no extra scaling
    #     lora_dropout=0.1,
    #     target_modules=["query", "key", "value", "dense"],  # BERT linear layers
    #     bias="none",
    # )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    # model = model.model
    # Ensure model resized if new pad token added
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    # For speed, we do not output hidden_states other than last
    model.eval()

    # Determine CLS strategy
    if decoder_cls_last_token is None:
        decoder_cls_last_token = is_dec

    return Backbone(
        tokenizer=tokenizer,
        model=model, # for lora
        config=config,
        is_decoder=is_dec,
        pad_token_id=tokenizer.pad_token_id,
        model_name_or_path=model_name_or_path,
        hidden_layer=int(hidden_layer),
    ), decoder_cls_last_token

# -------------------------
# Encoders & Pooling factory
# -------------------------

def build_pooler(name: str, hidden_size: int, args) -> nn.Module:
    name = name.lower()
    if name == "mean":
        return MeanPooler()
    elif name == "max":
        return MaxPooler()
    elif name == "cls":
        return CLSPooler(use_last_token_for_decoder=args.decoder_cls_last_token)
    elif name == "adapool":
        return AdaPool(in_dim=hidden_size, hidden_dim=args.scorer_hidden)
    elif name == "mlp":
        return MLPPool(inp_dim=hidden_size, hidden_dim=args.gat_hidden_dim, num_layers=args.num_layers)
    elif name == "glot":
        return GLOT(
            in_dim=hidden_size,
            hidden_dim=args.gat_hidden_dim,
            num_layers=args.num_layers,
            jk_mode=args.jk_mode,
            conv=args.gnn_type,
            adjacency=args.graph_adj,
            tau=args.tau,
            graph_metric=getattr(args, "graph_metric", "cosine"),
            curvature=getattr(args, "curvature", 1.0),
            rho=getattr(args, "rho", 1.0),
            rho_quantile=float(getattr(args, "rho_quantile", -1.0)),
            tau_quantile=float(getattr(args, "tau_quantile", -1.0)),
            knn_k=getattr(args, "knn_k", 8),
            feature_norm=bool(getattr(args, "feature_norm", 0)),
            feature_mode=getattr(args, "feature_mode", "none"),
            graph_scale=float(getattr(args, "graph_scale", 1.0)),
            graph_curvature=float(getattr(args, "graph_curvature", -1.0)),
            edge_weight_mode=getattr(args, "edge_weight_mode", "binary"),
            edge_temp=float(getattr(args, "edge_temp", 0.25)),
            positional_window=int(getattr(args, "positional_window", 0)),
            positional_only=bool(getattr(args, "positional_only", 0)),
            input_scale_norm=getattr(args, "input_scale_norm", "none"),
            input_scale_target=float(getattr(args, "input_scale_target", 15.0)),
            hyperbolic_gnn=bool(getattr(args, "hyperbolic_gnn", 0)),
            hyperbolic_readout=bool(getattr(args, "hyperbolic_readout", 0)),
            hyp_gnn_type=getattr(args, "hyp_gnn_type", "gcn"),
            readout_clip=float(getattr(args, "readout_clip", 0.0)),
            readout_scale=bool(getattr(args, "readout_scale", 0)),
            learnable_curvature=bool(getattr(args, "learnable_curvature", 0)),
            gnn_input_clip=float(getattr(args, "gnn_input_clip", 0.0)),
            gnn_input_scale=bool(getattr(args, "gnn_input_scale", 0)),
            gat_edge_attr=bool(int(getattr(args, "gat_edge_attr", 1))),
        )
    else:
        raise ValueError(f"Unknown pooling method: {name}")

def pool_hidden(pooler: nn.Module, hidden: torch.Tensor, attention_mask: torch.Tensor, is_decoder: bool, pooler_name: str):
    if isinstance(pooler, CLSPooler):
        return pooler(hidden, attention_mask, is_decoder)
    else:
        return pooler(hidden, attention_mask)

def encode_texts(tokenizer: AutoTokenizer, texts, max_length, device):
    batch = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in batch.items()}

def forward_hidden(backbone: Backbone, batch_inputs):
    with torch.no_grad():
        outputs = backbone.model(**batch_inputs, return_dict=True, output_hidden_states=True)
        layer = getattr(backbone, "hidden_layer", -1)
        if layer is not None and layer >= 0:
            # hidden_states[0] is the embedding output, so hidden_states[k] is
            # the output of transformer block k (1-indexed), matching the layer
            # numbering used by screen_backbones.py.
            hidden = outputs.hidden_states[layer]
        elif backbone.is_decoder:
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs.last_hidden_state

    attention_mask = batch_inputs["attention_mask"]
    return hidden.float(), attention_mask


# -------------------------
# Metrics (no SciPy dependency)
# -------------------------

def _rankdata(a):
    # average ranks for ties
    temp = a.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a), dtype=float)
    # handle ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    ranks = ranks + 1.0  # 1-based
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            avg = ranks[idx].mean()
            ranks[idx] = avg
    return ranks

def pearsonr(x, y):
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return float((x * y).mean())

def spearmanr(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearsonr(rx, ry)

def accuracy(preds, labels):
    return float((preds == labels).mean())

def mcc_binary(preds, labels, eps=1e-12):
    """Matthews correlation coefficient for 0/1 labels."""
    preds = preds.astype(int)
    labels = labels.astype(int)
    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / (denom + eps)

def f1_binary(preds, labels):
    # F1 for positive class (label==1)
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    if tp + fp + fn == 0:
        return 0.0
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, query_embeddings, passage_embeddings):
        # Compute similarity matrix
        similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Symmetric loss
        loss_query = F.cross_entropy(similarity_matrix, labels)
        loss_passage = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_query + loss_passage) / 2


# -------------------------
# Dataset loaders
# -------------------------

def load_stsb(task):
    # GLUE STS-B 
    # and also task = [mrpc, rte, wnli]
    ds = load_dataset("nyu-mll/glue", task)
    train = ds["train"]
    val = ds["validation"]
    # Map to common fields
    train = train.rename_columns({"sentence1": "text_a", "sentence2": "text_b"})
    val = val.rename_columns({"sentence1": "text_a", "sentence2": "text_b"})
    return train, val

def load_qqp():
    # ds = load_dataset("glue", "qqp")
    train = load_dataset("nyu-mll/glue", "qqp", split="train[:20000]")
    val = load_dataset("nyu-mll/glue", "qqp", split="validation")
    train = train.rename_columns({"question1": "text_a", "question2": "text_b"})
    val = val.rename_columns({"question1": "text_a", "question2": "text_b"})
    return train, val

def load_qnli():
    # ds = load_dataset("glue", "qnli")
    train = load_dataset("nyu-mll/glue", "qnli", split="train[:20000]")
    val = load_dataset("nyu-mll/glue", "qnli", split="validation")
    train = train.rename_columns({"question": "text_a", "sentence": "text_b"})
    val = val.rename_columns({"question": "text_a", "sentence": "text_b"})
    return train, val

def load_mnli():
    # ds = load_dataset("glue", "mnli")
    train = load_dataset("nyu-mll/glue", "mnli", split="train[:20000]")
    val_m = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    val_mm = load_dataset("nyu-mll/glue", "mnli", split="validation_mismatched")
    train = train.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    val_m = val_m.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    val_mm = val_mm.rename_columns({"premise": "text_a", "hypothesis": "text_b"})
    return train, val_m, val_mm

def load_sst2():
    ds = load_dataset("nyu-mll/glue", "sst2")
    return ds["train"], ds["validation"]

def load_cola():
    ds = load_dataset("nyu-mll/glue", "cola")
    return ds["train"], ds["validation"]

def load_imdb(dev_fraction: float = 0.1, split_seed: int = 12345):
    """IMDB ships only ``train`` and ``test`` -- there is no development split.

    Selecting hyper-parameters against ``test`` would mean tuning on the split
    we also report, which is precisely the practice this project criticises, so
    we carve a fixed dev slice out of ``train`` instead. ``split_seed`` is
    deliberately independent of ``--seed``: every arm and every confirmation
    seed must see the identical train/dev partition, otherwise the paired
    comparison across arms is no longer paired.

    Returns ``(train, test, dev)``; ``dev`` is ``None`` when ``dev_fraction``
    is non-positive, which reproduces the original two-way behaviour.
    """
    ds = load_dataset("imdb")
    if dev_fraction <= 0.0:
        return ds["train"], ds["test"], None
    parts = ds["train"].train_test_split(
        test_size=dev_fraction, seed=split_seed, stratify_by_column="label"
    )
    return parts["train"], ds["test"], parts["test"]

def load_embedding_dataset(train_file, num_samples):
    if num_samples == "subset":
        split = "train[:20000]"
    else:
        split = "train"
    ds = load_dataset("json", data_files=train_file, split=split)
    return ds


def collate_embedding(examples, tokenizer: AutoTokenizer, device, args):
    texts_a = [ex["query"] for ex in examples]
    texts_b = [ex["pos"][0] for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch_a = tokenizer(texts_a, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {
        "a_input_ids": batch_a["input_ids"].to(device),
        "a_attention_mask": batch_a["attention_mask"].to(device),
        "b_input_ids": batch_b["input_ids"].to(device),
        "b_attention_mask": batch_b["attention_mask"].to(device),
    }
    return batch


def collate_pairs(examples, tokenizer: AutoTokenizer, device, args):
    texts_a = [ex["text_a"] for ex in examples]
    texts_b = [ex["text_b"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch_a = tokenizer(texts_a, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {
        "a_input_ids": batch_a["input_ids"].to(device),
        "a_attention_mask": batch_a["attention_mask"].to(device),
        "b_input_ids": batch_b["input_ids"].to(device),
        "b_attention_mask": batch_b["attention_mask"].to(device),
        "labels": torch.tensor(labels, dtype=torch.float32, device=device)
    }
    return batch

def collate_pairs_cls(examples, tokenizer: AutoTokenizer, device, args):
    # labels as int for classification
    texts_a = [ex["text_a"] for ex in examples]
    texts_b = [ex["text_b"] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch_a = tokenizer(texts_a, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch_b = tokenizer(texts_b, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {
        "a_input_ids": batch_a["input_ids"].to(device),
        "a_attention_mask": batch_a["attention_mask"].to(device),
        "b_input_ids": batch_b["input_ids"].to(device),
        "b_attention_mask": batch_b["attention_mask"].to(device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device)
    }
    return batch

def collate_single(examples, tokenizer: AutoTokenizer, text_key, device, args):
    texts = [ex[text_key] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    padding = "max_length" if not args.adaptive_length else False
    max_length = args.max_length if not args.adaptive_length else None
    truncation = not args.adaptive_length
    batch = tokenizer(texts, padding=padding, truncation=truncation, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = torch.tensor(labels, dtype=torch.long, device=device)
    return batch

# -------------------------
# Training / Evaluation loops
# -------------------------
class BatchedHiddenStateDataset(torch.utils.data.Dataset):
    def __init__(self, batch_dir):
        self.batch_dir = batch_dir
        meta_file = os.path.join(batch_dir, "metadata.json")

        with open(meta_file, 'r') as f:
            self.metadata = json.load(f)

        self.batch_files = self.metadata["batch_files"]
        self.total_batches = self.metadata["total_batches"]
        self.total_samples = self.metadata["total_samples"]
        self.has_b = self.metadata.get("has_b", False)
        self.has_labels = self.metadata.get("has_labels", False)

        # Precompute cumulative sample counts per batch for indexing
        self.batch_sample_counts = []
        self.cumulative_samples = [0]

        for i, batch_file in enumerate(self.batch_files):
            # You could load shape here, but we assume consistent batch sizes except last
            # Alternatively, store sample_count per batch in metadata
            data = torch.load(batch_file, map_location='cpu')
            sample_count = len(data["a_hs"])
            self.batch_sample_counts.append(sample_count)
            self.cumulative_samples.append(self.cumulative_samples[-1] + sample_count)

        self.cumulative_samples = self.cumulative_samples[:-1]  # remove last dummy

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which batch this sample belongs to
        batch_idx = 0
        while batch_idx < len(self.batch_sample_counts) - 1 and idx >= self.cumulative_samples[batch_idx + 1]:
            batch_idx += 1

        # Load batch (you can add caching here if needed)
        batch_data = torch.load(self.batch_files[batch_idx], map_location='cpu')

        # Find position within batch
        local_idx = idx - self.cumulative_samples[batch_idx]

        # Extract tensors for this sample
        item = [
            batch_data["a_hs"][local_idx],
            batch_data["a_ms"][local_idx],
        ]

        if self.has_b:
            item.extend([
                batch_data["b_hs"][local_idx],
                batch_data["b_ms"][local_idx],
            ])

        if self.has_labels:
            item.append(batch_data["labels"][local_idx])

        return tuple(item)


@torch.no_grad()
def precompute_hidden_states(backbone: Backbone, loader, dataset_name, split, save_path, override=False):
    # The layer suffix MUST be part of the cache key: features from different
    # layers are different data, and silently reusing a last-layer cache for a
    # layer-2 run would be undetectable in the results.
    layer = getattr(backbone, "hidden_layer", -1)
    layer_tag = "" if layer is None or layer < 0 else f"_L{layer}"
    batch_dir = os.path.join(
        save_path,
        f"{backbone.model_name_or_path.replace('/', '_')}_{dataset_name.replace('/', '_')}_{split}{layer_tag}_batches"
    )
    meta_file = os.path.join(batch_dir, "metadata.json")

    # Check if already precomputed
    if not override and os.path.exists(meta_file):
        print(f"Loading from precomputed batches in {batch_dir}")
        return BatchedHiddenStateDataset(batch_dir)

    os.makedirs(batch_dir, exist_ok=True)

    device = next(backbone.model.parameters()).device
    batch_files = []
    total_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Precomputing {split}")):
        batch_data = {}

        # Move inputs to device
        if "b_input_ids" in batch:
            a_hidden, a_mask = forward_hidden(backbone, {
                "input_ids": batch["a_input_ids"].to(device),
                "attention_mask": batch["a_attention_mask"].to(device)
            })
            b_hidden, b_mask = forward_hidden(backbone, {
                "input_ids": batch["b_input_ids"].to(device),
                "attention_mask": batch["b_attention_mask"].to(device)
            })
            batch_data.update({
                "a_hs": a_hidden.cpu(),
                "a_ms": a_mask.cpu(),
                "b_hs": b_hidden.cpu(),
                "b_ms": b_mask.cpu(),
            })
        else:
            a_hidden, a_mask = forward_hidden(backbone, {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device)
            })
            batch_data.update({
                "a_hs": a_hidden.cpu(),
                "a_ms": a_mask.cpu(),
            })

        if "labels" in batch:
            batch_data["labels"] = batch["labels"].cpu()

        # Save batch
        batch_file = os.path.join(batch_dir, f"batch_{batch_idx:05d}.pt")
        torch.save(batch_data, batch_file)
        batch_files.append(batch_file)
        total_samples += len(next(iter(batch_data.values())))  # get batch size from any tensor

    # Save metadata
    metadata = {
        "total_batches": len(batch_files),
        "total_samples": total_samples,
        "batch_files": batch_files,
        "has_b": "b_hs" in batch_data,
        "has_labels": "labels" in batch_data,
        "a_hs_shape": batch_data["a_hs"].shape[1:],  # without batch dim
        "a_ms_shape": batch_data["a_ms"].shape[1:],
    }
    if "b_hs" in batch_data:
        metadata.update({
            "b_hs_shape": batch_data["b_hs"].shape[1:],
            "b_ms_shape": batch_data["b_ms"].shape[1:],
        })

    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(batch_files)} batches to {batch_dir}")
    return BatchedHiddenStateDataset(batch_dir)


def train_sts_regression(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    train_ds,
    val_ds,
    args,
    device
):
    # labels scale: default [0,5] -> scale to [0,1] per paper
    def scale_label(y):
        if args.label_scale == "0_1":
            return y / 5.0
        elif args.label_scale == "-1_1":
            return (y / 2.5) - 1.0
        else:
            return y

    run = wandb.init(project="GLOT")
    wandb.config.update(args)

    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, "sts", "train", "./data/", override=args.override_precompute)
        val_ds = precompute_hidden_states(backbone, val_loader, "sts", "val", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs(ex, backbone.tokenizer, device, args)
        )

    sample = next(iter(val_loader))
    if args.precompute_hidden_states:
        a_hidden, a_mask = sample[0].to(device), sample[1].to(device)
    else:
        a_hidden, a_mask = forward_hidden(backbone, {"input_ids": sample["a_input_ids"], "attention_mask": sample["a_attention_mask"]})
    
    z = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    proj = ProjectionHead(in_dim=dim, out_dim=dim, normalize=False).to(device)
    
    # Trainable parts: pooler if it has params (adapool or graphpooljk) + projection head
    trainable = []
    for m in [pooler, proj]:
        if any(p.requires_grad for p in m.parameters()):
            trainable += list(m.parameters())
    if args.finetune_backbone and not args.precompute_hidden_states:
        for p in backbone.model.parameters():
            if p.requires_grad:
                trainable.append(p)

    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # MSE on cosine similarity
    best_val = -1.0
    best_sp = -1.0
    best_pe = -1.0
    last_sp = 0.0
    last_pe = 0.0
    for epoch in range(args.epochs):
        pooler.train()
        proj.train()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{pooler_name}] STS-B Train ep{epoch+1}"):
            optimizer.zero_grad()
            # Encode A
            if args.precompute_hidden_states:
                a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                labels = scale_label(batch[-1].squeeze()).to(device)
            else:
                a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                labels = scale_label(batch["labels"])
            za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
            za = proj(za)
            # Encode B
            if args.precompute_hidden_states:
                b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
            else:
                b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
            zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
            zb = proj(zb)
            # cosine prediction
            yhat = F.cosine_similarity(za, zb)
            loss = F.mse_loss(yhat, labels)
            loss.backward()
            optimizer.step()
            print(f"\nPeak memory allocated on GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

            wandb.log({"loss/step": loss})
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Eval
        pooler.eval()
        proj.eval()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.eval()
        gts = []
        preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{pooler_name}] STS-B Eval ep{epoch+1}"):
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                    labels = scale_label(batch[-1].squeeze()).cpu().numpy()
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                    labels = scale_label(batch["labels"].squeeze()).cpu().numpy()
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                za = proj(za)
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                zb = proj(zb)
                yhat = F.cosine_similarity(za, zb).cpu().numpy()
                preds.append(yhat)
                gts.append(labels)
                # break # TODO:
        preds = np.concatenate(preds)
        gts = np.concatenate(gts)
        # Rescale back to original [0,5] for metrics consistency (optional)
        if args.label_scale == "0_1":
            preds_raw = preds * 5.0
            gts_raw = gts * 5.0
        elif args.label_scale == "-1_1":
            preds_raw = (preds + 1.0) * 2.5
            gts_raw = (gts + 1.0) * 2.5
        else:
            preds_raw = preds
            gts_raw = gts

        sp = spearmanr(gts_raw, preds_raw)
        pe = pearsonr(gts_raw, preds_raw)
        
        wandb.log({
            "metrics/pearson": pe,
            "metrics/spearman": sp,
        })

        if args.verbose:
            print(f"[{pooler_name}] epoch {epoch+1} MSE {avg_loss:.4f} Spearman {sp:.4f} Pearson {pe:.4f}", flush=True)
        best_val = max(best_val, (sp + pe) / 2.0)
        best_sp = max(best_sp, sp)
        best_pe = max(best_pe, pe)
        last_sp, last_pe = sp, pe
    
    run.finish()
    return {"spearman": best_sp, "pearson": best_pe, "best_val_avg": best_val,
            "spearman_final": last_sp, "pearson_final": last_pe}

def train_pair_classification(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    num_labels: int,
    train_ds,
    val_ds,
    args,
    device,
    val_ds_mm=None
):
    
    run = wandb.init(project="GLOT")
    wandb.config.update(args)
    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, args.task, "train", "./data/", override=args.override_precompute)
        val_ds = precompute_hidden_states(backbone, val_loader, args.task, "val", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
    if val_ds_mm is not None:
        val_loader_mm = torch.utils.data.DataLoader(
            val_ds_mm,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda ex: collate_pairs_cls(ex, backbone.tokenizer, device, args)
        )
        if args.precompute_hidden_states:
            val_ds_mm = precompute_hidden_states(backbone, val_loader_mm, args.task, "val_mm", "./data/", override=args.override_precompute)
            val_loader_mm = torch.utils.data.DataLoader(
            val_ds_mm,
            batch_size=args.eval_batch_size,
            shuffle=False
        )
    # Determine pooled dim
    sample = next(iter(val_loader))
    if args.precompute_hidden_states:
        a_hidden, a_mask = sample[0].to(device), sample[1].to(device)
    else:
        a_hidden, a_mask = forward_hidden(backbone, {"input_ids": sample["a_input_ids"], "attention_mask": sample["a_attention_mask"]})
    z = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    classifier = PairClassifier(dim=dim, num_labels=num_labels).to(device)
    params = list(classifier.parameters())
    # Include pooler params if any
    for p in pooler.parameters():
        if p.requires_grad:
            params.append(p)
    if args.finetune_backbone and not args.precompute_hidden_states:
        for p in backbone.model.parameters():
            if p.requires_grad:
                params.append(p)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    best_f1 = 0.0
    last_acc = 0.0
    last_f1 = 0.0

    for epoch in range(args.epochs):
        pooler.train()
        classifier.train()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{pooler_name}] PairCls Train ep{epoch+1}"):
            # Encode A
            if args.precompute_hidden_states:
                a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                labels = batch[-1].squeeze().to(device)
            else:
                a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                labels = batch["labels"]
            za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
            # Encode B
            if args.precompute_hidden_states:
                b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
            else:
                b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
            zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
            logits = classifier(za, zb)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(f"\nPeak memory allocated on GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

            wandb.log({"loss/step": loss})
            
        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Eval
        pooler.eval()
        classifier.eval()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{pooler_name}] PairCls Eval ep{epoch+1}"):
                # Encode A
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                    labels = batch[-1].squeeze().cpu().numpy()
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                    labels = batch["labels"].cpu().numpy()
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                # Encode B
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                logits = classifier(za, zb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(labels)
                # break
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy(preds_all, labels_all)
        f1 = f1_binary(preds_all, labels_all) if num_labels == 2 else float('nan')
        wandb.log({
            "metrics/acc": acc,
            "metrics/f1": f1
        })
        if args.verbose:
            print(f"[{pooler_name}] epoch {epoch+1} loss {avg_loss:.4f} acc {acc:.4f} f1 {f1:.4f}")
        best_acc = max(best_acc, acc)
        if not np.isnan(f1):
            best_f1 = max(best_f1, f1)
        last_acc, last_f1 = acc, f1
    
    result = {"acc": best_acc, "f1": best_f1, "acc_final": last_acc, "f1_final": last_f1}

    if val_ds_mm is not None:
        pooler.eval()
        classifier.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader_mm, desc=f"[{pooler_name}] PairCls Eval Mismatched ep{epoch+1}"):
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                    labels = batch[-1].squeeze().cpu().numpy()
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                    labels = batch["labels"].cpu().numpy()
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                logits = classifier(za, zb)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(labels)
                
        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy(preds_all, labels_all)
        f1 = f1_binary(preds_all, labels_all) if num_labels == 2 else float('nan')

        wandb.log({
            "metrics/acc_mm": acc,
            "metrics/f1_mm": f1
        })
        result["acc_mm"] = acc
        result["f1_mm"] = f1

    return result

def train_single_classification(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    num_labels: int,
    train_ds,
    val_ds,
    args,
    device
):

    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2,
        collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            # num_workers=2,
            collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, args.task, "train", "./data/", override=args.override_precompute)
        val_ds = precompute_hidden_states(backbone, val_loader, args.task, "val", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            # num_workers=2
        )
    else:
        train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=2,
        collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            # num_workers=2,
            collate_fn=lambda ex: collate_single(ex, backbone.tokenizer, text_key="text", device=device, args=args)
        )

    run = wandb.init(project="GLOT")
    wandb.config.update(args)
    # Determine pooled dim
    sample = next(iter(val_loader))
    if args.precompute_hidden_states:
        hidden, mask = sample[0].to(device), sample[1].to(device)
    else:
        hidden, mask = forward_hidden(backbone, {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]})
    z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    classifier = SingleClassifier(dim=dim, num_labels=num_labels).to(device)
    params = list(classifier.parameters())
    for p in pooler.parameters():
        if p.requires_grad:
            params.append(p)
    if args.finetune_backbone and not args.precompute_hidden_states:
        for p in backbone.model.parameters():
            p.requires_grad = True
            params.append(p)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # print(f"Total trainable params = {sum([p.numel() for p in params])}")

    best_acc = 0.0
    best_mcc = -1.0
    last_acc = 0.0
    last_mcc = 0.0
    for epoch in range(args.epochs):
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.train()
        pooler.train()
        classifier.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[{pooler_name}] SingleCls Train ep{epoch+1}"):
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            if args.precompute_hidden_states:
                hidden, mask = batch[0].to(device), batch[1].to(device)
                labels = batch[-1].squeeze().to(device)
            else:
                hidden, mask = forward_hidden(backbone, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
                labels = batch["labels"]
            z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, pooler_name)
            logits = classifier(z)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
            # print(f"\nPeak memory allocated on GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            losses.append(loss.item())
            wandb.log({
                "loss/step": loss
            })

        avg_loss = float(np.mean(losses)) if losses else 0.0

        # Eval
        pooler.eval()
        classifier.eval()
        if args.finetune_backbone and not args.precompute_hidden_states:
            backbone.model.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{pooler_name}] SingleCls Eval ep{epoch+1}"):
                if args.precompute_hidden_states:
                    hidden, mask = batch[0].to(device), batch[1].to(device)
                    labels = batch[-1].squeeze().cpu().numpy()
                else:
                    hidden, mask = forward_hidden(backbone, {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]})
                    labels = batch["labels"].cpu().numpy()
                z = pool_hidden(pooler, hidden, mask, backbone.is_decoder, pooler_name)
                logits = classifier(z)
                preds = logits.argmax(dim=-1).cpu().numpy()
                preds_all.append(preds)
                labels_all.append(labels)

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy(preds_all, labels_all)
        mcc = mcc_binary(preds_all, labels_all)
        wandb.log({
            "metrics/acc": acc,
            "metrics/mcc": mcc
        })
        if args.verbose:
            print(f"[{pooler_name}] epoch {epoch+1} loss {avg_loss:.4f} acc {acc:.4f} mcc {mcc:.4f}")
        best_acc = max(best_acc, acc)
        best_mcc = max(best_mcc, mcc)
        last_acc, last_mcc = acc, mcc
    return {"acc": best_acc, "mcc": best_mcc, "acc_final": last_acc, "mcc_final": last_mcc}

def train_pair_embedding(
    backbone: Backbone,
    pooler: nn.Module,
    pooler_name: str,
    train_ds,
    args,
    device,
):
    
    run = wandb.init(project="GLOT")
    wandb.config.update(args)
    if args.precompute_hidden_states:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_embedding(ex, backbone.tokenizer, device, args)
        )
        train_ds = precompute_hidden_states(backbone, train_loader, args.task, "train", "./data/", override=args.override_precompute)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda ex: collate_embedding(ex, backbone.tokenizer, device, args)
        )
    # Determine pooled dim
    sample = next(iter(train_loader))
    if args.precompute_hidden_states:
        a_hidden, a_mask = sample[0].to(device), sample[1].to(device)
    else:
        a_hidden, a_mask = forward_hidden(backbone, {"input_ids": sample["a_input_ids"], "attention_mask": sample["a_attention_mask"]})
    z = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
    dim = z.size(-1)

    params = []
    # Include pooler params if any
    for p in pooler.parameters():
        if p.requires_grad:
            params.append(p)
    if len(params) != 0:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=50)
    else:
        optimizer = None

    loss_fn = ContrastiveLoss()
    # The contrastive loss is strictly positive, so seeding the running minimum
    # at 0.0 made ``min(best_acc, avg_loss)`` return 0.0 on every epoch and the
    # reported best_loss was always exactly zero regardless of training.
    best_acc = float("inf")

    if optimizer is not None:
        for epoch in range(args.epochs):
            pooler.train()
            losses = []
            for batch in tqdm(train_loader, desc=f"[{pooler_name}] PairEmb Train ep{epoch+1}"):
                # Encode A
                if args.precompute_hidden_states:
                    a_hidden, a_mask = batch[0].to(device), batch[1].to(device)
                else:
                    a_hidden, a_mask = forward_hidden(backbone, {"input_ids": batch["a_input_ids"], "attention_mask": batch["a_attention_mask"]})
                za = pool_hidden(pooler, a_hidden, a_mask, backbone.is_decoder, pooler_name)
                # Encode B
                if args.precompute_hidden_states:
                    b_hidden, b_mask = batch[2].to(device), batch[3].to(device)
                else:
                    b_hidden, b_mask = forward_hidden(backbone, {"input_ids": batch["b_input_ids"], "attention_mask": batch["b_attention_mask"]})
                zb = pool_hidden(pooler, b_hidden, b_mask, backbone.is_decoder, pooler_name)
                
                loss = loss_fn(za, zb)
                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss)
                losses.append(loss.item())

                wandb.log({"loss/step": loss})
                
            avg_loss = float(np.mean(losses)) if losses else 0.0

            # Eval
            if args.verbose:
                print(f"[{pooler_name}] epoch {epoch+1} loss {avg_loss:.4f}")
            best_acc = min(best_acc, avg_loss)

            # Mid-training MTEB evaluation is off by default. It used the legacy
            # mteb.MTEB(...).run() API that the installed version no longer
            # exposes, and a crash here would kill the run before the checkpoint
            # below is written -- which is the whole point of this stage.
            if getattr(args, "eval_during_embedding", 0) and args.mteb_task:
                model = CustomMTEBModel(
                    model_name=None,
                    revision=None,
                    backbone=backbone,
                    pooler=pooler,
                    pooler_name=pooler_name,
                    device=device,
                    args=args
                )
                tasks = mteb.get_tasks(tasks=[args.mteb_task], languages=["eng"])
                results = mteb.evaluate(model, tasks=tasks,
                                        encode_kwargs={"batch_size": args.batch_size},
                                        overwrite_strategy="always")
                for result in results:
                    print(f"{result.task_name} | {result.get_score()}")
                    wandb.log({f"{result.task_name}": result.get_score()})

    
    # Create informative filename based on args
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Build config string with key parameters
    config_str = f"{args.task}_{args.model_name_or_path.replace('/', '_')}_{args.pooling_method}"

    # Add key hyperparameters
    if args.pooling_method == "glot":
        config_str += f"_layers{args.num_layers}_{args.jk_mode}"

    config_str += f"_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}_len{args.max_length}"

    if args.proj_dim > 0:
        config_str += f"_proj{args.proj_dim}"

    if args.num_train_samples != "full":
        config_str += f"_samples{args.num_train_samples}"

    # Create final path. When --save_ckpt_path is supplied we write exactly
    # there, so a caller can chain this stage into the MTEB stage by name; the
    # timestamped fallback below cannot be predicted from outside the process,
    # which is why the MTEB runs were never given a trained pooler.
    save_path = getattr(args, "save_ckpt_path", "") or os.path.join(
        args.save_dir, 
        f"{config_str}_{timestamp}.pth"
    )
    if optimizer is not None:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(pooler.state_dict(), save_path)
        if args.verbose:
            print(f"[{pooler_name}] saved pooler checkpoint -> {save_path}")

    # A frozen pooler never enters the training loop, so best_acc is still inf;
    # report that as NaN rather than writing inf into the results CSV.
    return best_acc if best_acc != float("inf") else float("nan")

def evaluate_mteb(
    backbone: Backbone,
    pooler,
    pooler_name,
    device,
    args
):
    run = wandb.init(project="GLOT")
    wandb.config.update(args)
    if args.checkpoint_path != "standard" and args.checkpoint_path != "":
        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint_path, weights_only=True)
        else:
            checkpoint = torch.load(args.checkpoint_path, weights_only=True, map_location="cpu")
        pooler.load_state_dict(checkpoint)

    model = CustomMTEBModel(
        model_name=None,
        revision=None,
        backbone=backbone,
        pooler=pooler,
        pooler_name=pooler_name,
        device=device,
        args=args
    )

    tasks = mteb.get_tasks(tasks=[args.mteb_task], languages=["eng"])
    results = mteb.evaluate(model, tasks=tasks, encode_kwargs={'batch_size': args.batch_size}, overwrite_strategy="always")

    scores = {}
    for result in results:
        print(f"{result.task_name} | {result.get_score()}")
        wandb.log({f"{result.task_name}": result.get_score()})
        scores[result.task_name] = float(result.get_score())

    run.finish()
    main_score = float(np.mean(list(scores.values()))) if scores else float("nan")
    return {"mteb_score": main_score, "mteb_task": args.mteb_task, "mteb_per_task": scores}

# -------------------------
# Result bookkeeping (ablation arm naming + per-run CSV)
# -------------------------

def _arm_name(args) -> str:
    """Derive the ablation-arm label from the three HyperGLOT switches.

    A = hyperbolic graph construction (graph_metric == "poincare")
    B = hyperbolic readout           (hyperbolic_readout)
    C = hyperbolic Token-GNN         (hyperbolic_gnn)
    """
    if getattr(args, "arm", ""):
        return args.arm
    a = (getattr(args, "graph_metric", "cosine") == "poincare")
    b = bool(getattr(args, "hyperbolic_readout", 0))
    c = bool(getattr(args, "hyperbolic_gnn", 0))
    if not (a or b or c):
        return "baseline"
    if a and b and c:
        return "ABC"
    if a and not b and not c:
        return "A"
    if b and not a and not c:
        return "B"
    if c and not a and not b:
        return "C"
    # any other mixture
    return "".join([n for n, f in (("A", a), ("B", b), ("C", c)) if f])


# Union of every metric key any task can emit, so the CSV schema is stable.
_RESULT_METRIC_KEYS = [
    "acc", "acc_final", "mcc", "mcc_final", "f1", "f1_final",
    "acc_mm", "f1_mm", "spearman", "spearman_final", "pearson", "pearson_final",
    "best_val_avg", "mteb_score", "best_loss",
]


def append_result_csv(args, metrics: dict, elapsed_sec: float):
    """Append one fully-specified result row to ``args.results_csv``.

    The row records the complete config (model, task, arm, all graph/geometry
    hyper-parameters, training hyper-parameters, seed) plus every metric the
    original GLOT article reports (accuracy, MCC, F1, Spearman/Pearson, MTEB
    score, matched/mismatched). Writing happens immediately after each run so a
    crash mid-sweep never loses finished work.
    """
    metrics = metrics or {}
    path = args.results_csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": getattr(args, "run_tag", ""),
        "model": args.model_name_or_path,
        "task": args.task,
        "mteb_task": args.mteb_task if args.task == "mteb" else "",
        "arm": _arm_name(args),
        "pooling": args.pooling_method,
        "graph_metric": args.graph_metric,
        "graph_adj": args.graph_adj,
        "hyperbolic_gnn": int(getattr(args, "hyperbolic_gnn", 0)),
        "hyperbolic_readout": int(getattr(args, "hyperbolic_readout", 0)),
        "gnn_type": args.gnn_type,
        "num_layers": args.num_layers,
        "jk_mode": args.jk_mode,
        "tau": args.tau,
        "rho": args.rho,
        "curvature": args.curvature,
        "knn_k": args.knn_k,
        "feature_norm": int(getattr(args, "feature_norm", 0)),
        "gat_hidden_dim": args.gat_hidden_dim,
        "scorer_hidden": args.scorer_hidden,
        "proj_dim": args.proj_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "seed": args.seed,
        "elapsed_sec": elapsed_sec,
    }
    for k in _RESULT_METRIC_KEYS:
        v = metrics.get(k, "")
        row[k] = round(v, 6) if isinstance(v, float) else v
    # Preserve any per-task MTEB breakdown as JSON for completeness.
    row["mteb_per_task"] = json.dumps(metrics.get("mteb_per_task", {})) if metrics.get("mteb_per_task") else ""

    fieldnames = list(row.keys())
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[results] appended row -> {path}", flush=True)


def git_push_results(path: str, message: str, branch: str = "", remote: str = "origin"):
    """Stage, commit and push the results CSV so every finished run is saved to
    GitHub immediately. Runs from the CSV's own directory (the repo), never
    raises -- git problems must not abort a training run.
    """
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
            # "nothing to commit" is not an error (row may be unchanged).
            if "nothing to commit" not in (commit.stdout + commit.stderr):
                print(f"[git] commit failed: {(commit.stdout + commit.stderr).strip()}", flush=True)
            return
        push_cmd = ["git", "push", remote] + ([branch] if branch else [])
        push = subprocess.run(push_cmd, cwd=repo_dir, capture_output=True, text=True)
        if push.returncode != 0:
            print(f"[git] push failed: {(push.stdout + push.stderr).strip()}", flush=True)
        else:
            print(f"[git] pushed results -> {remote} {branch}".rstrip(), flush=True)
    except Exception as exc:  # never let bookkeeping abort the run
        print(f"[git] skipped ({exc})", flush=True)


def run_tasks(backbone: Backbone, args, device):
    pooling_name = args.pooling_method

    task = args.task
    pooler = build_pooler(pooling_name, backbone.config.hidden_size, args).to(device)
    start = time.time()
    summary = {
        "model": args.model_name_or_path,
        "pooling": pooling_name,
        "task": task,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }
    metrics = {}

    if task == "stsb":
        train_ds, val_ds = load_stsb(task)
        metrics = train_sts_regression(backbone, pooler, pooling_name, train_ds, val_ds, args, device)

    elif task in ["qqp", "mrpc", "rte", "wnli"]: 
        if task == "qqp":
            train_ds, val_ds = load_qqp()
        else:
            train_ds, val_ds = load_stsb(task)
        metrics = train_pair_classification(backbone, pooler, pooling_name, num_labels=2, train_ds=train_ds, val_ds=val_ds, args=args, device=device)

    elif task == "mnli":
        train_ds, val_m, val_mm = load_mnli()
        # Train using matched, evaluate on matched and mismatched
        metrics = train_pair_classification(backbone, pooler, pooling_name, num_labels=3,
                                                    train_ds=train_ds, val_ds=val_m, args=args, device=device, val_ds_mm=val_mm)

    elif task == "sst2":
        train_ds, val_ds = load_sst2()
        train_ds = train_ds.rename_columns({"sentence": "text"})
        val_ds = val_ds.rename_columns({"sentence": "text"})
        metrics = train_single_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=val_ds, args=args, device=device)

    elif task == "qnli":
        train_ds, val_ds = load_qnli()
        metrics = train_pair_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=val_ds, args=args, device=device)

    elif task == "cola":
        train_ds, val_ds = load_cola()
        train_ds = train_ds.rename_columns({"sentence": "text"})
        val_ds = val_ds.rename_columns({"sentence": "text"})
        metrics = train_single_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=val_ds, args=args, device=device)

    elif task == "imdb":
        train_ds, test_ds, dev_ds = load_imdb(args.imdb_dev_fraction, args.imdb_split_seed)
        eval_ds = dev_ds if (args.eval_split == "dev" and dev_ds is not None) else test_ds
        metrics = train_single_classification(backbone, pooler, pooling_name, num_labels=2,
                                                train_ds=train_ds, val_ds=eval_ds, args=args, device=device)
    
    elif task == "embedding":
        train_ds = load_embedding_dataset(args.train_file, args.num_train_samples)
        best = train_pair_embedding(backbone, pooler, pooling_name, train_ds, args, device)
        metrics = {"best_loss": best}
    
    elif task == "mteb":
        metrics = evaluate_mteb(backbone, pooler, pooling_name, device, args)

    else:
        summary["skipped"] = f"Unknown or unsupported task: {task}"
    
    summary["metrics"] = metrics
    summary["elapsed_sec"] = round(time.time() - start, 2)

    # Persist a one-row result record (all article metrics + full config) so the
    # sweep is resumable and every finished run is saved immediately.
    if getattr(args, "results_csv", ""):
        append_result_csv(args, metrics, summary["elapsed_sec"])
        # Push the CSV to GitHub after each run so results survive a crash and are
        # available remotely as soon as the run finishes.
        if getattr(args, "git_push", 0):
            git_push_results(
                args.results_csv,
                message=f"results: {args.model_name_or_path} {task} {_arm_name(args)} seed{args.seed}",
                branch=getattr(args, "git_branch", ""),
                remote=getattr(args, "git_remote", "origin"),
            )

    if args.verbose:
        print(json.dumps(summary, indent=2, default=str))
    # Machine-readable one-liner for log scraping by the orchestrator.
    print("RESULT_JSON " + json.dumps({
        "model": args.model_name_or_path, "task": task, "arm": _arm_name(args),
        "graph_metric": args.graph_metric, "graph_adj": args.graph_adj,
        "hyperbolic_gnn": int(getattr(args, "hyperbolic_gnn", 0)),
        "hyperbolic_readout": int(getattr(args, "hyperbolic_readout", 0)),
        "seed": args.seed, "metrics": metrics,
    }, default=str), flush=True)

def build_argparser():
    p = argparse.ArgumentParser(description="Train & evaluate LM pooling methods (single-file script).")
    # Model / tokenizer
    p.add_argument("--model_name_or_path", type=str, required=True, help="HF model name or path")
    p.add_argument("--hidden_layer", type=int, default=-1,
                   help="Transformer layer to read token features from (1-indexed; "
                        "-1 = last layer, which is what GLOT uses). Measured Gromov "
                        "delta-hyperbolicity on bert-base (2*delta/diam, 0 = perfect "
                        "tree): L1 .079, L2 .087, L3 .090, L8 .151, L12 .196 -- the "
                        "last layer is the LEAST tree-like in the network, so it is the "
                        "worst place to apply hyperbolic geometry. The layer is part of "
                        "the hidden-state cache key, so caches never collide.")
    p.add_argument("--decoder_cls_last_token", type=int, default=0,
                   help="If True, CLS pooling uses last non-pad token (for decoder-only). Default: auto-detect.")
    # Tasks & data
    p.add_argument("--task", type=str, default="stsb", help="The dataset to run experiments")
    p.add_argument("--train_file", type=str, default="./data/msmarco-triplets.jsonl", help="Download from https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz")
    p.add_argument("--num_train_samples", type=str, default="subset", help="choose from [subset, full]")
    p.add_argument("--checkpoint_path", type=str, default="standard", help="Pooler Checkpoint path to evaluate on MTEB")
    p.add_argument("--save_ckpt_path", type=str, default="",
                   help="Exact path to write the trained pooler to after --task=embedding. "
                        "Leave empty for the legacy timestamped filename. Set this when a "
                        "driver needs to feed the checkpoint into --checkpoint_path.")
    p.add_argument("--eval_during_embedding", type=int, default=0,
                   help="Run an MTEB evaluation after every embedding-training epoch. Off by "
                        "default: it costs a full evaluation per epoch and is not needed to "
                        "produce the checkpoint.")
    p.add_argument("--eval_split", type=str, default="test", choices=["dev", "test"],
                   help="Which split to score on for tasks that expose a held-out dev slice "
                        "(currently IMDB). Use 'dev' for tuning and 'test' for confirmation, "
                        "so no configuration is ever selected on the reported split.")
    p.add_argument("--imdb_dev_fraction", type=float, default=0.1,
                   help="Fraction of IMDB train held out as dev. 0 disables the split.")
    p.add_argument("--imdb_split_seed", type=int, default=12345,
                   help="Seed for the IMDB train/dev partition. Independent of --seed on "
                        "purpose: all arms and seeds must share one partition.")
    p.add_argument("--mteb_task", type=str, default="SciFact", help="Clustering or Retrieval")
    p.add_argument("--save_dir", type=str, default="./saved_models/", help="Directory to save logs/results.")
    p.add_argument("--max_length", type=int, default=128, help="Max length for texts")
    p.add_argument("--adaptive_length", type=int, default=0, help="To use full sentence length")
    # Training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", type=int, default=1)
    # Pooling
    p.add_argument("--pooling_method", type=str, default="max",
                   help="[cls, adapool, max, mean, glot]")
    p.add_argument("--gnn_type", default="gat", type=str)
    p.add_argument("--scorer_hidden", type=int, default=256, help="Hidden dim for adaptive scorer/readout.")
    # GraphPoolJK
    p.add_argument("--gat_hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2, help="Number of GAT layers (K=0 reduces to adaptive scorer).")
    p.add_argument("--jk_mode", type=str, default="cat", choices=["cat", "lstm", "max"])
    p.add_argument("--graph_adj", type=str, default="threshold", choices=["threshold", "knn"])
    p.add_argument("--tau", type=float, default=0.3, help="Threshold for adjacency or mid-point for sigmoid.")
    p.add_argument("--input_scale_norm", type=str, default="none",
                   choices=["none", "rms", "median"],
                   help="Rescale token features entering the pooler to "
                        "--input_scale_target. 'rms' uses the mean token norm, "
                        "'median' uses the median (use this when mean/median is far "
                        "from 1: ModernBERT L12 is 7.66, L22 is 9.87, so its mean is "
                        "outlier-dominated). Default 'none' reproduces upstream GLOT.")
    p.add_argument("--input_scale_target", type=float, default=15.0,
                   help="Target mean token norm for --input_scale_norm=rms. Defaults "
                        "to 15, BERT's natural scale, because GLOT's lr/architecture "
                        "were tuned there. Do NOT set this to 1: measured BERT 0.4554 "
                        "-> 0.4082 and every ModernBERT layer collapsed to MCC 0.0.")
    p.add_argument("--positional_window", type=int, default=0,
                   help="If >0, additionally connect tokens whose sequence positions "
                        "differ by <= w. GLOT's similarity graph is permutation-"
                        "invariant, so the GNN cannot see word order; CoLA measures "
                        "acceptability, which largely IS word order. 0 = upstream.")
    p.add_argument("--positional_only", type=int, default=0,
                   help="Ablation: use ONLY positional edges, dropping similarity "
                        "edges. Required to attribute any positional gain to word "
                        "order rather than to simply having more edges.")
    # --- Stage A (HyperGLOT) graph-construction options ---
    p.add_argument("--graph_metric", type=str, default="cosine", choices=["cosine", "poincare"],
                   help="Edge metric: cosine (GLOT) or Poincare hyperbolic distance (Stage A).")
    p.add_argument("--curvature", type=float, default=1.0, help="Poincare ball curvature c (graph_metric=poincare).")
    p.add_argument("--rho", type=float, default=1.0, help="Hyperbolic-distance threshold for poincare+threshold.")
    p.add_argument("--rho_quantile", type=float, default=-1.0,
                   help="If in (0,1), threshold at this quantile of the observed pairwise "
                        "Poincare distances instead of using --rho. STRONGLY PREFERRED: raw "
                        "BERT token norms (~15) saturate expmap0 onto the ball boundary, so "
                        "real distances sit in ~[8.8, 11.8] and any absolute rho below ~9 "
                        "produces an EMPTY graph (measured: 16/16 sentences with 0 edges for "
                        "rho in {0.5..5.0}). The quantile form is density-matched to cosine's "
                        "tau and transfers across backbones and curvatures.")
    p.add_argument("--knn_k", type=int, default=8, help="Neighbours when graph_adj=knn.")
    p.add_argument("--tau_quantile", type=float, default=-1.0,
                   help="Cosine twin of --rho_quantile: keep the most-similar fraction q "
                        "of token pairs. Required to density-match the BASELINE against "
                        "the hyperbolic arms -- an absolute tau grid cannot reach the low "
                        "densities rho_quantile reaches by construction, so without this "
                        "a hyperbolic arm can 'win' purely on sparsity.")
    p.add_argument("--feature_norm", type=int, default=0, help="L2-normalise tokens before mapping into the ball.")
    p.add_argument("--feature_mode", type=str, default="none",
                   choices=["none", "l2", "unit", "center", "center_unit",
                            "cls_root", "cls_root_unit"],
                   help="Geometry conditioning applied before expmap0. Raw BERT token "
                        "norms have cv=0.057 (almost no depth signal) and mean 14.68, "
                        "which saturates expmap0 so the Poincare graph collapses onto "
                        "the cosine graph. 'center' re-roots the ball at the token mean "
                        "(cv 0.154); 'cls_root' re-roots it at [CLS], so a token's norm "
                        "reads as its specificity relative to the sentence gist; the "
                        "'_unit' variants additionally fix mean||x||=1 so the curvature "
                        "becomes scale-free (effective radius = sqrt(c)).")
    p.add_argument("--graph_scale", type=float, default=1.0,
                   help="Extra feature multiplier before expmap0. Equivalent to "
                        "curvature via s = sqrt(c); provided for readability.")
    p.add_argument("--graph_curvature", type=float, default=-1.0,
                   help="Curvature used ONLY for token-graph construction. <0 falls "
                        "back to --curvature. Stage A saturates at c=1 (its graph "
                        "becomes the cosine graph, Jaccard 0.999) and needs c~1e-2, "
                        "whereas the Stage B/C ball is normally run at c=1; sharing "
                        "one flag silently couples two unrelated choices.")
    p.add_argument("--edge_weight_mode", type=str, default="binary",
                   choices=["binary", "soft", "soft_z", "hyp", "hyp_z", "depth"],
                   help="GLOT fills its edge_dim=1 GAT attribute with 1.0, discarding "
                        "relation strength. 'soft' emits a Fermi-Dirac/sigmoid weight "
                        "(Nickel & Kiela 2017; Chami et al. 2019) on the SAME edge set. "
                        "'hyp' keeps GLOT's cosine edge set but weights it by hyperbolic "
                        "proximity, which is strictly additive w.r.t. the baseline. The "
                        "'_z' variants standardise per graph instead of squashing through "
                        "a sigmoid, which otherwise compresses every weight into ~[0.77,1]. "
                        "'depth' is the signed hyperbolic depth gap along the directed "
                        "edge -- the only ASYMMETRIC option, and the one piece of "
                        "hierarchy a symmetric cosine graph cannot represent.")
    p.add_argument("--edge_temp", type=float, default=0.25,
                   help="Soft-weight temperature as a fraction of the observed spread "
                        "of the score, so it transfers between cosine and Poincare.")
    # --- Stage B / C (HyperGLOT) options ---
    p.add_argument("--hyperbolic_gnn", type=int, default=0,
                   help="Stage C: replace the Euclidean Token-GNN with an HGCN stack.")
    p.add_argument("--hyperbolic_readout", type=int, default=0,
                   help="Stage B: replace the Euclidean readout with a Poincare gyro-midpoint.")
    p.add_argument("--hyp_gnn_type", type=str, default="gcn", choices=["gcn", "gat"],
                   help="Stage C hyperbolic conv type when hyperbolic_gnn=1: gcn (HGCN) or gat (attention).")
    p.add_argument("--readout_clip", type=float, default=0.0,
                   help="Stage B: clip Euclidean feature norm before the exp map (0=off). Fixes boundary saturation.")
    p.add_argument("--readout_scale", type=int, default=0,
                   help="Stage B: use a learnable input scale before the exp map.")
    p.add_argument("--learnable_curvature", type=int, default=0,
                   help="Make the Poincare ball curvature c a trainable parameter.")
    p.add_argument("--gnn_input_clip", type=float, default=0.0,
                   help="Stage C: clip token norm before the entry exp map (0=off). Fixes boundary saturation.")
    p.add_argument("--gnn_input_scale", type=int, default=0,
                   help="Stage C: use a learnable input scale before the entry exp map.")
    p.add_argument("--gat_edge_attr", type=int, default=1,
                   help="1 (default, reproduces the paper): build the Euclidean GAT with "
                        "edge_dim=1 and feed it edge_attr. 0: drop the edge-attribute path "
                        "so the Euclidean and hyperbolic GATs differ in curvature ALONE. "
                        "HyperbolicGATConv ignores edge_weight, and the constant attribute "
                        "the binary graph supplies is not inert because GATConv adds it "
                        "before a leaky_relu -- so any Euclidean-vs-hyperbolic contrast run "
                        "at the default varies two things at once. Use 0 for the corrective "
                        "Stage C factorial.")
    p.add_argument("--arm", type=str, default="",
                   help="Optional label for the ablation arm (baseline|A|B|C|ABC); recorded in results CSV.")
    p.add_argument("--results_csv", type=str, default="",
                   help="If set, append a one-row result record to this CSV after the run.")
    p.add_argument("--run_tag", type=str, default="", help="Free-form label recorded in the results CSV.")
    p.add_argument("--git_push", type=int, default=0,
                   help="If 1, git add+commit+push the results CSV to GitHub after each run.")
    p.add_argument("--git_remote", type=str, default="origin", help="Git remote to push results to.")
    p.add_argument("--git_branch", type=str, default="", help="Git branch to push results to (default: current).")
    # Projection head
    p.add_argument("--proj_dim", type=int, default=256, help="If >0, apply linear projection to this dim before cosine.")
    # Labels
    p.add_argument("--label_scale", type=str, default="0_1", choices=["0_1", "-1_1", "raw"],
                   help="How to scale STS labels before regression.")
    p.add_argument("--precompute_hidden_states", type=int, default=0, help="Precompute hidden states")
    p.add_argument("--override_precompute", type=int, default=0, help="Override precompute")
    p.add_argument("--finetune_backbone", type=int, default=0, help="Valid when precompute is False and backbone should be finetuned")
    return p

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    backbone, dcls = load_backbone(args.model_name_or_path, max_length=args.max_length, decoder_cls_last_token=args.decoder_cls_last_token, task=args.task, hidden_layer=int(getattr(args, "hidden_layer", -1)))
    args.decoder_cls_last_token = dcls
    backbone.model.to(device)

    run_tasks(backbone, args, device)

if __name__ == "__main__":
    main()
