"""Train a learned MoG (Mixture of Granularity) policy.

The pipeline is split into two stages, controlled by --stage:

  1. gen   : Load the HippoRAG index, build pairwise training examples
             (query, query_embedding, seeds, positive_node, negative_node)
             and save them to {save_dir}/mog_training_examples.json.
             Requires an existing index (graph.pickle).

  2. train : Load {save_dir}/mog_training_examples.json AND the HippoRAG
             index, then train an MoGPolicy using **differentiable PPR**
             (torch sparse matrix, depth=10). All parameters (α, β, γ,
             τ_pass, τ_paper, τ_ent, damping) are trained end-to-end via
             backprop through the PPR iterations.

  3. all   : Run gen then train in a single invocation (default).

Usage:
    # Stage 1 — generate training data
    python tests/train_mog_policy.py \
        --config configs/samsung_heuristic_diffusion_paper.yaml \
        --dataset musique \
        --num_train 500 \
        --stage gen

    # Stage 2 — train from the generated data
    python tests/train_mog_policy.py \
        --config configs/samsung_heuristic_diffusion_paper.yaml \
        --dataset musique \
        --epochs 80 \
        --stage train

Prerequisite: indexing must be complete (graph.pickle exists).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "sk-lR8wA1pMmPqT4uXzYbB7dF9cE5")

from src.hipporag.HippoRAG import HippoRAG
from main import build_config, get_gold_docs, parse_args

from hipporag_extensions import MoGPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: data generation (unchanged)
# ---------------------------------------------------------------------------

def find_passage_vertex_idx(rag: HippoRAG, passage_text: str) -> Optional[int]:
    """Find the vertex index for a passage by its text content."""
    for passage_key in rag.passage_node_keys:
        row = rag.chunk_embedding_store.get_row(passage_key)
        if row and row["content"] == passage_text:
            return rag.node_name_to_vertex_idx.get(passage_key)
    return None


def build_training_examples(
    rag: HippoRAG,
    samples: List[dict],
    gold_docs: List[List[str]],
) -> List[Dict]:
    """Build training examples from queries.

    For each query:
    - Run get_seeds_and_local_graph to get seed_scores
    - Get query embedding
    - Find positive_node: GT supporting passage vertex idx
    - Find negative_node: a random non-GT passage node from seed_scores
    """
    examples = []
    skipped = 0

    for i, sample in enumerate(samples):
        query = sample["question"]
        logger.info(f"[{i+1}/{len(samples)}] Building example for: {query[:80]}...")

        seed_scores, _ = rag.get_seeds_and_local_graph(query)
        if seed_scores is None or not seed_scores:
            logger.warning(f"  Skipping: no seeds")
            skipped += 1
            continue

        # Get query embedding
        query_emb = rag.query_to_embedding.get('passage', {}).get(query)
        if query_emb is None:
            logger.warning(f"  Skipping: no query embedding")
            skipped += 1
            continue

        # Find positive node: GT passage vertex idx
        gold_passages = gold_docs[i]
        positive_nodes = []
        for gold_text in gold_passages:
            pos_idx = find_passage_vertex_idx(rag, gold_text)
            if pos_idx is not None:
                positive_nodes.append(pos_idx)

        if not positive_nodes:
            logger.warning(f"  Skipping: GT passage not found in graph")
            skipped += 1
            continue

        # Find negative candidates: passage nodes in seed_scores that are not positive
        positive_set = set(positive_nodes)
        passage_idx_set = set(rag.passage_node_idxs)
        negative_candidates = [
            idx for idx in seed_scores.keys()
            if idx in passage_idx_set and idx not in positive_set
        ]

        if not negative_candidates:
            logger.warning(f"  Skipping: no negative candidates")
            skipped += 1
            continue

        # Use hard negatives: rank by seed score (descending) and pick top-k
        # This selects passages that are semantically similar but not GT
        negative_candidates.sort(key=lambda idx: seed_scores.get(idx, 0.0), reverse=True)
        top_negatives = negative_candidates[:5]  # keep top-5 hardest negatives

        # Generate one example per positive node, using a hard negative
        for pos_node in positive_nodes:
            neg_node = random.choice(top_negatives)
            examples.append({
                "query": query,
                "query_embedding": query_emb.tolist(),
                "seeds": {str(k): float(v) for k, v in seed_scores.items()},
                "positive_node": pos_node,
                "negative_node": neg_node,
            })

        logger.info(f"  Created {len(positive_nodes)} examples (pos={positive_nodes}, neg={neg_node})")

    logger.info(f"Built {len(examples)} examples, skipped {skipped} queries")
    return examples


# ---------------------------------------------------------------------------
# Stage 2: differentiable PPR training
# ---------------------------------------------------------------------------

class MoGMLP(nn.Module):
    """Torch MLP: query_emb → (alpha, beta, gamma, tau_pass, tau_paper, tau_ent, damping).

    Architecture: Linear(input, hidden) → ReLU → Linear(hidden, 7)
    - First 3 outputs: softmax → (alpha, beta, gamma)
    - Next 3 outputs: sigmoid → (tau_pass, tau_paper, tau_ent)
    - Last output: sigmoid → damping in [0.3, 0.95]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        out = self.fc2(h)
        # Softmax for (alpha, beta, gamma)
        abg = torch.softmax(out[:3], dim=0)
        # Sigmoid for thresholds
        taus = torch.sigmoid(out[3:6])
        # Sigmoid for damping, mapped to [0.3, 0.95]
        damping = 0.3 + 0.65 * torch.sigmoid(out[6:7])
        return abg, taus, damping


def build_sparse_adjacency(graph, device: torch.device) -> torch.Tensor:
    """Build a row-normalised sparse adjacency matrix from an igraph graph.

    Returns:
        torch.sparse_coo_tensor of shape (N, N), row-normalised.
    """
    n = graph.vcount()
    edges = graph.get_edgelist()
    if not edges:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            (n, n),
        )

    src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)

    # Edge weights
    if "weight" in graph.es.attribute_names():
        w = torch.tensor(graph.es["weight"], dtype=torch.float32, device=device)
    else:
        w = torch.ones(len(edges), dtype=torch.float32, device=device)

    # Symmetrise (undirected): add both directions
    src_both = torch.cat([src, dst])
    dst_both = torch.cat([dst, src])
    w_both = torch.cat([w, w])

    # Row-normalise: divide by out-degree
    out_deg = torch.zeros(n, dtype=torch.float32, device=device)
    out_deg.scatter_add_(0, src_both, w_both)
    out_deg = torch.clamp(out_deg, min=1e-8)
    w_norm = w_both / out_deg[src_both]

    indices = torch.stack([dst_both, src_both])  # (row, col) for M @ v
    M = torch.sparse_coo_tensor(
        indices, w_norm, (n, n)
    ).coalesce()

    logger.info(
        f"Built sparse adjacency: {n} nodes, {len(edges)} edges, "
        f"{M._nnz()} non-zeros"
    )
    return M


def differentiable_ppr(
    M_sparse: torch.Tensor,
    seeds_vec: torch.Tensor,
    damping: torch.Tensor,
    gates: torch.Tensor,
    depth: int = 10,
) -> torch.Tensor:
    """Differentiable PPR via sparse matrix multiply.

    Iterates: scores = damping * seeds + (1-damping) * M @ (scores * gates)

    Args:
        M_sparse: (N, N) row-normalised adjacency (torch sparse)
        seeds_vec: (N,) personalization vector (torch)
        damping: scalar in [0.3, 0.95] (torch, differentiable)
        gates: (N,) node gates (torch, differentiable)
        depth: PPR iterations

    Returns:
        scores: (N,) PPR scores (fully differentiable)
    """
    scores = seeds_vec.clone()
    for _ in range(depth):
        gated = scores * gates
        propagated = torch.sparse.mm(M_sparse, gated.unsqueeze(1)).squeeze(1)
        scores = damping * seeds_vec + (1.0 - damping) * propagated
    return scores


def compute_gates_torch(
    abg: torch.Tensor,
    taus: torch.Tensor,
    query_emb: torch.Tensor,
    all_embs: torch.Tensor,
    node_type_ids: torch.Tensor,
    k: float = 5.0,
) -> torch.Tensor:
    """Compute differentiable node gates.

    Args:
        abg: (3,) routing weights (alpha, beta, gamma)
        taus: (3,) thresholds (tau_pass, tau_paper, tau_ent)
        query_emb: (D,) query embedding
        all_embs: (N, D) all node embeddings
        node_type_ids: (N,) type id per node (0=passage, 1=paper, 2=entity)
        k: sigmoid slope

    Returns:
        gates: (N,) differentiable node gates
    """
    # Cosine similarity (all_embs assumed normalised)
    sims = torch.matmul(all_embs, query_emb)  # (N,)

    # Per-type gate: σ(k * (sim - τ)) × mog_weight
    gates = torch.zeros_like(sims)
    for t in range(3):
        mask = (node_type_ids == t)
        if mask.any():
            gate_t = torch.sigmoid(k * (sims[mask] - taus[t])) * abg[t]
            gates[mask] = gate_t

    return gates


def train_mog_policy_differentiable(
    examples: List[Dict],
    rag: HippoRAG,
    input_dim: int = 4096,
    hidden_dim: int = 32,
    epochs: int = 80,
    learning_rate: float = 0.001,
    device: str = "auto",
    ppr_depth: int = 10,
    gate_k: float = 5.0,
    margin: float = 0.1,
    output_path: str = None,
) -> MoGPolicy:
    """Train MoGPolicy using differentiable PPR (end-to-end backprop).

    All parameters (α, β, γ, τ_pass, τ_paper, τ_ent, damping) are trained
    via backprop through the PPR iterations.
    """
    dev = get_device(device)
    logger.info(
        f"Training MoGPolicy (differentiable PPR, device={dev}): "
        f"{len(examples)} examples, {epochs} epochs, input_dim={input_dim}, "
        f"ppr_depth={ppr_depth}"
    )

    n_nodes = rag.graph.vcount()
    logger.info(f"  Graph: {n_nodes} nodes, {rag.graph.ecount()} edges")

    # Build node mappings (passage_node_idxs etc. are set in prepare_retrieval_objects(), not __init__)
    logger.info("  Building node mappings via prepare_retrieval_objects()...")
    rag.prepare_retrieval_objects()

    # --- Build sparse adjacency matrix ---
    logger.info("  Building sparse adjacency matrix...")
    M_sparse = build_sparse_adjacency(rag.graph, dev)

    # --- Build node type ids and embeddings ---
    logger.info("  Building node type ids and embeddings...")
    node_type_ids = torch.zeros(n_nodes, dtype=torch.long, device=dev)
    # 0=passage, 1=paper, 2=entity

    for idx in rag.passage_node_idxs:
        if idx < n_nodes:
            node_type_ids[idx] = 0
    for idx in getattr(rag, 'paper_node_idxs', []):
        if idx < n_nodes:
            node_type_ids[idx] = 1
    for idx in rag.entity_node_idxs:
        if idx < n_nodes:
            node_type_ids[idx] = 2

    # Build all node embeddings (N, D)
    all_embs = torch.zeros(n_nodes, input_dim, dtype=torch.float32, device=dev)
    if rag.passage_embeddings is not None and len(rag.passage_embeddings) > 0:
        for i, idx in enumerate(rag.passage_node_idxs):
            if idx < n_nodes:
                all_embs[idx] = torch.tensor(rag.passage_embeddings[i], dtype=torch.float32, device=dev)
    if hasattr(rag, 'paper_embeddings') and rag.paper_embeddings is not None and len(rag.paper_embeddings) > 0:
        for i, idx in enumerate(getattr(rag, 'paper_node_idxs', [])):
            if idx < n_nodes:
                all_embs[idx] = torch.tensor(rag.paper_embeddings[i], dtype=torch.float32, device=dev)
    if rag.entity_embeddings is not None and len(rag.entity_embeddings) > 0:
        for i, idx in enumerate(rag.entity_node_idxs):
            if idx < n_nodes:
                all_embs[idx] = torch.tensor(rag.entity_embeddings[i], dtype=torch.float32, device=dev)

    # Normalise embeddings
    norms = all_embs.norm(dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-8)
    all_embs = all_embs / norms

    logger.info(f"  Node type distribution: passage={int((node_type_ids==0).sum())}, "
                f"paper={int((node_type_ids==1).sum())}, "
                f"entity={int((node_type_ids==2).sum())}")

    # --- Preprocess training examples ---
    preprocessed = []
    for ex in examples:
        seeds = {int(k): float(v) for k, v in ex["seeds"].items()}
        pos_idx = ex["positive_node"]
        neg_idx = ex["negative_node"]
        query_emb = torch.tensor(ex["query_embedding"], dtype=torch.float32, device=dev)
        # Normalise query embedding
        q_norm = query_emb.norm()
        if q_norm > 0:
            query_emb = query_emb / q_norm

        # Build seeds vector
        seeds_vec = torch.zeros(n_nodes, dtype=torch.float32, device=dev)
        for idx, score in seeds.items():
            if 0 <= idx < n_nodes:
                seeds_vec[idx] = score
        total = seeds_vec.sum()
        if total > 0:
            seeds_vec = seeds_vec / total

        preprocessed.append({
            "seeds_vec": seeds_vec,
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "query_emb": query_emb,
        })

    logger.info(f"  Preprocessed {len(preprocessed)} examples for differentiable training")

    # --- Split: fixed eval set (first 100) + train set ---
    eval_size = min(100, len(preprocessed) // 5)
    eval_set = preprocessed[:eval_size]
    train_set = preprocessed[eval_size:]
    logger.info(f"  Split: {len(train_set)} train, {len(eval_set)} eval (fixed)")

    # --- Initialise torch MLP ---
    mlp = MoGMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    # StepLR: halve lr every step_size epochs (e.g. 0.003 → 0.0015 → 0.00075)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(epochs // 3, 1), gamma=0.5
    )

    def _save_checkpoint(epoch_num: int):
        """Save current MLP weights as a MoGPolicy checkpoint."""
        ckpt = MoGPolicy(input_dim=input_dim, hidden_dim=hidden_dim)
        ckpt._W1 = mlp.fc1.weight.data.cpu().numpy().T.copy()
        ckpt._b1 = mlp.fc1.bias.data.cpu().numpy().copy()
        ckpt._W2 = mlp.fc2.weight.data.cpu().numpy().T.copy()
        ckpt._b2 = mlp.fc2.bias.data.cpu().numpy().copy()
        return ckpt

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        epoch_start = time.time()

        for item in train_set:
            seeds_vec = item["seeds_vec"]
            pos_idx = item["pos_idx"]
            neg_idx = item["neg_idx"]
            query_emb = item["query_emb"]

            # Forward: query_emb → MLP → (α, β, γ, τ, damping)
            abg, taus, damping = mlp(query_emb)

            # Compute differentiable gates
            gates = compute_gates_torch(
                abg, taus, query_emb, all_embs, node_type_ids, k=gate_k
            )

            # Differentiable PPR
            scores = differentiable_ppr(
                M_sparse, seeds_vec, damping, gates, depth=ppr_depth
            )

            # Normalise scores so margin is meaningful (PPR scores are ~1e-4)
            score_max = scores.max().clamp(min=1e-8)
            pos_score = scores[pos_idx] / score_max
            neg_score = scores[neg_idx] / score_max

            # Pairwise hinge loss (margin in normalised [0,1] space)
            loss = F.relu(margin - (pos_score - neg_score))

            if (pos_score - neg_score).item() > 0:
                correct += 1

            total_loss += float(loss.item())

            # Backward (end-to-end, fully differentiable)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        epoch_time = time.time() - epoch_start
        accuracy = correct / len(train_set) if train_set else 0
        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{epochs}: loss={total_loss:.4f}, "
                f"train_acc={accuracy:.4f} ({correct}/{len(train_set)}), "
                f"lr={current_lr:.6f}, time={epoch_time:.1f}s"
            )

        # Checkpoint + eval every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_policy = _save_checkpoint(epoch + 1)

            # Save checkpoint to disk
            if output_path:
                ckpt_path = output_path.replace(".pkl", f"_epoch{epoch+1}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(ckpt_policy, f)

            # Eval on fixed eval_set (no_grad)
            eval_correct = 0
            eval_loss = 0.0
            with torch.no_grad():
                for item in eval_set:
                    abg, taus, damping = mlp(item["query_emb"])
                    gates = compute_gates_torch(
                        abg, taus, item["query_emb"], all_embs, node_type_ids, k=gate_k
                    )
                    scores = differentiable_ppr(
                        M_sparse, item["seeds_vec"], damping, gates, depth=ppr_depth
                    )
                    score_max = scores.max().clamp(min=1e-8)
                    pos_score = scores[item["pos_idx"]] / score_max
                    neg_score = scores[item["neg_idx"]] / score_max
                    loss = F.relu(margin - (pos_score - neg_score))
                    if (pos_score - neg_score).item() > 0:
                        eval_correct += 1
                    eval_loss += float(loss.item())

            eval_acc = eval_correct / len(eval_set) if eval_set else 0
            logger.info(
                f"  [Checkpoint] Epoch {epoch+1}: "
                f"eval_loss={eval_loss:.4f}, eval_acc={eval_acc:.4f} "
                f"({eval_correct}/{len(eval_set)}), lr={current_lr:.6f}"
            )

    # --- Copy weights back to numpy MoGPolicy ---
    policy = _save_checkpoint(epochs)

    logger.info("Copied torch weights back to numpy MoGPolicy for inference.")
    return policy


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_policy(policy: MoGPolicy, path: str) -> None:
    """Save a trained MoG policy to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(policy, f)
    logger.info(f"Saved trained MoG policy to {path}")


def load_training_examples(path: str) -> List[Dict]:
    """Load pre-built training examples from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    logger.info(f"Loaded {len(examples)} training examples from {path}")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train learned MoG policy")
    parser.add_argument("--config", default="configs/samsung_heuristic_diffusion_paper.yaml")
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--num_train", type=int, default=500, help="Number of training queries (first N)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--output", default=None, help="Output path for trained policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device: auto (detect GPU), cuda, or cpu (default: auto)",
    )
    parser.add_argument(
        "--stage",
        choices=["gen", "train", "all"],
        default="all",
        help="gen: only build & save training_examples.json; "
             "train: only train from existing training_examples.json; "
             "all: do both in one run (default)",
    )
    parser.add_argument(
        "--ppr_depth",
        type=int,
        default=10,
        help="PPR iteration depth for differentiable PPR (default: 10)",
    )
    parser.add_argument(
        "--gate_k",
        type=float,
        default=5.0,
        help="Sigmoid slope for node gates (default: 5.0)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Hinge loss margin (default: 0.01)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build config
    sys.argv = ["train", "--config", args.config, "--dataset", args.dataset]
    cli_args = parse_args()
    config = build_config(cli_args)

    # Default training data path
    training_data_path = os.path.join(config.save_dir, "mog_training_examples.json")

    # --------------------------------------------------------------
    # Stage: gen (or all) — build training examples via HippoRAG
    # --------------------------------------------------------------
    if args.stage in ("gen", "all"):
        query_path = os.path.join(PROJECT_ROOT, f"reproduce/dataset/{args.dataset}.json")
        with open(query_path) as f:
            samples = json.load(f)
        samples = samples[:args.num_train]
        logger.info(f"Loaded {len(samples)} training samples from {query_path}")

        gold_docs = get_gold_docs(samples, args.dataset)

        # Enable diffusion config for get_seeds_and_local_graph to work
        config.use_query_aware_diffusion = True
        config.use_query_induced_graph = True

        rag = HippoRAG(global_config=config)

        # Verify index exists
        graph_path = os.path.join(rag.working_dir, "graph.pickle")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Index not found at {graph_path}")

        # Build training examples
        logger.info("Building training examples...")
        build_start = time.time()
        examples = build_training_examples(rag, samples, gold_docs)
        build_time = time.time() - build_start
        logger.info(f"Built {len(examples)} examples in {build_time:.1f}s")

        # Save training examples to JSON
        os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
        with open(training_data_path, "w") as f:
            json.dump(examples, f)
        logger.info(f"Saved {len(examples)} training examples to {training_data_path}")

        if not examples:
            raise ValueError("No training examples were built. Check your index and data.")

        if args.stage == "gen":
            logger.info("Stage 'gen' finished. Run with --stage train to train the policy.")
            return

    # --------------------------------------------------------------
    # Stage: train (or all) — train policy from examples
    # --------------------------------------------------------------
    if args.stage in ("train", "all"):
        if args.stage == "train":
            if not os.path.exists(training_data_path):
                raise FileNotFoundError(
                    f"Training examples not found at {training_data_path}. "
                    "Run with --stage gen first."
                )
            examples = load_training_examples(training_data_path)

        # Determine input_dim from first example
        input_dim = len(examples[0]["query_embedding"]) if examples else 4096
        logger.info(f"Query embedding dimension: {input_dim}")

        # Load HippoRAG for differentiable PPR
        logger.info("Loading HippoRAG index for differentiable PPR training...")
        config.use_query_aware_diffusion = True
        config.use_query_induced_graph = True
        rag = HippoRAG(global_config=config)

        graph_path = os.path.join(rag.working_dir, "graph.pickle")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Index not found at {graph_path}")

        output_path = args.output or os.path.join(
            config.save_dir, "mog_policy.pkl"
        )

        logger.info(f"Training MoG policy (differentiable PPR, device={args.device})...")
        train_start = time.time()
        policy = train_mog_policy_differentiable(
            examples,
            rag=rag,
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            ppr_depth=args.ppr_depth,
            gate_k=args.gate_k,
            margin=args.margin,
            output_path=output_path,
        )

        train_time = time.time() - train_start
        logger.info(f"Training completed in {train_time:.1f}s")

        save_policy(policy, output_path)
        logger.info(f"Done! MoG policy saved to {output_path}")


if __name__ == "__main__":
    main()
