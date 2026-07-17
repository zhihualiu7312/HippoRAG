"""Train and run inference for a pairwise-learned diffusion policy.

This script provides a lightweight standalone entrypoint for the pairwise
ranking formulation of the learnable diffusion policy described in the
research plan.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hipporag_extensions import LearnedDiffusionPolicy, adaptive_personalized_pagerank


def build_demo_examples() -> List[Dict[str, object]]:
    return [
        {
            "query": "How does hybrid bonding affect reliability?",
            "seeds": {"seed": 1.0},
            "positive_node": "positive",
            "negative_node": "negative",
        },
        {
            "query": "What is FinFET and why is it used?",
            "seeds": {"seed": 1.0},
            "positive_node": "positive",
            "negative_node": "negative",
        },
    ]


def train(args: argparse.Namespace) -> LearnedDiffusionPolicy:
    policy = LearnedDiffusionPolicy(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        max_restart=args.max_restart,
        min_restart=args.min_restart,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
    )
    examples = build_demo_examples()
    policy.fit_pairwise(
        graph={
            "seed": {"positive": 1.0, "negative": 1.0},
            "positive": {},
            "negative": {},
        },
        examples=examples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    return policy


def run_inference(policy: LearnedDiffusionPolicy, query: str) -> Dict[str, object]:
    graph = {
        "seed": {"positive": 1.0, "negative": 1.0},
        "positive": {},
        "negative": {},
    }
    seeds = {"seed": 1.0}
    trained_policy = policy.predict(query, seeds)
    scores = adaptive_personalized_pagerank(graph, seeds, trained_policy)
    return {
        "query": query,
        "policy": {
            "restart": trained_policy.restart,
            "depth": trained_policy.depth,
            "relation_weights": dict(trained_policy.relation_weights),
            "node_gates": dict(trained_policy.node_gates),
        },
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and inference for pairwise-learned diffusion policy")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--min-restart", type=float, default=0.05)
    parser.add_argument("--max-restart", type=float, default=0.45)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--query", type=str, default="How does hybrid bonding affect reliability?")
    parser.add_argument("--output", type=str, default="outputs/learned_diffusion_policy_result.json")
    args = parser.parse_args()

    policy = train(args)
    result = run_inference(policy, args.query)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
