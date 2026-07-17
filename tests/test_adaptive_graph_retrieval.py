import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hipporag_extensions import (
    DiffusionPolicy,
    QueryInducedGraphConstructor,
    adaptive_personalized_pagerank,
)


def test_policy_from_query_increases_depth_for_complex_query():
    simple = DiffusionPolicy.from_query("What is FinFET?", {"finfet": 1.0})
    complex_policy = DiffusionPolicy.from_query(
        "How does Cu hybrid bonding reduce TSV stress across reliability mechanisms?",
        {"cu": 0.5, "bonding": 0.3, "tsv": 0.2},
    )

    assert complex_policy.depth > simple.depth
    assert complex_policy.restart > simple.restart


def test_query_induced_constructor_keeps_local_seed_neighborhood():
    graph = {
        "seed": {"concept": 1.0, "noise": 0.1},
        "concept": {"paper": 1.0},
        "paper": {},
        "noise": {"far": 1.0},
    }
    constructor = QueryInducedGraphConstructor(
        max_nodes=3,
        max_hops=2,
        scorer=lambda query, node: 0.0 if node == "noise" else 1.0,
        min_keep_score=0.5,
    )

    local_graph = constructor.construct("hybrid bonding", graph, {"seed": 1.0})

    assert set(local_graph) == {"seed", "concept", "paper"}
    assert "noise" not in local_graph["seed"]


def test_adaptive_pagerank_uses_relation_weights_and_gates():
    graph = {
        "seed": {"semantic_hit": 1.0, "citation_noise": 1.0},
        "semantic_hit": {},
        "citation_noise": {"far_noise": 1.0},
    }
    relations = {
        ("seed", "semantic_hit"): "semantic",
        ("seed", "citation_noise"): "citation",
    }
    policy = DiffusionPolicy(
        restart=0.2,
        depth=5,
        relation_weights={"semantic": 3.0, "citation": 0.1},
        node_gates={"citation_noise": 0.0},
    )

    scores = adaptive_personalized_pagerank(graph, {"seed": 1.0}, policy, edge_relations=relations)

    assert scores["semantic_hit"] > scores["citation_noise"]
    assert scores.get("far_noise", 0.0) == 0.0
    assert abs(sum(scores.values()) - 1.0) < 1e-9
