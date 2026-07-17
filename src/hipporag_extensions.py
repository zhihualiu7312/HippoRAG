"""Experimental HippoRAG extensions for query-adaptive graph retrieval."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import exp
from typing import Callable, Hashable, Iterable, List, Mapping, MutableMapping, Protocol, Sequence

Node = Hashable
Graph = Mapping[Node, Mapping[Node, float]]


class PolicyModel(Protocol):
    """Protocol for optional learned policy backends."""

    def predict(self, query: str, seeds: Mapping[Node, float]) -> Mapping[str, object]:
        """Return policy fields such as restart, depth, relation_weights, or gates."""


@dataclass(slots=True)
class DiffusionPolicy:
    """Controls query-adaptive diffusion on a graph."""

    restart: float = 0.15
    depth: int = 20
    relation_weights: Mapping[str, float] = field(default_factory=dict)
    node_gates: Mapping[Node, float] = field(default_factory=dict)
    default_gate: float = 1.0

    @classmethod
    def from_query(
        cls,
        query: str,
        seeds: Mapping[Node, float],
        model: PolicyModel | None = None,
        *,
        min_restart: float = 0.05,
        max_restart: float = 0.45,
        min_depth: int = 4,
        max_depth: int = 20,
    ) -> "DiffusionPolicy":
        """Create a policy from a learned model or deterministic query features."""

        if model is not None:
            raw = dict(model.predict(query, seeds))
            return cls(
                restart=float(raw.get("restart", cls.restart)),
                depth=int(raw.get("depth", cls.depth)),
                relation_weights=dict(raw.get("relation_weights", {})),
                node_gates=dict(raw.get("node_gates", {})),
                default_gate=float(raw.get("default_gate", 1.0)),
            ).clamped(min_restart, max_restart, min_depth, max_depth)

        token_count = max(1, len(query.split()))
        seed_entropy_proxy = len([score for score in seeds.values() if score > 0])
        complexity = min(1.0, (token_count / 18.0 + seed_entropy_proxy / 12.0) / 2.0)
        restart = min_restart + (max_restart - min_restart) * complexity
        depth = round(min_depth + (max_depth - min_depth) * complexity)
        return cls(restart=restart, depth=depth).clamped(
            min_restart, max_restart, min_depth, max_depth
        )

    def clamped(
        self,
        min_restart: float = 0.05,
        max_restart: float = 0.45,
        min_depth: int = 1,
        max_depth: int = 100,
    ) -> "DiffusionPolicy":
        """Return a validated policy bounded for stable diffusion."""

        restart = min(max(self.restart, min_restart), max_restart)
        depth = min(max(int(self.depth), min_depth), max_depth)
        return DiffusionPolicy(
            restart=restart,
            depth=depth,
            relation_weights=dict(self.relation_weights),
            node_gates=dict(self.node_gates),
            default_gate=self.default_gate,
        )

    def gate_for(self, node: Node) -> float:
        """Return the propagation gate for a node in [0, 1]."""

        return min(1.0, max(0.0, float(self.node_gates.get(node, self.default_gate))))


@dataclass
class LearnedDiffusionPolicy:
    """A lightweight learnable diffusion policy trained with pairwise ranking loss."""

    hidden_dim: int = 8
    learning_rate: float = 0.05
    min_restart: float = 0.05
    max_restart: float = 0.45
    min_depth: int = 4
    max_depth: int = 20
    restart_weight: float = 0.5
    depth_weight: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "_query_node_biases", {})
        object.__setattr__(self, "_relation_weights", {})

    def fit_pairwise(
        self,
        graph: Graph,
        examples: Sequence[Mapping[str, object]],
        epochs: int = 50,
        learning_rate: float | None = None,
    ) -> None:
        """Fit the policy using a simple pairwise ranking objective over diffusion outcomes."""

        if learning_rate is not None:
            self.learning_rate = learning_rate

        for _ in range(epochs):
            for example in examples:
                query = str(example.get("query", ""))
                seeds = dict(example.get("seeds", {}))
                positive_node = str(example.get("positive_node", ""))
                negative_node = str(example.get("negative_node", ""))
                edge_relations = dict(example.get("edge_relations", {}))

                policy = self.predict(query, seeds)
                scores = adaptive_personalized_pagerank(graph, seeds, policy, edge_relations=edge_relations or None)
                positive_score = scores.get(positive_node, 0.0)
                negative_score = scores.get(negative_node, 0.0)

                margin = positive_score - negative_score
                if margin <= 0.0:
                    update_step = self.learning_rate * 0.2
                else:
                    update_step = self.learning_rate * 0.01

                if positive_node:
                    self._query_node_biases[(query, positive_node)] = self._query_node_biases.get((query, positive_node), 0.0) + update_step
                if negative_node:
                    self._query_node_biases[(query, negative_node)] = self._query_node_biases.get((query, negative_node), 0.0) - update_step

                positive_relation = self._extract_relation(edge_relations, positive_node)
                negative_relation = self._extract_relation(edge_relations, negative_node)
                if positive_relation is not None:
                    current = self._relation_weights.get(positive_relation, 1.0)
                    self._relation_weights[positive_relation] = max(0.1, min(3.0, current + update_step))
                if negative_relation is not None:
                    current = self._relation_weights.get(negative_relation, 1.0)
                    self._relation_weights[negative_relation] = max(0.1, min(3.0, current - update_step))

    def predict(self, query: str, seeds: Mapping[Node, float]) -> DiffusionPolicy:
        """Return a diffusion policy derived from the learned query-node biases and relation weights."""

        token_count = max(1, len(query.split()))
        seed_entropy_proxy = len([score for score in seeds.values() if score > 0])
        complexity = min(1.0, (token_count / 18.0 + seed_entropy_proxy / 12.0) / 2.0)

        restart = self.min_restart + (self.max_restart - self.min_restart) * complexity
        depth = round(self.min_depth + (self.max_depth - self.min_depth) * complexity)

        node_gates: dict[Node, float] = {}
        for node in set(self._query_node_biases):
            if node[0] == query:
                bias = self._query_node_biases[node]
                node_gates[node[1]] = max(0.0, min(1.0, 1.0 + bias))

        relation_weights = {
            relation: max(0.1, min(3.0, float(weight)))
            for relation, weight in self._relation_weights.items()
        }

        return DiffusionPolicy(
            restart=restart,
            depth=depth,
            relation_weights=relation_weights,
            node_gates=node_gates,
            default_gate=1.0,
        ).clamped(self.min_restart, self.max_restart, self.min_depth, self.max_depth)

    def _extract_relation(
        self,
        edge_relations: Mapping[tuple[Node, Node], str],
        node: Node,
    ) -> str | None:
        for (src, dst), relation in edge_relations.items():
            if dst == node:
                return str(relation)
        return None


@dataclass(slots=True)
class QueryInducedGraphConstructor:
    """Builds a query-specific local graph before diffusion."""

    max_nodes: int = 5_000
    max_hops: int = 2
    min_keep_score: float = 0.0
    scorer: Callable[[str, Node], float] | None = None

    def construct(
        self,
        query: str,
        graph: Graph,
        seeds: Mapping[Node, float],
    ) -> dict[Node, dict[Node, float]]:
        """Return an induced local graph containing selected seed-neighborhood nodes."""

        selected: set[Node] = set()
        frontier: deque[tuple[Node, int]] = deque()

        for node, _ in sorted(seeds.items(), key=lambda item: item[1], reverse=True):
            if node in graph and self._keep(query, node):
                selected.add(node)
                frontier.append((node, 0))
            if len(selected) >= self.max_nodes:
                break

        while frontier and len(selected) < self.max_nodes:
            node, hop = frontier.popleft()
            if hop >= self.max_hops:
                continue

            neighbors = sorted(graph.get(node, {}).items(), key=lambda item: item[1], reverse=True)
            for neighbor, _ in neighbors:
                if len(selected) >= self.max_nodes:
                    break
                if neighbor not in selected and self._keep(query, neighbor):
                    selected.add(neighbor)
                    frontier.append((neighbor, hop + 1))

        return {
            node: {
                neighbor: weight
                for neighbor, weight in graph.get(node, {}).items()
                if neighbor in selected
            }
            for node in selected
        }

    def _keep(self, query: str, node: Node) -> bool:
        if self.scorer is None:
            return True
        return self.scorer(query, node) >= self.min_keep_score


def adaptive_personalized_pagerank(
    graph: Graph,
    seeds: Mapping[Node, float],
    policy: DiffusionPolicy,
    *,
    edge_relations: Mapping[tuple[Node, Node], str] | None = None,
) -> dict[Node, float]:
    """Run PPR with adaptive restart, depth, relation weights, and node gates."""

    if not seeds:
        return {}

    personalization = _normalize(seeds)
    scores: dict[Node, float] = dict(personalization)
    nodes = set(graph) | set(personalization)
    restart = policy.restart

    for _ in range(policy.depth):
        propagated: MutableMapping[Node, float] = defaultdict(float)
        for node in nodes:
            node_score = scores.get(node, 0.0) * policy.gate_for(node)
            if node_score <= 0.0:
                continue

            weighted_neighbors = _weighted_neighbors(graph.get(node, {}), node, policy, edge_relations)
            total = sum(weighted_neighbors.values())
            if total <= 0.0:
                propagated[node] += node_score
                continue

            for neighbor, weight in weighted_neighbors.items():
                propagated[neighbor] += node_score * weight / total

        nodes.update(propagated)
        scores = {
            node: restart * personalization.get(node, 0.0) + (1.0 - restart) * propagated.get(node, 0.0)
            for node in nodes
        }

    return _normalize(scores)


def _weighted_neighbors(
    neighbors: Mapping[Node, float],
    node: Node,
    policy: DiffusionPolicy,
    edge_relations: Mapping[tuple[Node, Node], str] | None,
) -> dict[Node, float]:
    weighted: dict[Node, float] = {}
    for neighbor, weight in neighbors.items():
        relation_weight = 1.0
        if edge_relations is not None:
            relation = edge_relations.get((node, neighbor))
            relation_weight = float(policy.relation_weights.get(relation, 1.0))
        adjusted = float(weight) * relation_weight
        if adjusted > 0.0:
            weighted[neighbor] = adjusted
    return weighted


def _normalize(scores: Mapping[Node, float]) -> dict[Node, float]:
    positive = {node: max(0.0, float(score)) for node, score in scores.items()}
    total = sum(positive.values())
    if total <= 0.0:
        return {node: 0.0 for node in positive}
    return {node: score / total for node, score in positive.items()}


def sigmoid_keep_scorer(query_embedding: Iterable[float], node_embedding: Iterable[float]) -> float:
    """Small helper for learned constructors that threshold embedding similarity."""

    dot = sum(q * n for q, n in zip(query_embedding, node_embedding))
    return 1.0 / (1.0 + exp(-dot))


__all__ = [
    "DiffusionPolicy",
    "LearnedDiffusionPolicy",
    "QueryInducedGraphConstructor",
    "adaptive_personalized_pagerank",
    "sigmoid_keep_scorer",
]
