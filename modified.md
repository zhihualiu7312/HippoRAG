요지

A learned MLP policy predicts routing weights, gate thresholds, gate slopes and PPR damping, replacing all hand-tuned heuristics with a single data-driven policy.
Multi-granularity gates apply per-type sigmoid filtering to suppress noise propagation from irrelevant nodes.
LearnedPPR routing dynamically allocates granularity-specific importance weights based on the query embedding, emphasizing the most relevant node type per query





확보
가능
권리
범위
1. A retrieval method using a tri-granular knowledge graph (entity-passage-paper) with per-type sigmoid gating that filters irrelevant nodes before PPR propagation.
2. A learned MLP policy that predicts routing weights, per-type gate thresholds, per-type gate slopes, and PPR damping factor from the query embedding, unifying all adaptive mechanisms into a single learned policy.
3. Gated PPR that applies per-type sigmoid gates at every iteration step, continuously suppressing noise propagation from irrelevant nodes.
4. Differentiable PPR via sparse matrix operations enabling end-to-end training of all routing and gating parameters through pairwise hinge loss.


출원
사유
  Achieving world-first differentiable query-aware graph diffusion system to maximize retrieval accuracy in knowledge-graph-based retrieval.

선행과의
차이점
PPR damping: Fixed for all queries vs Predicted by learned MLP policy
Node filtering: Uniform propagation vs Per-type sigmoid gates with independent thresholds and slopes
Granularity routing: Equal weights vs Learned MLP policy prediction
Noise suppression: None vs Sigmoid gates block irrelevant nodes before PPR propagation


Patent technology domain
Our method is a kind of Graph-based information retrieval using Personalized PageRank (PPR)
We proposes a query-aware multi-granularity gated with learned PPR policy network for retrieval.
    - LearnedPPR policy replaces static, hand-tuned retrieval heuristics with a learned policy that dynamically predicts multi-granular routing weights, per-type gate thresholds, per-type gate slopes, and PPR damping factor, completely replacing hand-crafted heuristic rules with data-driven optimization.
    - Gated PPR applies per-type sigmoid gates at every PPR iteration step, continuously suppressing signal propagation from irrelevant nodes and preventing noise cascade through the graph.


Scenario: Retrieval from private semiconductor data is critical
Knowledge graph augmented generation (GraphRAG) can greatly help the proprietary knowledge understanding and the cross-layer intelligence extraction of the fab.
Improved and more efficient materials engineering and process recipe development
Better management of private enterprise assets, intellectual property, and experimental logs, including a precise understanding of their multi-granular interdependencies to maximize yield and R&D throughput
Automated cross-document reasoning, assessing as-simulated physical properties to as-fabricated wafer metrics

LearnedPPR can support adaptive semantic routing for R&D on the discovery of next-generation materials and device physics.
Simulation and retrieval of atomic-to-fab scale workflows in a highly data-secure and cost-effective way

Motivation - Why LearnedPPR?

Problem: Graph-based retrieval systems (e.g., HippoRAG) use Personalized PageRank (PPR) on knowledge graphs for multi-hop reasoning, but suffer from three critical limitations:

Consequence: Fixed, granularity-agnostic PPR causes noise to propagate indiscriminately across the knowledge graph, degrading retrieval precision at the top-k range that matters most for downstream generation. Without query-adaptive control over propagation depth and node-level filtering, graph-based retrieval systems cannot distinguish relevant multi-hop evidence from structural noise, fundamentally limiting their scalability to enterprise-scale knowledge graphs.

Application: Semiconductor enterprises possess vast proprietary data (equipment manuals, process recipes, defect reports, yield data, FAB operation logs) requiring multi-hop reasoning across heterogeneous systems. Existing retrieval systems fail on dense proprietary knowledge graphs, causing critical evidence to be submerged by noise.

Summary of the Invention 
       The present invention discloses a differentiable query-aware graph diffusion method and system for multi-hop retrieval, the core of which lies in the LearnedPPR framework with a full-path differentiable soft-gating mechanism.

Core Idea:  Replace fixed, granularity-agnostic PPR with a learned, query-adaptive, multi-granular diffusion policy.

Multi-granularity gates (per-type sigmoid filtering)
Applies independent sigmoid gates with per-type thresholds and slopes to entity, passage, and paper nodes, suppressing noise propagation from irrelevant high-degree entities before PPR iteration.


2. LearnedPPR policy
A lightweight MLP model predicts routing weights, gate thresholds, gate slopes, and PPR damping, unifying all adaptive mechanisms into a single learned policy.

3. Differentiable PPR (end-to-end training via hinge loss)
Formulates PPR as differentiable sparse matrix operations, enabling all 10 routing and gating parameters to be trained end-to-end through pairwise hinge loss with hard negative mining.



Three Key Effects:

1. Noise suppression: Per-type sigmoid gates applied at every PPR iteration step continuously block irrelevant nodes, producing largest gains at R@5–R@20 where noise cascade is most damaging.

2. Learned query-adaptive control: A single MLP predicts routing weights, gate thresholds/slopes, and PPR damping from the query embedding, so each query receives its own optimal propagation policy without manual tuning.

3. End-to-end optimization: Differentiable PPR via sparse matrix operations enables all parameters to be trained jointly through pairwise hinge loss with hard negative mining, achieving +12.0% R@5, +11.7% R@10 over baseline HippoRAG2.


Invention Details (1/6)
Tri-Granular Knowledge Graph — Difference from HippoRAG

What learnedPPR adds:
Paper Nodes: For each document, passages sharing the same title are grouped, LLM generates a summary and encodes, then creates a Paper node as a macro-level structural anchor
Passage-Paper Edges: Weighted by cosine similarity between passage and paper embeddings:

Paper Node in PPR : Paper nodes participate in PPR propagation with an independent weight λ_paper, contributing document-level evidence alongside entity-level and passage-level evidence

Why it matters: 
In HippoRAG, passages from the same document are disconnected unless they share entities — "missing the forest for the trees." Paper nodes bind related passages through document-level summaries, enabling thematic queries to locate relevant documents first, then propagate to constituent passages.



Multi-Granularity Gates:
Per-type independent parameters: Each granularity (entity, passage, paper) has its own threshold and slope, allowing fine-grained control over what passes through
Sigmoid gating: Nodes below its own threshold receive gate value and will be blocked from PPR propagation; nodes above threshold receive gate value will be fully propagated
Applied to reset vector: Gate values multiply the PPR reset vector, not the adjacency matrix — this suppresses noise at the source before propagation begins

Key insight: The gate is a soft filter — it doesn‘t hard-delete nodes, but scales their contribution to the PPR reset vector. Nodes with low query relevance receive near-zero reset mass, so PPR barely propagates through them. This suppresses noise cascade at the source before propagation begins. In HippoRAG2, all nodes propagate equally regardless of query relevance


LearnedPPR Policy
Single MLP predicts all 10 parameters from the query embedding, unifying routing weights, gate thresholds, gate slopes, and PPR damping into one learned policy
Routing weights (α, β, γ): Allocated via softmax → multiply onto per-type gate values → dynamically emphasize the most relevant granularity per query
Gate parameters (τ_T, k_T): Replace fixed/heuristic thresholds and slopes → gates become query-adaptive
PPR damping: Replaces fixed damping and complexity-based restart → exploration depth is learned per query

Difference from baseline: HippoRAG2 uses fixed damping, no gates, no routing — all parameters require manual tuning


Differentiable PPR Training
  Differentiable PPR: PPR iteration is implemented as `torch.sparse.mm` — all operations preserve the computational graph for backpropagation
  Pairwise hinge loss: Encourages the gold passage to receive higher PPR score than hard negatives by at least margin Δ — directly optimizes retrieval ranking quality


Key Insight: M learns where to start, M2 learns where to propagate, M3 learns both



Training Data Construction

 Positive node (v⁺): The vertex index of the gold (ground-truth) passage from the dataset's annotated answers — this is the passage the model should rank highest
 Hard negative node (v⁻): Selected from the seed set's top-5 highest-scoring passages that are NOT gold — these are the most confusing false positives that the model must learn to suppress. Hard negatives are far more informative than random negatives because they are semantically close to the query but irrelevant to the answer
- Pairwise training: Each training example is a (positive, negative) pair, the hinge loss pushes the positive passage's PPR score above the negative's by at least margin Δ, directly optimizing retrieval ranking quality
Key Insight: By using hard negatives (top-5 highest-scoring non-gold passages) instead of random negatives, the model is forced to learn fine-grained gating that distinguishes semantically similar but irrelevant passages from the gold answer — directly optimizing the most confusing retrieval errors.
