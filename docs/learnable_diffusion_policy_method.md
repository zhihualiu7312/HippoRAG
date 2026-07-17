# Learnable Diffusion Policy for Query-Adaptive Graph Retrieval

## Method Definition

We introduce a learnable diffusion policy that replaces the fixed transition rule of standard Personalized PageRank with a query-conditioned transition model. Given a query $q$ and a set of seed nodes $S$, the model produces a diffusion policy $\pi_q = (r_q, d_q, W_q, G_q)$, where:

- $r_q$ is the restart probability,
- $d_q$ is the diffusion depth,
- $W_q$ is a query-conditioned edge-weight scaling map,
- $G_q$ is a node gate map that controls which nodes are allowed to propagate mass.

The resulting transition matrix is defined as:

$$
P_q = D_q^{-1}(A \odot W_q)
$$

where $A$ is the adjacency matrix of the graph, $W_q$ is the query-conditioned edge reweighting matrix, and $D_q$ is the corresponding degree normalization matrix.

The diffusion process remains standard Personalized PageRank, but the transition matrix is now adaptive to the query.

## Training Objective

We train the policy with pairwise ranking loss. For each training example, we construct a local graph around the query seeds and run diffusion to obtain scores for a positive node and a negative node.

The objective is:

$$
\mathcal{L} = \max(0, m - s_{pos} + s_{neg})
$$

where $s_{pos}$ is the final diffusion score of the positive node and $s_{neg}$ is the score of the negative node. This encourages the model to assign higher propagated scores to nodes that are more relevant for the current query.

## Pseudocode

```text
Input: query q, seed scores S, graph G

1. Build a query-induced local graph around S
2. Create a diffusion policy from the learned model:
   - restart probability r_q
   - diffusion depth d_q
   - edge scaling weights W_q
   - node gates G_q
3. Run Personalized PageRank with the query-conditioned transition matrix
4. Obtain final node scores
5. Optimize pairwise ranking loss:
      score(positive) > score(negative)
```

## Comparison with HippoRAG2

| Aspect | HippoRAG2 | Learnable Diffusion Policy |
| --- | --- | --- |
| Diffusion policy | Fixed transition rule | Query-conditioned transition model |
| Adaptation | Static across queries | Learned per query |
| Training objective | No learned diffusion policy | Pairwise ranking over propagated scores |
| Main contribution | Graph-based retrieval | Learnable query-aware diffusion |
| Complexity | Low and simple | Slightly higher, but still efficient |
