Algorithm: QG-PPR Inference
─────────────────────────────────────────────
Input:  query q, knowledge graph G, trained MLP π_θ
Output: ranked passage list R

1.  q_emb ← Embed(q)

2.  // MLP predicts all parameters
3.  (w_T, τ_T, k_T, d) ← π_θ(q_emb)

4.  // Fact retrieval (same as HippoRAG)
5.  r ← BuildResetVector(FactRetrieval(q, G))

6.  // Compute gated reset vector
7.  for each node v ∈ V:
8.      g_v ← σ(k_T · (cos(q_emb, e_v) - τ_T)) · w_T
9.  r_gated ← r ⊙ g

10. // Run PPR with gated reset and learned damping
11. p ← PPR(G, reset=r_gated, damping=dq)

12. R ← TopK(passage scores from p)
13. return R

