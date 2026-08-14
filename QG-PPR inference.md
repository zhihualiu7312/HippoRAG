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

Algorithm: QG-PPR Inference

Input:
    query q
    fixed graph G = (V, E)
    trained policy πθ

Output:
    ranked passage list R

1.  hq ← Embed(q)

2.  // Query-conditioned policy
3.  (θR, θD, dq) ← πθ(hq)
    // θR: reset-gate parameters
    // θD: diffusion-gate parameters

4.  // Initial relevance from fact retrieval
5.  s ← BuildResetVector(FactRetrieval(q, G))

6.  // Query-conditioned reset gate
7.  gR ← ResetGate(hq, V; θR)

8.  // Gated relevance injection
9.  sR ← Normalize(s ⊙ gR)

10. // Query-conditioned diffusion gate
11. gD ← DiffusionGate(hq, V; θD)

12. // Gated PPR diffusion
13. p(0) ← s

14. for t = 0 ... T−1:
15.     p(t+1) ←
           (1 − dq) · sR
           + dq · M · (gD ⊙ p(t))

16. R ← TopK(p(T) restricted to passage nodes)

17. return R

