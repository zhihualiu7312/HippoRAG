Granularity-aware reset gating dynamically allocates query relevance across different node granularities, such as Entity, Passage, and Paper, before PPR diffusion. It allows the model to emphasize the granularity that is most informative for the current query when initializing the diffusion process.

用两句英文介绍granularity-aware diffusion gating  

Granularity-aware diffusion gating dynamically controls how much relevance each Entity, Passage, or Paper node is allowed to propagate during PPR diffusion. It suppresses noisy propagation from weakly relevant or overly connected nodes while preserving query-aligned multi-hop evidence.
