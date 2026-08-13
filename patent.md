在投稿QG-PPR 之前，需要先申请一个专利。这个专利审查有些非专业人员参加评审，如何让他们能尽快get 到QG-PPR 的主要发明点 是非常重要的。

首先，介绍当前sota hipporag2 有哪些drawbacks， 用 大白话+图 的方式介绍做不好的原因，用一页ppt 或者多页 表示，下面是之前写的motivation，先判断是否正确，然后给出修改建议

Problem 1: Fixed graph diffusion
Graph-based retrieval systems such as PPR-based GraphRAG methods can propagate relevance through multiple hops of a knowledge graph. However, a fixed propagation strategy may indiscriminately transmit relevance through both useful and irrelevant nodes. This is particularly problematic in dense knowledge graphs containing:
High-degree entities;
Semantically related but answer-irrelevant passages;
Documents sharing common terminology;
Heterogeneous relations across multiple semantic granularities.
As diffusion proceeds, irrelevant relevance can cascade through the graph and eventually dominate the top-ranked passages.


Problem 2: Multi-granular graph noise
A multi-granular graph contains different types of nodes with substantially different semantic roles. The importance of each granularity depends on the query.
A query about a specific process chemistry may require strong entity-level signals, while a query concerning a broader technology trend may benefit from paper-level evidence.


Problem 3: Query-independent propagation
Different queries require different retrieval behaviors.
For example:
A highly specific process query may require narrow diffusion around a small number of relevant entities;
A document-level literature query may benefit from broader propagation through paper-level nodes;
A cross-document technical query may require entity-to-paper-to-passage diffusion.
A single fixed PPR configuration cannot optimally represent these different retrieval requirements.


然后，介绍当前技术所在项目的商用价值， 修改下面时间线的plan和expect benefits，如何能更评审人一下子抓住重点？这两部分是一页ppt
26-3Q: Case Verification (BLPM, Si:SiGe, X-Mask)
26-4Q: DS Co-Scientist Release (Targeting CTO's Next-Gen Process Development Team / DRAM Process Development Team)
DS Co-Scientist Release (Entire CTO-wide)

27-3Q: Expansion Case Verification (CTO's Process Development Dept / TD / Mask)
27-4Q: DS Co-Scientist Release (CTO-wide)

(Search & QA) Enhanced context search based on Recipe IDs and improved QA performance
(Experiment Design) Designing necessary experiments based on current experiment history (experiment gap filling)
(Tool) Supporting the documentation of engineers' tacit knowledge

下一页想介绍QG-PPR 在整个项目DS Co-Scientist 中的必要性，如何写更合适
