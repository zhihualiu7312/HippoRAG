평가 항목	자체 평가
(가중치)	
당·타사	- 당/타사 제품에 활용되고 있는가?
활용가능성	A. 현 제품 적용 중이거나 향후 3년 내 적용 예상
-15	B. 향후 5년내 적용 예상
	C. 향후 10년 내 적용 예상
사업기여도	- 사업에 미치는 영향이 어느 정도인가?
-15	A.사업부 활용 계획 명확 또는 부재시 사업에 큰 걸림돌
	B.제품의 핵심 기술로 적용 가능성 높음
	C.제품의 주변 기술로 적용 가능성 있음
	D.제품에 적용 가능성 희박
기술 독창성	-  기존 기술과 approach idea가 얼마나 차이가 있는가?
-5	A. 유사 기술이 없음(독창적 idea)
	B. 기존 기술을 의미 있게 융합한 새로운 기술
	C. 유사한 선행 기술의 개량(one of them)
	D. 극히 유사한 선행기술 존재                               
기술 대체   곤란성	- 대체 가능한 기술이 있는가?
-10	A. 향후 5년간 대체 기술 난해
	B. 현재는 대체 기술 난해
	C. 설계 변경시 대체 가능
	D. 설계 변경 없이도 대체 가능
기술의 효과	-  기존기술 대비 성능이 현저한가?
-15	 ※ '성능'은 cost down 등의 모든 개념을 포괄한 것임
	A. 선행에서 예측하지 못한(이질적인) 새로운 효과
	B. 선행에서 예측가능하나, 현저한 효과
	C. 선행보다 향상된 효과
	D. 선행과 차별되는 효과가 미미

Recommended concise version for the evaluation form

Applicability — A:
Applicable to GraphRAG, enterprise knowledge retrieval, technical literature search, and knowledge-intensive QA, with potential for product adoption within three years.

Business Contribution — B:
Can serve as a core retrieval technology for GraphRAG systems by improving evidence recall and downstream QA quality for complex and multi-hop queries.

Technical Originality — A:
Introduces a query-conditioned node-level diffusion policy that dynamically controls granularity-aware relevance injection and propagation on a fixed multi-granular graph, providing a distinct alternative to fixed PPR and query-conditioned edge-level diffusion.

Difficulty of Technical Substitution — A:
The integrated query-conditioned gating and adaptive PPR diffusion mechanism is not readily replaceable through simple parameter or model substitution and would require a different graph retrieval architecture.

Technical Effectiveness — A:
Significantly improves retrieval performance over HippoRAG2, achieving +7.32 percentage points in R@10, +7.85 points in R@50, and +5.30 points in R@200 on MuSiQue.
