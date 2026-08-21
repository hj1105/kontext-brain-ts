# 유명 공개 RAG 평가셋 후보 조사

작성일: 2026-08-21  
범위: `kontext-brain-ts`의 정적 KB retrieval + answer/judge 평가에 추가할 수 있는 공개 데이터셋  
출처 원칙: 데이터셋 저자 논문, 공식 저장소, 공식 배포 페이지만 사용

## 결론

가장 실용적인 추가 순서는 다음과 같다.

1. **BEIR SciFact + NFCorpus**를 빠른 retrieval-only 회귀 게이트로 먼저 추가한다. 두 셋은 전체 코퍼스가 각각 약 5K, 3.6K 문서이고 test query가 300, 323개라 매 실험마다 전수 검색하기 쉽다. 다만 BEIR의 qrels는 answer/reference-answer가 아니므로 기존 answer/judge 지표에는 사용하지 않는다. [BEIR 공식 데이터셋 표](https://github.com/beir-cellar/beir/wiki/Datasets-available)
2. **MuSiQue-Ans dev**를 범용 다중 홉 일반화 게이트로 추가한다. 공식 배포물의 인스턴스별 20개 문단을 split 전체에서 문서 ID 기준으로 중복 제거해 하나의 파생 정적 코퍼스로 만든다. 이 방식은 공식 leaderboard 설정을 그대로 재현한 것이 아니라는 점을 결과에 명시해야 한다.
3. **2WikiMultiHopQA dev**를 Entity–Relation–Evidence 평가의 핵심 추가셋으로 사용한다. 공개 `para_with_hyperlink.zip`을 공통 KB로 쓰면 인스턴스별 10문단 설정보다 실제 정적-KB retrieval에 가깝고, sentence supporting facts와 Wikidata evidence triple을 모두 점수화할 수 있다. [공식 저장소와 코퍼스 링크](https://github.com/Alab-NII/2wikimultihop)
4. **HotpotQA fullwiki dev**와 **KILT Natural Questions dev**는 대형 코퍼스 최종 검증으로 사용한다. 각각 5.23M 문서와 5.90M Wikipedia page 규모라 반복 튜닝 루프에는 비싸지만, 유명 공개 벤치마크와의 비교 가능성이 가장 높다. [HotpotQA 공식 fullwiki 설명](https://hotpotqa.github.io/index.html), [KILT 공식 knowledge source](https://github.com/facebookresearch/KILT)
5. **FRAMES**는 최종 answer/reasoning 검증에는 좋지만, 제공된 gold Wikipedia URL만 합쳐 코퍼스를 만들면 gold로 코퍼스를 구성한 셈이라 retrieval 비교가 낙관적으로 편향된다. 전체 Wikipedia snapshot을 쓰지 않는 한 retrieval 순위표에는 넣지 않는다.
6. **CRAG**는 query별 검색 페이지와 mock API를 제공하는 동적·query-scoped 벤치마크다. 정적 KB 트랙에 넣지 말고 기존 `dynamic-api` 확장 트랙으로만 유지한다.

현재 `DatasetBundle` 계약은 `CorpusDocument[]`, `BenchmarkQuery.referenceAnswer`, `goldEvidenceIds`, `goldEvidenceText`를 이미 갖고 있어 HotpotQA, 2Wiki, MuSiQue, KILT/NQ 변환에 충분하다. BEIR만 `referenceAnswer=null`인 retrieval-only 셋으로 선언하면 된다.

## 후보 비교

| 후보 | 공개 평가 규모 | 코퍼스 | evidence/answer | 라이선스 판단 | 정적-KB 적합성 | 권장 역할 |
|---|---:|---|---|---|---|---|
| HotpotQA fullwiki | dev 7,405; 전체 112,779 QA | 공식 fullwiki intro corpus 5.23M 문서; 배포 압축 약 1.55GB | answer + sentence-level supporting facts | 데이터와 processed Wikipedia: CC BY-SA 4.0 | 매우 높음 | 대형 최종 다중 홉 검증 |
| 2WikiMultiHopQA | dev 12,576; train 167,454; test 12,576; 총 192,606 | 별도 전체 article/paragraph + hyperlink 배포물 | answer + sentence support + evidence triple | repo/mirror에 Apache-2.0 표시; Dropbox data 적용 범위와 Wikipedia 원문 의무는 별도 확인 | 매우 높음 | KG/관계/다중 홉 핵심 검증 |
| MuSiQue-Ans | train 19,938; dev 2,417; test 2,459; 총 24,814 | 공식 설정은 query당 20개 문단; 전체 7,676 supporting paragraphs | answer + supporting paragraphs + decomposition | CC BY 4.0 | 중간 | 파생 shared-corpus 일반화 게이트 |
| MuSiQue-Full | MuSiQue-Ans의 각 split을 answerable/unanswerable pair로 2배 | MuSiQue-Ans와 동일, 일부 문맥을 제거한 insufficiency pair | answerability + answer + support | CC BY 4.0 | 중간 | abstention/context sufficiency 별도 트랙 |
| KILT Natural Questions | train 87,372; dev 2,837; hidden-answer test 1,444 | 2019-08-01 Wikipedia, 34.76GiB, 5,903,530 pages | answer + KILT Wikipedia provenance | KILT code MIT; NQ repo Apache-2.0; Wikipedia 및 원 데이터 조건 별도 적용 | 매우 높음 | 대형 단일 홉/open-domain 최종 검증 |
| Original Natural Questions | train 307,372; dev 7,830; hidden test 7,842 | 각 질문에 해당 Wikipedia HTML 한 페이지가 포함됨; 공통 open-domain corpus는 없음 | long/short answer offsets | 공식 repo Apache-2.0; Wikipedia 원문 조건 별도 | 낮음(원형), 높음(KILT 변환) | KILT/NQ 사용 권장 |
| BEIR SciFact | test query 300; corpus 약 5K | BEIR `corpus.jsonl` | qrels만 있음; 생성 reference answer 없음 | BEIR code Apache-2.0, 개별 데이터셋 라이선스는 원 소유자 조건 | retrieval만 높음 | 빠른 과학 도메인 회귀 게이트 |
| BEIR NFCorpus | test query 323; corpus 약 3.6K | BEIR `corpus.jsonl` | qrels만 있음; 생성 reference answer 없음 | BEIR code Apache-2.0, 개별 데이터셋 라이선스는 원 소유자 조건 | retrieval만 높음 | 빠른 의료/영양 도메인 회귀 게이트 |
| FRAMES | test-only 824 | 각 질문의 gold Wikipedia article URL 2–15개; 본문 snapshot은 별도 생성해야 함 | gold answer + relevant article URLs + reasoning labels | 공식 dataset card Apache-2.0; Wikipedia 본문은 별도 | 조건부 | answer/reasoning 최종 검증 |
| CRAG | 총 4,409 QA | query별 최대 50 HTML 검색 결과 + mock KG/API | gold/alternate answer; 공식 relevance label은 아님 | CC BY-NC 4.0 | 낮음 | dynamic-api 확장 트랙 |

## 상세 검토

### 1. HotpotQA

공식 논문은 총 112,779개를 train-easy 18,089, train-medium 56,814, train-hard 15,661, dev 7,405, test-distractor 7,405, test-fullwiki 7,405로 나눈다. 데이터는 sentence-level supporting facts를 평가하도록 설계됐고, 로컬 점수화에는 gold가 공개된 dev를 사용한다. fullwiki는 Wikipedia 전체의 첫 문단들에서 관련 문서를 찾아야 하는 설정이다. [공식 EMNLP 논문, Table 1](https://aclanthology.org/D18-1259.pdf)

공식 사이트는 데이터셋과 processed Wikipedia를 CC BY-SA 4.0으로 배포한다. fullwiki용 introduction paragraph archive는 BZip2 약 1.55GB이고, 2017-10-01 English Wikipedia를 사용한다. [공식 다운로드](https://hotpotqa.github.io/index.html), [processed Wikipedia 명세](https://hotpotqa.github.io/wiki-readme.html)

적합성:

- query, answer, `[title, sentence_id]` supporting fact가 현재 계약에 직접 매핑된다.
- fullwiki corpus는 공통 정적 snapshot이므로 query별 gold 문서를 KB 구축에 누출하지 않는다.
- sentence를 evidence ID로 쓰고, 반환 문서가 긴 경우 sentence span과 원문 document ID를 함께 보존해야 citation F1을 공정하게 계산할 수 있다.
- 단점은 5.23M 문서 전체 KG/embedding 구축 비용이다. BEIR가 배포하는 HotpotQA 변환본도 query 7,405, corpus 5.23M, 평균 relevance 2.0으로 동일한 규모를 확인할 수 있다. [BEIR 공식 표](https://github.com/beir-cellar/beir/wiki/Datasets-available)

권장 실행:

- small: 고정 500 dev query, **전체 fullwiki corpus를 그대로 사용**. KG 재구축을 피하도록 corpus/index digest cache를 공유한다.
- full: dev 7,405 retrieval 전수 + 사전에 고정한 200개 answer/judge.
- 개발 중 gold supporting facts는 scoring에만 사용하고 retrieval/reranking/capability selection에는 절대 전달하지 않는다.

### 2. 2WikiMultiHopQA

공식 논문은 train-medium 154,878, train-hard 12,576, dev 12,576, test 12,576, 총 192,606개의 분할을 보고한다. 질문 유형은 comparison, inference, compositional, bridge-comparison이고, 모델이 answer, supporting facts, evidence triple을 함께 예측하도록 설계됐다. [공식 COLING 논문, Table 1과 task 정의](https://aclanthology.org/2020.coling-main.580.pdf)

공식 저장소는 HotpotQA 호환 `context`, `supporting_facts`, `answer` 외에 `[subject, relation, object]` evidence와 entity IDs를 제공한다. 별도 `para_with_hyperlink.zip`은 article ID, title, sentences, hyperlink mentions를 포함하므로 공통 정적 corpus와 hyperlink graph를 동시에 만들 수 있다. [공식 저장소](https://github.com/Alab-NII/2wikimultihop)

적합성:

- 현재 범용 Entity–Event–Claim–Evidence 모델의 entity identity, relation/fact, citation을 동시에 평가하기 가장 직접적이다.
- 공식 샘플의 `context`는 gold 2개(bridge-comparison은 4개)와 hard distractor를 합친 query-local 10문단이다. 정적 KB 비교에는 query-local context 대신 공식 `para_with_hyperlink` 전체 배포물을 사용해야 한다.
- test의 answer/evidence는 비공개이므로 로컬 반복에서는 dev를 결정론적으로 development/holdout으로 다시 나누고 holdout은 최종 한 번만 열어야 한다.

라이선스는 GitHub 저장소와 제1저자의 공식 Hugging Face mirror가 Apache-2.0으로 표시된다. 그러나 저장소 README는 외부 Dropbox 원 데이터에 그 라이선스가 적용된다는 문구를 별도로 적지 않는다. 따라서 이 표시는 코드/mirror 배포 근거로 기록하되, 원 Dropbox data의 범위와 포함된 Wikipedia 문단의 attribution/share-alike 의무는 재배포 전에 확인한다. [공식 repo license 표시](https://github.com/Alab-NII/2wikimultihop), [제1저자 배포 mirror](https://huggingface.co/datasets/xanhho/2WikiMultihopQA)

권장 실행:

- small: dev에서 type별 층화한 고정 500개 retrieval, 100개 answer/judge; corpus는 full `para_with_hyperlink`.
- full: dev 12,576 retrieval 전수, 고정 200개 answer/judge, evidence triple F1을 보조 지표로 추가.

### 3. MuSiQue

MuSiQue-Ans는 2–4 hop 질문 24,814개로, train 19,938, dev 2,417, test 2,459다. 논문은 21,020 unique single-hop questions, 7,676 supporting paragraphs를 보고한다. 각 인스턴스의 공식 context는 supporting paragraph와 hard distractor를 합친 20개 문단이다. MuSiQue-Full은 각 answerable 질문에 대응하는 unanswerable context를 하나씩 만들어 split별 크기가 2배다. [공식 TACL 논문, dataset statistics와 context 구성](https://aclanthology.org/2022.tacl-1.31.pdf)

공식 저장소는 train/dev/test, answer/support, decomposition, answerability를 배포하며 CC BY 4.0이다. 또한 seed single-hop dataset과의 leakage를 피할 수 있도록 dev/test에 사용된 single-hop IDs를 제공한다. [공식 저장소](https://github.com/StonyBrookNLP/musique)

적합성 및 주의점:

- 공식 task는 하나의 global corpus retrieval이 아니라 query-local 20문단 reading comprehension이다.
- `kontext-brain-ts`에 넣을 때는 train/dev/test에 들어 있는 paragraph를 안정 ID `(title, text-hash)`로 중복 제거해 shared corpus를 만드는 **파생 static-KB 설정**이 필요하다.
- 이 변환에서 동일 문단의 문장부호/공백 차이가 중복 문서로 남지 않도록 canonical text digest를 기록한다.
- MuSiQue-Full의 insufficient-context pair를 global union corpus에 동시에 넣으면 제거된 evidence가 다른 인스턴스에서 다시 나타날 수 있다. 따라서 Full은 일반 global retrieval이 아니라 query-scoped context-sufficiency 트랙에서만 평가하거나, 우선 MuSiQue-Ans만 사용한다.

권장 실행:

- small: dev hop별 층화 500개, derived shared corpus, retrieval 전수 + answer/judge 100개.
- full: dev 2,417 retrieval 전수 + 고정 200개 answer/judge.
- 공식 leaderboard 점수와 혼동하지 않도록 결과 이름을 `musique-ans-derived-static-kb-v1`로 명시한다.

### 4. KILT / Natural Questions

Original NQ는 real Google search query와 해당 Wikipedia page HTML/answer annotation을 묶은 reading comprehension 데이터다. 공식 저장소는 train 307,372, dev 7,830, hidden test 7,842를 보고하며 Apache-2.0이다. 각 질문에는 하나의 timestamped Wikipedia HTML 문서가 포함되므로 원형 그대로는 open-domain 공통 KB retrieval 평가가 아니다. [Natural Questions 공식 저장소](https://github.com/google-research-datasets/natural-questions)

KILT는 이를 고정된 2019-08-01 Wikipedia knowledge source에 맞춘 open-domain QA로 변환한다. KILT knowledge source는 34.76GiB, 5,903,530 pages이고, NQ KILT split은 train 87,372, dev 2,837, test-without-answers 1,444다. 각 공개 train/dev output은 answer와 Wikipedia ID/paragraph span provenance를 가진다. [KILT 공식 저장소와 직접 다운로드](https://github.com/facebookresearch/KILT)

적합성:

- 고정 corpus, answer, provenance가 모두 있어 현재 정적 KB 계약에 정확히 맞는다.
- NQ는 주로 single-hop/open-domain factual retrieval이라 Medical/Novel 및 multi-hop 셋과 다른 축의 일반화 검증이 된다.
- 공식 test는 answer가 공개되지 않으므로 dev 2,837을 사용한다.
- KILT repository의 MIT 표시는 코드 라이선스다. KILT가 통합한 NQ와 Wikipedia 지식 소스에는 각각의 원 데이터/콘텐츠 조건이 남으므로 dataset metadata에 `composite; verify upstream`를 기록해야 한다.

권장 실행:

- small: dev 고정 500개 query, 전체 KILT knowledge source와 공유 index.
- full: dev 2,837 retrieval 전수 + 고정 200개 answer/judge.
- KILT HotpotQA dev 5,600도 동일 corpus 위에서 받을 수 있지만 HotpotQA fullwiki와 중복되므로 첫 추가에서는 NQ만 선택한다. [KILT data catalogue](https://github.com/facebookresearch/KILT)

### 5. BEIR

BEIR는 다양한 도메인의 zero-shot **정보검색** 벤치마크다. 공식 포맷은 `corpus.jsonl`, `queries.jsonl`, `qrels/*.tsv`이고, 공식 예시는 NDCG, MAP, Recall, Precision, MRR을 계산한다. [BEIR 공식 저장소](https://github.com/beir-cellar/beir)

`kontext-brain-ts`에 적합한 첫 두 셋은 다음과 같다.

- SciFact: test query 300, corpus 약 5K, 평균 relevant doc/query 1.1.
- NFCorpus: test query 323, corpus 약 3.6K, 평균 relevant doc/query 38.2.

이들은 매우 작아서 KG 구축·retrieval 회귀를 빠르게 검출하고, scientific claim과 biomedical/health 정보에 대한 out-of-domain 신호를 준다. 다만 qrels는 생성용 reference answer나 claim-level gold answer가 아니므로 correctness/faithfulness/citation judge를 억지로 실행해서는 안 된다. [BEIR 공식 데이터셋 표와 다운로드 링크](https://github.com/beir-cellar/beir/wiki/Datasets-available)

라이선스 주의: BEIR 코드 저장소는 Apache-2.0이지만, BEIR 저자들은 자신들이 포맷만 재배포할 뿐 개별 dataset 사용 권한을 보증하지 않으며 사용자가 원 소유자의 라이선스를 확인해야 한다고 명시한다. 따라서 adapter metadata에는 BEIR가 아닌 SciFact/NFCorpus 원 라이선스와 citation을 별도 기록해야 한다. [공식 disclaimer](https://github.com/beir-cellar/beir#disclaimer)

권장 실행:

- small/full 구분 없이 두 셋 모두 corpus와 test query 전수 실행.
- 지표: Recall@5/10/20, nDCG@10, MRR@10, latency, input token, embedding cost.
- 제품 후보가 이 두 셋 중 하나라도 크게 퇴행하면 Medical 점수가 올라도 승격하지 않는다.

### 6. FRAMES

FRAMES는 test-only 824개 질문으로, 각 질문에 gold answer, reasoning labels, 2–15개의 relevant Wikipedia article URL을 제공한다. 공식 card는 Apache-2.0이고, numerical/tabular/multiple-constraint/temporal/post-processing reasoning을 포함한다. [Google 공식 dataset card](https://huggingface.co/datasets/google/frames-benchmark)

공식 논문은 824개 전부가 평가셋이며 약 36%가 2개 article, 약 35%가 3개 article을 필요로 한다고 설명한다. 관련 Wikipedia **본문 snapshot 자체는 dataset에 포함되지 않으므로**, 재현 가능한 평가에는 dataset revision과 별도로 page revision/content digest를 고정해야 한다. [공식 논문](https://arxiv.org/pdf/2409.12941)

주의점:

- gold URL들의 union만 corpus로 만들면 평가 질문을 보고 corpus 범위를 정한 셈이다. answer/reasoning 평가는 가능하지만 open-domain retrieval precision 비교에는 약한 negative corpus 때문에 부적합하다.
- 공정한 retrieval 비교에는 고정된 전체 Wikipedia snapshot을 사용하거나, 최소한 질문과 무관하게 미리 정한 배경 Wikipedia corpus를 합쳐야 한다.
- 현재 저장소에 이미 공식 TSV와 Wikipedia 페이지 수집 adapter가 있으므로, runtime이 허용될 때 final-only 검증으로 재활성화하는 것이 낫다.

권장 실행:

- small: reasoning type별 고정 100개 answer-only 검증. retrieval leaderboard에는 미포함.
- full: 824개 전수. 전체 Wikipedia snapshot을 쓴 경우에만 evidence recall을 공식 비교표에 포함.

### 7. CRAG

CRAG는 총 4,409개 QA, 5개 domain, 8개 question category와 시간에 따라 변하는 질문을 포함한다. 공식 배포는 각 query에 대해 최대 50개 full HTML search result와 mock KG/API를 제공하며, search result relevance는 보장되지 않는다. [공식 논문](https://arxiv.org/abs/2406.04744), [공식 dataset schema](https://github.com/facebookresearch/CRAG/blob/main/docs/dataset.md)

따라서 다음 이유로 정적 KB 본평가에 적합하지 않다.

- retrieval corpus가 query별로 생성되어 있어 모든 페이지를 global corpus로 합치면 query-to-page 생성 관계가 숨은 누출 신호가 될 수 있다.
- static, slow-changing, fast-changing, real-time 질문을 한데 포함하므로 고정 KB coverage와 최신성 실패가 혼재한다.
- 라이선스가 CC BY-NC 4.0이라 상용 배포/재배포 조건도 별도 관리해야 한다. [공식 저장소](https://github.com/facebookresearch/CRAG)

CRAG는 기존 manifest처럼 `dynamic-api` track에 두고, 공식 correct/missing/incorrect(-1) 평가와 abstention을 별도 보고하는 것이 맞다.

## 권장 small/full 실행 프로토콜

### 데이터 누출 방지

- dataset 이름, category, reference answer, gold evidence는 retrieval, query rewrite, graph traversal, reranking, capability selection에 전달하지 않는다.
- 동일한 source-only capability selector와 동일한 global default를 모든 데이터셋에 적용한다.
- 공개 dev에서 튜닝해야 하는 데이터셋은 query ID hash로 `development 20% / frozen holdout 80%`를 먼저 고정한다. development 결과만 반복 확인하고, holdout은 후보 구성이 확정된 후 1회 실행한다.
- answer/judge sample IDs, hash seed, corpus digest, adapter commit, upstream revision을 manifest에 고정한다.
- MuSiQue derived corpus와 FRAMES page snapshot처럼 공식 설정에서 변환된 데이터는 별도 dataset ID와 변환 명세를 갖는다.

### 비용 단계

1. **매 변경 fast gate**
   - 기존 Medical/Novel 고정 smoke.
   - BEIR SciFact 300 + NFCorpus 323 retrieval 전수.
   - 실패/퇴행 후보는 KG 전체 재구축 전에 중단.
2. **small generalization gate**
   - MuSiQue-Ans dev 고정 500.
   - 2Wiki dev 고정 500.
   - HotpotQA fullwiki 고정 500과 KILT/NQ 고정 500은 shared corpus/index가 준비된 뒤 실행.
   - retrieval이 통과한 후보만 dataset당 100 answer/judge.
3. **full candidate evaluation**
   - Medical 2,062 + Novel 2,010 retrieval 전수.
   - MuSiQue dev 2,417, 2Wiki dev 12,576, HotpotQA dev 7,405, KILT/NQ dev 2,837 retrieval 전수.
   - Medical과 각 외부 QA셋 고정 200 answer/judge.
   - FRAMES 824는 final-only answer/reasoning 검증.
4. **승격 기준**
   - 특정 한 데이터셋의 최고점이 아니라, Medical primary 지표가 v9보다 개선되고 Novel 및 외부 holdout의 recall/context precision이 허용 오차 이상 퇴행하지 않아야 한다.
   - 추천 기본 허용 오차: 큰 셋의 Recall@10/Context Precision 절대 `-0.005` 이하 퇴행 금지, BEIR nDCG@10 절대 `-0.01` 이하 퇴행 금지.
   - answer 품질은 correctness, strict faithfulness, claim F1, citation F1 중 하나를 올리면서 나머지를 절대 `-0.01` 넘게 떨어뜨리지 않는 Pareto 조건을 사용한다.
   - LightRAG와의 비교에서는 bundled-context packaging으로 raw context precision이 직접 비교되지 않는 셀을 별도 표시한다.

## 구현 우선순위

1. `beir-scifact`, `beir-nfcorpus` retrieval-only canonical adapters.
2. `musique-ans-derived-static-kb-v1` adapter와 변환 manifest.
3. `2wikimultihopqa-static-v1` adapter; `para_with_hyperlink` parser 및 sentence/evidence mapping.
4. 공용 Wikipedia 대형 index abstraction을 만든 뒤 HotpotQA fullwiki와 KILT/NQ를 같은 방식으로 연결.
5. FRAMES/CRAG는 primary static-KB 최적화 루프 밖에서 별도 track으로 유지.

이 순서라면 작은 독립 도메인에서 회귀를 빠르게 잡고, 다중 홉 derived/static corpus에서 범용성을 확인한 다음, 가장 비싼 5M-page open-domain 검증에 들어갈 수 있다.
