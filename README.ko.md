# kontext-brain

[English](./README.md) | **한국어**

> AI 에이전트를 위한 Evidence 기반 N-layer 지식 그래프 RAG - TypeScript / Node.js

[![node](https://img.shields.io/badge/node-%3E%3D20-brightgreen)](https://nodejs.org)
[![pnpm](https://img.shields.io/badge/pnpm-9-orange)](https://pnpm.io)
[![typescript](https://img.shields.io/badge/typescript-5.x-blue)](https://www.typescriptlang.org/)

kontext-brain은 여러 소스의 지식을 평면적인 벡터 인덱스로만 취급하지 않고 Resource,
source-native Chunk, Entity, Fact, ACL-aware Evidence로 구조화하는 검색 프레임워크입니다.
함께 제공되는 RAG 평가 하네스의 기본값은 **v13 anchored-evidence stack**입니다. 원본
질문을 축으로 한 multi-query 검색, 그래프/vector/BM25 fusion, coverage-aware reranking,
source hydration, 근거 필요 조건을 따르는 답변 생성을 포함합니다. 자세한 내용은
[RAG evaluation v2](./bench/src/rag-eval-v2/README.md)와
[development report](./bench/data/rag-eval-v2/cross-framework-all-datasets-2026-08-23.md)를 참고하세요.

평가 프로파일은 보고된 데이터셋에서 반복적으로 조정됐고, 일부 비교는 미리 계산한
Kontext KG를 사용하며, raw run directory는 커밋하지 않습니다. 따라서 이 결과는
독립적으로 재현된 최종 리더보드가 아니라 **회귀 검증 근거**로 봐야 합니다.

Production 경로는 외부 시스템을 source of truth로 유지하면서 Evidence 기반 파생 인덱스를
관리합니다. 질문은 접근 가능한 Resource, Chunk, Entity, Fact와 선택적 Ontology anchor에서
시작할 수 있으며, bounded best-first **Lift → Expand → Ground** 탐색을 수행합니다. 답변에는
Evidence로 선택된 source Chunk만 hydration합니다. Ontology-first staged routing은 하위 호환을
위해 남아 있지만 production N-layer retrieval의 정의는 아닙니다.

---

## 이 프로젝트는 무엇인가

8개의 배포 가능한 패키지와 벤치마크 하네스로 구성된 모듈형 monorepo입니다.

| 패키지 | 역할 |
|---|---|
| `@kontext-brain/core` | 데이터 모델, 검색 파이프라인, mapping 전략, extractive QA. LLM 의존성 없는 순수 TypeScript |
| `@kontext-brain/llm` | Claude, OpenAI, Ollama를 위한 LangChain.js adapter |
| `@kontext-brain/mcp` | 공식 `@modelcontextprotocol/sdk`를 사용한 stdio/SSE connector와 Notion·Jira·GitHub PR·Slack layer adapter |
| `@kontext-brain/loader` | YAML/zod config loader와 high-level entry point인 `KontextAgent`, `autoSetup()` |
| `@kontext-brain/tool-server` | Claude Desktop, Claude Code, Cursor 등에 6개의 kontext MCP tool을 제공하는 서버 |
| `@kontext-brain/postgres` | PostgreSQL/pgvector KG, RLS-aware retrieval, ontology deployment, proposal queue, extraction job |
| `@kontext-brain/object-storage` | S3 호환 Resource 본문 압축 저장소 |
| `@kontext-brain/github` | 온톨로지 제안을 누적해 draft PR로 발행하는 publisher |

프로젝트 런타임에 Python은 사용하지 않으며, 전체가 TypeScript / Node.js로 구현되어 있습니다.

### Production 아키텍처

```text
동기화 / 구조화                              질문 / 답변

Notion · Slack · GitHub · MCP              question + Principal
             │                                      │
             ▼                                      ▼
      정규화 + 동기화                     ACL/RLS가 적용된 seed fusion
             │                           Resource | Chunk | Entity | Fact
             ▼                                  | 선택적 Ontology
Resource ──포함──► Chunk                          │
                    │                             ▼
                    ├─언급──► Entity        bounded best-first search
                    │          │            Lift ↔ Expand ↔ Ground
                    │          ▼                       │
                    │         Fact                     ▼
                    │          ▲             접근 가능한 Evidence 순위화
                    └─근거──► Evidence        + Chunk 본문 지연 로딩
                           선택적 factKey               │
                                                       ▼
                                              final reasoning LLM
                                                       │
                                                       ▼
                                              citation 검증
                                              실패하면 답변 거부

PostgreSQL: 구조화 KG, ACL/RLS, scoring profile
Object storage: 현재 Resource와 Chunk 본문
```

동기화와 질문 처리는 별도 흐름입니다. 질문마다 `autoSetup()`이나 동기화를 실행하지 않습니다.
탐색 가능한 node 종류는 Ontology, Resource, Chunk, Entity, Fact입니다. Evidence는 Resource와
Chunk에 연결되고 선택적으로 Fact를 지지하는 최종 순위화 grounding record이며, 탐색 node가
아닙니다. Seed, edge, Evidence를 반환하기 전에 ACL을 반드시 적용합니다. 접근 가능한 Chunk에
grounding된 뒤에만 source 본문을 불러오며, final reasoning LLM은 graph layer가 아니라 검색 결과를
소비하는 downstream 단계입니다. Evidence가 없거나 답변이 Evidence ID를 citation하지 않으면
fail-closed로 답변을 거부합니다.

로컬/file-store용 legacy 경로는 다음과 같은 선형 호환 pipeline으로 남아 있습니다.

```text
Ontology routing → node별 meta search → 본문 fetch/compression → final LLM
```

Mapping, vector, metadata, content, chunking, LLM component는 계속 교체 가능한 port이지만, 이 고정
L1-L4 순서를 production Evidence KG와 혼동하면 안 됩니다. Source-grounded LLM Wiki는 미리 선언한
Ontology routing을 대체할 미래 방향으로 평가 중이지만 아직 active production route가 아닙니다.
생성된 Wiki 본문 자체는 답변 Evidence가 될 수 없습니다.

---

## 왜 사용하는가

- **처음부터 다중 소스**: Notion, GitHub, Slack을 서로 단절된 벡터 인덱스가 아니라 하나의 Evidence 기반 KG로 동기화합니다.
- **추적 가능한 grounding**: 선택된 context마다 Evidence, Resource, Chunk ID와 해당 항목에 도달한 탐색 경로·점수를 확인할 수 있습니다.
- **비용 조절**: final LLM이 없는 extractive retrieval과 풍부한 LLM 답변을 질문별로 선택할 수 있습니다.
- **MCP-native**: 공식 MCP SDK를 사용해 source server를 소비하고, 동시에 AI agent host에 kontext tool을 제공합니다.
- **통제된 선택적 Ontology 경로**: 첫 setup은 작은 Ontology를 만들 수 있지만, 이후 unmatched document는 active deployment를 즉시 변경하지 않고 deduplicated proposal queue와 draft PR로 전달합니다.

### 적용 위치

| 사용 사례 | 권장 통합 방식 |
|---|---|
| Notion, Slack, GitHub, Jira 등을 포함한 사내 지식 | MCP source를 Evidence KG에 동기화하고 ACL/RLS가 필요하면 PostgreSQL runtime 사용 |
| 기존 AI client와 coding agent | `@kontext-brain/tool-server`를 실행하여 6개 MCP tool 제공 |
| TypeScript application/service | `KontextLoader`로 YAML config를 불러와 `retrieve()` 또는 `answer()` 호출 |
| 로컬 또는 소규모 팀 지식 베이스 | File store와 Ollama provider를 사용하여 PostgreSQL·hosted completion API 없이 실행 |
| RAG 연구와 회귀 테스트 | `bench/src/rag-eval-v2`에서 dataset, model, sample, metric, manifest, resumable checkpoint 버전 관리 |

Production `KontextAgent`는 회사별 store, ACL, model, connector 차이를 수용하기 위해 config-driven으로
남습니다. **v13 기본값은 비교 평가용 RAG adapter에만 적용**됩니다. Benchmark policy가
production security/storage config를 암묵적으로 덮어쓰지 않도록 분리한 것입니다.

### v13의 동작

1. 사용자의 원본 질문을 보존하고 질문만을 변형한 검색 관점을 최대 3개 생성합니다.
2. Weighted reciprocal-rank fusion으로 vector/BM25 ranking을 합칩합니다. 원본 질문의 가중치는 2, 확장 질문은 각각 1입니다.
3. Graph/context candidate를 추가한 뒤 공통 50-candidate pool에 coverage-aware LLM reranker를 적용합니다.
4. 선택된 source를 50,000자 context budget 안에서 5,000자 window로 hydration합니다.
5. 질문으로부터 도출된 evidence need만 답하며, need당 하나의 atomic claim과 가장 적합한 citation 하나만 사용합니다. 최대 8개 claim입니다.

Dataset name, reference answer, gold evidence, judge output은 runtime 의사결정에 사용되지 않습니다.
그럼에도 v13/v15 policy는 보고된 데이터셋의 발전 결과를 보고 선택되었으며 untouched
holdout result가 아닙니다. v15는 precomputed KG에서 빠진 원본 resource를 복구하는
corpus-completeness repair를 추가한 실험 후보입니다.

---

## 빠른 시작

### 사전 요구사항

```bash
node --version    # >= 20
corepack enable   # pnpm 활성화
```

로컬 LLM인 Ollama 또는 Claude/OpenAI API key 중 하나가 필요합니다.

```bash
# 로컬 LLM
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text

# 또는 Claude
export ANTHROPIC_API_KEY=sk-ant-...
```

### 설치와 빌드

```bash
git clone <repo>
cd kontext-brain-ts
pnpm install
pnpm -r build
pnpm test
```

### 예제 실행

```bash
pnpm --filter @kontext-brain/example-basic start
pnpm --filter @kontext-brain/example-auto-setup start
bench/node_modules/.bin/tsx bench/src/rag-eval-v2/cli.ts doctor
pnpm --filter @kontext-brain/bench start
```

---

## 라이브러리 사용법

### Pattern A - YAML에서 전체 agent 생성

```typescript
import { KontextLoader } from "@kontext-brain/loader";

const agent = await KontextLoader.fromFile("kontext.yaml");
await agent.autoSetup();
const retrieval = await agent.retrieve("REST API 버전은 어떻게 관리해야 하나요?");
console.log(retrieval.context);

const result = await agent.answer("REST API 버전은 어떻게 관리해야 하나요?");
console.log(result.answer);
console.log(result.selectedMetaDocs);
console.log(result.contextTokensUsed);
```

### Production Evidence KG

Production path는 PostgreSQL/pgvector에 구조화 상태를, S3 호환 저장소에 Resource별 압축된
최신 본문 하나를 유지합니다. 외부 MCP system은 source of truth로 남습니다. 소스가
변경되면 기존 파생 Evidence를 stale 처리하고 교체본을 atomic activation합니다. 안정적인 Fact는
버전을 계속 쌓는 대신 lifecycle event를 기록합니다.

```typescript
import { Pool } from "pg";
import { S3Client } from "@aws-sdk/client-s3";
import { S3ResourceContentStore } from "@kontext-brain/object-storage";
import { createPostgresKnowledgeRuntime, migratePostgres } from "@kontext-brain/postgres";
import { GenericMCPResourceSnapshotAdapter } from "@kontext-brain/mcp";
import { KontextLoader } from "@kontext-brain/loader";

const pool = new Pool({ connectionString: process.env.DATABASE_URL });
await migratePostgres(pool);

const contentStore = new S3ResourceContentStore(new S3Client({}), {
  bucket: process.env.KONTEXT_CONTENT_BUCKET!,
});
const runtime = createPostgresKnowledgeRuntime(pool, contentStore, [
  new GenericMCPResourceSnapshotAdapter("notion", "notion", {
    groupIds: ["knowledge-users"],
  }),
]);

const agent = await new KontextLoader({
  knowledgeRuntime: {
    organizationId: "acme",
    knowledgeRetriever: runtime.knowledgeRetriever,
    mcpKnowledgeSynchronizer: runtime.mcpKnowledgeSynchronizer,
    ontologyProposalQueue: runtime.ontologyProposalQueue,
    ontologyActivation: runtime.ontologyActivation,
  },
}).fromFile("kontext.yaml");

const principal = {
  organizationId: "acme",
  subjectId: "user-123",
  groupIds: ["knowledge-users"],
};
const evidence = await agent.retrieve("주문 42는 결제됐나요?", principal);
const answer = await agent.answer("주문 42는 결제됐나요?", principal);
```

Migration은 organization RLS, Resource/Chunk-Ontology many-to-many link, Fact, Evidence, Fact event,
pgvector column, idempotent extraction job, ontology deployment/proposal, structured audit row를 생성합니다.
접근 가능한 active Evidence가 없거나 답변이 Evidence ID를 citation하지 않으면 `answer()`는
fail-closed로 동작합니다.

N-Layer traversal scoring은 고정 상수가 아니라 **관측값 기반**입니다. Search adapter는
lexical/vector rank, neighbor-list fanout/rank, normalized query evidence, relation provenance,
ACL 필터를 통과한 evidence count, conflict, freshness를 보고합니다. Versioned base profile과
선택적 query-bound route policy가 dataset/organization ID로 분기하지 않고 이 관측값을 priority로
변환합니다. 각 result trace는 profile/schema digest, missing signal, seed provider, route decision,
path length, evidence score breakdown을 기록합니다.

```typescript
const staged = await runtime.scoringProfiles.stage("acme", candidateProfile, evaluationSummary);
await runtime.scoringProfiles.setShadow("acme", staged.profileDigest);
await runtime.scoringProfiles.setCanaryPercent("acme", 5);
await runtime.scoringProfiles.activate("acme", staged.profileDigest);
// await runtime.scoringProfiles.rollback("acme", previousProfileDigest);
```

관련 설계와 검증은 [ADR 0005](./docs/adr/0005-versioned-traversal-scoring.md),
[ADR 0006](./docs/adr/0006-query-adaptive-route-scoring.md),
[adaptive evaluation](./bench/data/rag-eval-v2/adaptive-route-v3-reevaluation-2026-08-24.md),
[raw direct-only ablation](./bench/data/rag-eval-v2/adaptive-route-v3-direct-only-ablation-2026-08-25.md)을
참고하세요. 이후 [source-hydrated direct-only ablation](./bench/data/rag-eval-v2/source-hydrated-direct-only-ablation-2026-08-25.md)은
그래프가 aggregate recall을 작게 높이는 대신 precision을 낮추며, 두 데이터셋 holdout에서 엄격한
승리를 보이지 못했음을 확인했습니다. 따라서 source-hydrated direct retrieval이 현재 quality
candidate이고, adaptive graph traversal은 recall-first 실험 옵션입니다.
[Rollout runbook](./docs/runbooks/scoring-profile-rollout.md)에 나머지 gate가 정리돼 있습니다.

`kontext.yaml` Ollama 예제:

```yaml
llm:
  traversal: { provider: ollama, model: qwen2.5:1.5b, baseUrl: http://localhost:11434 }
  reasoning: { provider: ollama, model: qwen2.5:1.5b, baseUrl: http://localhost:11434 }

mcp:
  - { name: notion-docs, url: http://localhost:8101, type: notion, transport: sse }
  - { name: github-issues, command: "npx", args: ["@modelcontextprotocol/server-github"], transport: stdio }

ontology:
  - { id: backend, description: REST API server database JWT, weight: 0.9 }
  - { id: frontend, description: React UI components, weight: 0.9 }

storage:
  type: file
  path: ./.kontext-store

graph:
  maxDepth: 2
  maxTokens: 4000
  strategy: WEIGHTED_DFS
```

### Pattern B - 프로그램 코드로 직접 구성

각 component를 세밀하게 제어해야 하면 `OntologyGraph`, `KeywordMappingStrategy`,
`ScoreBasedSelector`, `IngestPipeline`, `KontextAgent`를 직접 조합할 수 있습니다. 완전한
TypeScript 예제는 [영문 README](./README.md#pattern-b--programmatic-no-yaml)에서 확인할 수 있습니다.

### Pattern C - 질문 시점 LLM이 없는 extractive retrieval

```typescript
import { ExtractiveRetriever } from "@kontext-brain/core";

const extractor = new ExtractiveRetriever(fetcherRegistry, 2);
const candidates = await metaIndex.search(nodeId, query, 3);
const result = await extractor.answer(query, candidates);
```

### Pattern D - MCP로 AI agent에 제공

```bash
pnpm --filter @kontext-brain/tool-server start kontext.yaml
./packages/tool-server/dist/cli.js kontext.yaml
```

서버가 제공하는 6개 tool:

| Tool | Input | Output |
|---|---|---|
| `kontext_query` | `{ question }` | 근거 있는 답변과 source |
| `kontext_query_context` | `{ question }` | LLM reasoning 없이 retrieval context만 반환 |
| `kontext_ingest` | `{ data, source? }` | entity를 추출해 graph에 적재 |
| `kontext_describe` | `{}` | ontology, pipeline, MCP adapter 설명 |
| `kontext_sync` | `{ connectorName? }` | 추가/변경분을 점진적으로 분류하고 삭제 resource 제거 |
| `kontext_auto_setup` | `{ targetNodeCount? }` | LLM이 ontology를 생성/확장하고 문서를 분류 |

---

## 교체 가능한 interface

각 retrieval stage는 port이며 기본 구현체는 core에 포함됩니다.

| Port | 기본 구현 | 교체 예시 |
|---|---|---|
| `LLMAdapter` | `LangChainLLMAdapter` | `Promise<string>`을 반환하는 함수 |
| `VectorStore` | `InMemoryVectorStore`, `LangChainVectorStore` | Pinecone, Weaviate, pgvector |
| `MetaIndexStore` | `InMemoryMetaIndexStore`, `VectorMetaIndexStore` | DB-backed implementation |
| `ContentFetcher` | `MCPContentFetcherBridge` | HTTP, S3, filesystem, custom API |
| `NodeMappingStrategy` | Keyword, Vector, LLM, Hybrid | Corpus별 조정 |
| `MetaDocumentSelector` | ScoreBased, `LLMMetaDocumentSelector` | Reranker model |
| `StepExecutor` | Ontology, Meta, Vector, Content, Section, Chunk | 새 pipeline step |
| `Tokenizer` | Whitespace, CharNGram, Composite, MultiLanguage | 언어별 tokenizer |
| `ChunkingStrategy` | RegexHeader, Paragraph, Recursive | Domain-specific splitter |
| `TokenEstimator` | Default, Korean | tiktoken, claude-tokenizer |
| `OntologyStore` | InMemory, File | DB persistence |
| `MCPConnector` | Stdio, SSE | Custom transport |

### 상태와 영속성

`KontextAgent`는 orchestration boundary입니다. 가벼운 file store는 로컬 개발용 legacy ontology
schema/meta-index/MCP-sync snapshot만 저장하며, production Resource·Chunk·Entity·Fact·Evidence를
담지 않습니다. Production에서는 `@kontext-brain/postgres`가 canonical structured store이고,
`@kontext-brain/object-storage`가 normalized current body를 보관합니다.

Loader는 YAML에 정의된 ontology의 SHA-256 hash를 active snapshot과 비교합니다. 변경된 candidate는
atomic activation 전에 검증되고, relation이 잘못되었거나 parent cycle이 있으면 기존 graph를 유지합니다.

Production search adapter는 instance KG 전체를 memory에 hydration하지 않습니다. Bounded frontier에
필요한, ACL로 허용된 인접 row만 불러옵니다. Resource body는 SQL ACL check 이후에만 가져옵니다.

---

## 성능: 현재 retrieval candidate

2026-08-25 평가는 GraphRAG-Bench Medical 2,062개와 Novel 2,010개의 전체 retrieval 질의를
포함합니다. 현재 quality candidate는 **source-hydrated direct hybrid retrieval**입니다.
Vector/lexical candidate를 fusion/reranking한 뒤 36,000자 budget 안에 5,000자 연속 source window로
hydration합니다. Matched graph ablation이 default promotion gate를 통과하지 못해 graph traversal은
`maxHops: 0`으로 비활성화됩니다.

| Dataset | Query | Evidence Recall@10 | Raw direct 대비 향상 | p95 retrieval |
|---|---:|---:|---:|---:|
| Medical | 2,062 | **0.80892** | **+0.09360 (+9.36%p)** | **4.18 ms** |
| Novel | 2,010 | **0.43980** | **+0.06915 (+6.92%p)** | **12.40 ms** |

Evidence recall은 필요한 gold evidence 중 top-10 context에 포함된 비율입니다. 이 결과는 answer
accuracy나 citation score가 아닙니다. Frozen OpenAI `text-embedding-3-small` checkpoint, vector seed
10, lexical seed 5, 동일 candidate/reranking 설정을 사용했습니다.

`Context precision`은 이름 때문에 answer precision과 혼동하기 쉽지만 보조 지표입니다. 각 query에서
gold-evidence text의 50% 이상을 단독으로 cover하는 source window의 비율입니다. Window packaging에
민감하며, 35.7% 또는 65.6%의 answer만 정답이라는 뜻이 아닙니다.

| Dataset | Raw direct | 현재 candidate | 절대 향상 |
|---|---:|---:|---:|
| Medical | 0.37410 | **0.65641** | **+0.28230** |
| Novel | 0.18483 | **0.35696** | **+0.17214** |

### Cross-framework 비교

공유 protocol로 완료된 가장 최신 비교는 **Kontext v15**를 사용합니다. 위의 최신
source-hydrated direct candidate는 아직 모든 외부 system과 다시 비교하지 않았습니다. Retrieval은
Medical 2,062개와 Novel 2,010개 전체를, answer/judge metric은 dataset당 동일한 deterministic
200-query sample을 사용합니다.

| Dataset | System | Recall@10 | Answer correctness | Strict faithfulness | Citation F1 |
|---|---|---:|---:|---:|---:|
| Medical | **Kontext v15** | 89.1% | **95.0%** | **96.1%** | **95.8%** |
| Medical | LightRAG 1.5.6 | **93.3%** | 89.4% | 94.2% | 94.8% |
| Medical | Microsoft GraphRAG 3.1.1 | 83.0% | 78.2% | 87.4% | 85.2% |
| Medical | Vector + BM25-RRF | 70.7% | 87.4% | 89.5% | 90.0% |
| Novel | **Kontext v15** | 82.1% | **85.7%** | **92.9%** | 93.7% |
| Novel | LightRAG 1.5.6 | **85.7%** | 85.0% | 92.7% | **94.1%** |
| Novel | Microsoft GraphRAG 3.1.1 | 77.2% | 76.7% | 86.5% | 87.6% |

LightRAG이 두 dataset의 retrieval recall에서 앞서지만, 이 run에서 Kontext v15는 가장 높은
answer correctness와 strict faithfulness를 보였습니다. Novel citation F1은 LightRAG이 조금 높습니다.

이는 독립 leaderboard가 아니라 임시 개발 비교입니다. Kontext는 precomputed KG를, 외부 system은
native index를 사용하므로 index build cost가 동일하지 않습니다. Native context packaging 단위도 달라
context precision은 비교하지 않았습니다. 자세한 제약은
[cross-framework report](./bench/data/rag-eval-v2/cross-framework-all-datasets-2026-08-23.md)를 참고하세요.

### Matched graph-traversal ablation

최신 graph treatment는 `maxHops`만 0에서 8로 바꿉니다.

| Dataset | Direct recall | Graph recall | Recall delta (95% CI) | Context-precision delta (95% CI) | p95 direct → graph |
|---|---:|---:|---:|---:|---:|
| Medical | 0.80892 | **0.81474** | +0.00582 [0.00048, 0.01115] | -0.00602 [-0.00934, -0.00269] | 4.18 → 24.41 ms |
| Novel | 0.43980 | **0.44478** | +0.00498 [0.00100, 0.00896] | -0.00277 [-0.00508, -0.00048] | 12.40 → 43.09 ms |

Graph traversal은 aggregate recall을 약 0.5%p 높이지만 context precision을 작게 낮추고 latency를 크게
늘립니다. Recall gain이 두 regression holdout 모두에서 엄격한 승리로 재현되지 않았으므로,
graph traversal은 기본값이 아니라 명시적 recall-first 옵션입니다.

---

## Auto-setup flow

온톨로지 없이 MCP source만 연결해 시작하려면:

```typescript
const agent = await KontextLoader.fromFile("kontext.yaml");
const result = await agent.autoSetup({ targetNodeCount: 8 });

console.log(`Built ${result.nodesCreated} ontology nodes`);
console.log(`Classified ${result.documentsClassified} documents`);
console.log(`${result.documentsUnmapped} unmapped`);
console.log(result.ontologyYaml);
```

내부 절차:

1. 모든 connector에서 `MCPConnector.listResources()` 호출
2. `OntologyAutoBuilder.build()`가 category, parent/level hierarchy, edge를 생성
3. `DocumentClassifier.classify()`가 각 document를 최적 node에 mapping하고 unmappable document에 대해 새 node 제안
4. Node별 `MetaIndexStore.index()`
5. Node description과 선택적 document body를 `VectorStore.upsert()`

---

## 프로젝트 구조

```text
kontext-brain-ts/
├── package.json
├── pnpm-workspace.yaml
├── tsconfig.base.json
├── biome.json
├── packages/
│   ├── core/
│   ├── llm/
│   ├── mcp/
│   ├── loader/
│   ├── postgres/
│   ├── object-storage/
│   ├── github/
│   └── tool-server/
├── examples/
├── tests/integration/
└── bench/
    ├── src/rag-eval-v2/
    └── data/rag-eval-v2/
```

---

## 기술 스택

- **Language**: TypeScript 5.x, strict mode, `noUncheckedIndexedAccess`
- **Runtime**: Node.js 20+, ESM, native fetch
- **Package manager**: pnpm 9 workspace
- **Build**: tsup
- **Test**: vitest
- **Lint/format**: Biome
- **Validation**: zod
- **YAML**: `yaml`
- **HTTP / MCP**: `@modelcontextprotocol/sdk` stdio/SSE
- **LLM**: `@langchain/anthropic`, `@langchain/openai`, `@langchain/ollama`
- **Embedding**: LangChain.js `Embeddings`; 기본 Ollama `nomic-embed-text`, OpenAI `text-embedding-3-small`

Python, Java, Rust build dependency가 없으며 Node 20과 pnpm만으로 실행됩니다.

---

## 현재 상태

- ✅ Core, llm, mcp, loader, tool-server package의 typecheck/build 통과
- ✅ Retrieval, persistence, incremental MCP sync, graph traversal, entity, tool server의 unit/integration test
- ✅ Medical/Novel 전체 4,072개 질의를 대상으로 한 versioned retrieval evaluation
- ✅ 최신 source-hydrated direct candidate의 matched graph on/off ablation
- ⚠️ 실제 Notion/GitHub/Slack MCP server E2E smoke test는 아직 필요
- ⚠️ Graph traversal은 recall gain과 precision 하락을 교환하며 strict two-dataset holdout gate를 통과하지 못해 opt-in
- ⚠️ 최신 retrieval candidate의 production activation 전 answer-level faithfulness/citation 평가 필요

최초 Kotlin 프로젝트였으나 MCP ecosystem과 AI agent OSS의 Node/TypeScript 중심성, frontend
개발자의 접근성을 위해 TypeScript로 port했습니다.

---

## 제품 방향: self-hosted OSS와 선택적 managed deployment

이 저장소의 모든 runtime과 framework package는 Apache-2.0이며 self-host할 수 있습니다. 제품 방향은
**AI agent를 위한 통제된 지식 계층**입니다. Notion·Slack·GitHub을 연결하면 각 agent answer가
사용자 ACL을 준수하고, source Evidence를 citation하며, grounding이 없으면 답변을 거부합니다.
향후 managed deployment는 동일한 open-source runtime에 운영·관리 기능을 더할 수 있습니다.

### 핵심 제안

> 대부분의 RAG demo는 사용자가 접근할 수 없는 문서를 노출하고, 왜 그렇게 답했는지
> 설명하지 못하며, 모를 때 hallucination합니다. kontext production path는 이 세 문제를
> 해결합니다. 동일한 runtime을 직접 self-host하거나 managed deployment로 운영할 수 있습니다.

### 현재 구현과 managed deployment에 필요한 운영 계층

| Layer | 상태 | 위치 |
|---|---|---|
| Retrieval pipeline, ontology graph, pluggable retriever | ✅ | `@kontext-brain/core` Apache-2.0 |
| MCP client/server, source adapter | ✅, real-server E2E 필요 | `@kontext-brain/mcp`, `tool-server` Apache-2.0 |
| Multi-tenant KG, org RLS, ACL retrieval, Evidence, fail-closed `answer()` | ✅ | `@kontext-brain/postgres` Apache-2.0 |
| 압축 source-of-truth body store | ✅ | `@kontext-brain/object-storage` Apache-2.0 |
| Ontology proposal governance | ✅ | `@kontext-brain/github` Apache-2.0 |
| Control plane: signup, org provisioning, OAuth, metering, billing | ⬜ | 향후 service layer |
| Admin UI: source 연결, ontology/audit/ACL preview | ⬜ | 향후 service layer |
| Hosted MCP + REST agent endpoint | ⬜ | `tool-server` + `postgres` wrapper |
| SSO/SCIM, SOC2, on-prem installer | ⬜ | Enterprise phase |

---

## 라이선스

저장소와 8개 배포 package 전체를 **Apache License 2.0**으로 제공합니다. `postgres`,
`object-storage`, `github`도 포함됩니다. 라이선스 조건에 따라 상업적 사용, 수정, 배포,
self-hosting, hosted service 제공이 가능합니다. 자세한 내용은 [`LICENSE`](./LICENSE)와
[`LICENSING.md`](./LICENSING.md)를 참고하세요.
