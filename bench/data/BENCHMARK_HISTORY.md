# Benchmark history

This file indexes superseded experiments that are intentionally kept out of
the project README. Current performance, the promoted evaluation default, and
the frozen cross-framework protocol are documented in the root README and the
[current cross-framework report](./rag-eval-v2/cross-framework-all-datasets-2026-08-23.md).

These runs used different datasets, samples, models, metrics, or protocols.
They explain how the design evolved, but their headline numbers must not be
combined with or substituted for the current evaluation.

| Round | Question investigated | Detailed artifact |
|---|---|---|
| 1–6 | Small synthetic tech-doc corpus, ontology routing, compression, and early extractive variants | Preserved in Git history and benchmark source artifacts |
| 7–8 | SQuAD 2.0 retrieval and hybrid entity/vector retrieval | [SQuAD judge report](./squad-judge-report.md) |
| 9 | Attribute retrieval and structured predicates | [Round 9](./round9-report.md) |
| 10 | Attempts to push sampled SQuAD accuracy beyond the hybrid baseline | [Round 10](./round10-report.md) |
| 11 | HotpotQA multi-hop generalization failure analysis | [Round 11](./round11-report.md) |
| 12 | LLM-capacity isolation with the same retrieval contexts | [Round 12](./round12-report.md) |
| 13 | Hybrid retrieval versus vanilla vector RAG under the same answerer | [Round 13](./round13-report.md) |
| 14 | Entity-decomposed multi-hop retrieval | [Round 14](./round14-report.md) |
| 15 | First sampled GraphRAG-Bench comparison | [Round 15](./round15-report.md) |
| 16 | Confounder-reduced GraphRAG-Bench comparison | [Round 16](./round16-report.md) |
| 17 | 8B model and matched chunking comparison | [Round 17](./round17-report.md) |
| 18 | Initial ontology/KG indexing attempt and resource constraints | [Round 18](./round18-report.md) |
| 19 | LLM-extracted typed KG and the PPR retrieval regression | [Round 19](./round19-report.md) |

The authoritative current line begins with the RAG evaluation v2 harness. Its
trial registration and results are under [`bench/data/rag-eval-v2`](./rag-eval-v2/)
and [`bench/src/rag-eval-v2/TUNING_LOG.md`](../src/rag-eval-v2/TUNING_LOG.md).
