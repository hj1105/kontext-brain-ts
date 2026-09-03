# Real-OSS complex code-quality benchmark candidates

Date: 2026-09-03

## Objective

Choose historical, resolved OSS tasks that can distinguish issue-only coding from raw-source retrieval and symbol-governed Kontext context. A useful task must require non-trivial behavior across symbols, expose deterministic regression tests, and have provenance that existed before the solution without exposing the gold patch.

The candidate pool is the 500-instance official [SWE-bench Verified dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified). The project documents Verified as expert-validated problems and documents the task fields, including the base commit, problem statement, gold patch, and test patch, in its [dataset guide](https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/datasets.md). Counts below were computed from the official Parquet snapshot downloaded on 2026-09-03.

## Recommendation

Start with `django__django-16263`.

It is not the largest patch, but it is the best Kontext discrimination task: the implementation must preserve a graph of annotation references across expressions, filters, HAVING/QUALIFY handling, aggregation pushdown, and subquery selection. The underlying Django ticket accumulated several approaches over years, including an early `exists()` suggestion, a workaround using `values('pk')`, an earlier partial improvement, and a later accepted design. This gives the ontology real work: retain current decisions, mark superseded guidance, and bind the resulting invariants to the affected query-compiler symbols.

The [Django ticket #28477](https://code.djangoproject.com/ticket/28477) states that count queries should remove annotations only when filters, other annotations, or ordering do not reference them. The merged [PR #16263](https://github.com/django/django/pull/16263) changed four source files and describes the distinction between queries that still require subquery wrapping and those where annotation references can be inlined and pruned.

Frozen task facts:

- Base commit: `321ecb40f4da842926e1bc07e11df4aabe53ca4b`
- Verified difficulty: `1-4 hours`
- Source patch: 4 files, 70 additions, 23 deletions
- Tests: 3 FAIL_TO_PASS and 100 PASS_TO_PASS
- License: BSD-3-Clause; see the official [Django license](https://github.com/django/django/blob/main/LICENSE)

Suggested ontology records:

- Domain terms: annotation, annotation reference, summary aggregate, annotation mask, subquery pushdown, HAVING, QUALIFY.
- Decision: prune only annotations unreachable from requested aggregates and query predicates.
- Invariant: any annotation referenced transitively by an aggregate or predicate must survive masking.
- Invariant: pre-existing aggregate/window annotations, slicing, distinct, grouping, or combinators can still require a subquery.
- Superseded guidance: systematically wrapping every annotated aggregation in a subquery.
- Behavior-bearing symbols: expression reference discovery, lookup resolution, filter construction, where-tree reference discovery, and aggregation planning.

## Ranked alternatives

### 2. `sympy__sympy-16597` — assumptions implication graph

The initial [issue #16432](https://github.com/sympy/sympy/issues/16432) asks whether an even symbol must be finite. The discussion exposes a genuine domain-model conflict: ordinary integers should be finite, while infinite bounds and SymPy's historical meaning of `real` complicate the implication graph. The merged [PR #16597](https://github.com/sympy/sympy/pull/16597) propagated the decision through assumption rules, generated facts, power reasoning, tree output, and indexed bounds.

- Base commit: `6fd65310fa3167b9626c38a5487e171ca407d988`
- Verified difficulty: `1-4 hours`
- Source patch: 6 files, 44 additions, 26 deletions
- Tests: 3 FAIL_TO_PASS and 74 PASS_TO_PASS
- Kontext value: very high; accepted meanings of `finite`, `real`, `extended_real`, `rational`, `algebraic`, and `transcendental` must remain consistent across generated and handwritten logic.
- Risk: the 2019 environment is less convenient on a modern local Python, so the official containerized harness is preferable.

### 3. `sphinx-doc__sphinx-7590` — C++ user-defined literal parsing

The [issue #7294](https://github.com/sphinx-doc/sphinx/issues/7294) reports failure to parse C++ user-defined literals. The merged [PR #7590](https://github.com/sphinx-doc/sphinx/pull/7590) modifies shared C-family literal regexes, the C parser, and the C++ AST/parser. The difficult part is distinguishing standard numeric suffixes from user-defined suffixes and preserving parsing, rendering, symbol lookup, and ID mangling behavior.

- Base commit: `2e506c5ab457cba743bb47eb5b8c8eb9dd51d23d`
- Verified difficulty: `>4 hours`
- Source patch: 3 files, 89 additions, 18 deletions
- Tests: 1 FAIL_TO_PASS and 24 PASS_TO_PASS
- Kontext value: high for grammar rules and shared-parser invariants.
- Risk: the regression oracle is narrow, so additional property tests should cover integer, floating, hexadecimal, string, and character literals with both standard and user-defined suffixes.

### 4. `pydata__xarray-6992` — MultiIndex state transitions

The [issue #7036](https://github.com/pydata/xarray/issues/7036) exposes a broken internal relationship between `_coord_names` and `_variables` after index refactoring. The merged [PR #6992](https://github.com/pydata/xarray/pull/6992) restores several `set_index` and `reset_index` behaviors across ordinary indexes and MultiIndexes.

- Base commit: `45c0a114e2b7b27b83c9618bc05b36afac82183c`
- Verified difficulty: `>4 hours`
- Source patch: 2 files, 85 additions, 38 deletions
- Tests: 12 FAIL_TO_PASS and 945 PASS_TO_PASS
- License: Apache-2.0; see the official [xarray license](https://github.com/pydata/xarray/blob/main/LICENSE)
- Kontext value: high; coordinate membership, index ownership, dimension replacement, level reduction, ordering, and backward compatibility form a state-transition model.
- Risk: PR #6992 resolves multiple related issues, while the benchmark problem statement names only one symptom. This makes it a strong context-recovery stress test but a weaker first task for a fair product claim.

### 5. `pytest-dev__pytest-5787` — chained-exception serialization

This task changes recursive exception serialization and reconstruction in `src/_pytest/reports.py`. It must preserve explicit causes, implicit contexts, traceback chains, crash locations, and compatibility with distributed execution.

- Base commit: `955e54221008aba577ecbaefa15679f6777d3bf8`
- Verified difficulty: `1-4 hours`
- Source patch: 1 file, 143 additions, 89 deletions
- Tests: 2 FAIL_TO_PASS and 123 PASS_TO_PASS
- Primary change: [pytest PR #5787](https://github.com/pytest-dev/pytest/pull/5787)
- Kontext value: medium-high; recursive data-shape invariants matter more than file count.

## Exclusions

- `astropy__astropy-13398`: complex four-file coordinate transformation, but its problem statement includes a nearly complete implementation sketch. It would repeat the Flask benchmark's main weakness: the baseline receives most of the answer.
- `pylint-dev__pylint-4551`: four-file type-hint/UML implementation, but the official instance has no PASS_TO_PASS tests, making regression comparisons too weak.
- Live, unresolved GitHub issues: unsuitable for the first controlled experiment because there is no accepted gold patch or stable hidden regression oracle.

## Experimental controls

1. Give every arm the same frozen repository and issue statement.
2. Give raw-source RAG and Kontext the same timestamp-bounded provenance corpus. Kontext may transform it into decisions, terms, invariants, and symbol links; RAG receives the untransformed records.
3. Exclude the solution PR diff, commit messages that reveal the patch, later repository history, and the SWE-bench gold patch from all solver-visible inputs.
4. Apply only the held-out test patch after the agent exits.
5. Measure task success, FAIL_TO_PASS, PASS_TO_PASS, allowed paths, exact/semantic gold-patch similarity, number of behavior-bearing symbols consulted, time, and tokens.
6. Use at least three repetitions per arm with rotating arm order. Treat one repetition as smoke evidence only.

## Proposed sequence

1. Implement and validate `django__django-16263` first.
2. Add `sympy__sympy-16597` to exercise explicit domain semantics and cascading invariants.
3. Add `sphinx-doc__sphinx-7590` to exercise parser and grammar logic.
4. Use `pydata__xarray-6992` as a deliberately context-heavy stress test, reporting its multi-issue oracle limitation separately.

## Executed Django smoke benchmark

The first candidate was implemented and run on 2026-09-03 with Codex
`gpt-5.6-terra`, medium reasoning, one repetition per arm. Every arm received
the same isolated one-commit Django checkout. The baseline received the issue
statement; raw RAG and Kontext received the same six public provenance records,
including the accepted design in ticket comment 13. Kontext transformed that
corpus into 11 effective Decisions, Domain Terms, and Invariants linked to nine
behavior-bearing work items.

The final gate was strengthened during calibration. The official SWE-bench
oracle alone allowed a candidate that passed all 3 FAIL_TO_PASS and 100
PASS_TO_PASS tests but broke `QuerySet.datetimes().count()`. The final public
command therefore runs 821 tests across `aggregation`, `aggregation_regress`,
`annotations`, `queries`, and `lookup` before the held-out oracle is applied.
The official upstream patch passes the same 821-test gate.

| Arm | Public 821 | F2P | P2P | Allowed paths | Logic receipts | Strict success |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | pass | 1/3 | 100/100 | pass | n/a | no |
| Raw RAG | fail | 2/3 | 97/100 | pass | n/a | no |
| Kontext | pass | 3/3 | 100/100 | pass | 10 calls for 9 required | yes |

Kontext was the only arm to satisfy every strict gate in this smoke run. It
used 2,143,811 input tokens and 13,088 output tokens in 417 seconds. Raw RAG
used 2,376,564 input tokens and 20,286 output tokens in 502 seconds. Baseline
was cheaper, at 994,133 input and 16,601 output tokens in 384 seconds, but did
not implement two of the three held-out behaviors. Token counts include cached
runtime input and must not be interpreted as billable-token estimates.

Calibration also exposed a product defect: a periodic workspace observer could
overwrite a newer Logic Work Item's write binding with a stale binding when
two governed symbols shared one file. The binding store now uses a conditional
update, and a regression test proves that a stale observer cannot replace the
newer binding. The first affected Kontext run stopped after six of nine work
items; after the fix, the protocol completed all work items. The final run made
one extra `begin_logic` call while refreshing a work item, hence 10 observed
calls for nine required consultations.

This remains smoke evidence, not a broad superiority claim. It is one model,
one difficult task, and one repetition. The Kontext arm also receives the nine
target symbols derived from the historical merged change shape, which is a
deliberate planning advantage over raw RAG even though neither arm sees the
source patch. Release evidence requires at least three rotated repetitions and
the next tasks in the proposed sequence.
