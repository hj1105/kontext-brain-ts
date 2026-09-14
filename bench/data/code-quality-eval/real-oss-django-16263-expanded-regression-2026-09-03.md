# Kontext Brain real-OSS code-quality benchmark

Generated: 2026-09-03T02:24:06.444Z

Repository: [django/django](https://github.com/django/django) at `321ecb40f4da842926e1bc07e11df4aabe53ca4b` (BSD-3-Clause).
Task: [django__django-16263](https://code.djangoproject.com/ticket/28477); upstream fix: [PR](https://github.com/django/django/pull/16263).
Evidence strength: **smoke**.

| Arm | Eligible | Success | FAIL_TO_PASS | PASS_TO_PASS | Allowed paths | Context | Tokens in/out | Duration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1/1 | 0.0% | 33.3% | 100.0% | 100.0% | n/a | 994,133/16,601 | 384004 ms |
| rag | 1/1 | 0.0% | 66.7% | 97.0% | 100.0% | n/a | 2,376,564/20,286 | 502129 ms |
| kontext | 1/1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 2,143,811/13,088 | 417464 ms |

## Ontology ingestion

The Kontext arm indexed 865 Python code Resources, 12468 symbols (8385 behavior-bearing), 6 provenance Resources, and 11 effective normative records. Targets: `BaseExpression.get_refs` (planned), `Ref.get_refs` (planned), `Q.resolve_expression` (`code-symbol:b88805f100426c08540960e3ba4af992a666905fa28fd9585391cfd336d9431f`), `refs_expression` (`code-symbol:62029df62e2ef7f7c27a342baac9d82911998e4c45905fcd70de541819fb9217`), `Query.get_aggregation` (`code-symbol:498f6d96470f308f656c8612eac8388e0c86042622e3f3bd98c29ec4e386223c`), `Query.solve_lookup_type` (`code-symbol:52c26ce9eac440df796cc6f76cefdaaf38eac3845065be01f2b9c38addbf3a0d`), `Query.build_filter` (`code-symbol:d794651b355e6b8cec3ad84a52ea197b72b605bbf43301f208fb87a430c7d168`), `Query._add_q` (`code-symbol:a025663ec4a3890006b6954d6319bdedde652a9f7bf2ad2bc210800a6285b3f2`), `WhereNode.get_refs` (planned). Governing records: `decision:django-preserve-required-aggregation-subqueries`, `decision:django-prune-unused-terminal-annotations`, `domain-term:django-alias`, `domain-term:django-annotation`, `domain-term:django-summary-aggregate`, `invariant:django-direct-aggregation-inlines-annotation-references`, `invariant:django-predicate-annotation-dependencies-survive`, `invariant:django-query-semantics-before-query-shape`, `invariant:django-subquery-mask-keeps-summary-dependencies`, `invariant:django-subquery-trigger-is-semantic`, `invariant:django-transitive-annotation-dependencies-survive`.

## Per-run evidence

| Repetition | Arm | Eligible | Success | Public | F2P | P2P | Paths | Logic context | Changed files |
|---:|---|---|---|---|---:|---:|---|---:|---|
| 1 | baseline | yes | no | pass | 1/3 | 100/100 | pass | 0/0 | `django/db/models/sql/query.py` |
| 1 | rag | yes | no | fail | 2/3 | 97/100 | pass | 0/0 | `django/db/models/expressions.py`, `django/db/models/query_utils.py`, `django/db/models/sql/query.py`, `django/db/models/sql/where.py` |
| 1 | kontext | yes | yes | pass | 3/3 | 100/100 | pass | 10/9 | `django/db/models/expressions.py`, `django/db/models/query_utils.py`, `django/db/models/sql/query.py`, `django/db/models/sql/where.py` |

## Limitations

- This run covers one real SWE-bench Verified task from one public library; it is a smoke test, not an external-validity claim.
- The task is an upstream historical replay. The agent sees the pinned pre-fix commit, while the grader alone applies the upstream regression-test patch.
- The Kontext arm receives nine ontology-linked behavior-bearing targets derived from the merged change shape; this is intentional planning guidance and gives it more structure than raw RAG, while no source patch is exposed.
- The frozen public corpus is a curated, timestamp-bounded projection of the issue, base documentation, and base code comments rather than a live repository search.
- Subscription runtime load and model nondeterminism remain uncontrolled; arm order rotates between repetitions.
